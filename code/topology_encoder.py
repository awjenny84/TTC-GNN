# -*- coding: utf-8 -*-
"""
Paper-aligned topology feature encoder.

Implements:
    x_v = c_v + a_v + t_v
    h_v^(0) = W0 x_v + b0

c_v: embedding of directed in/out-degree centrality.
a_v: embedding of log-scaled posting volume for human users; bots use a
     dedicated learnable activity vector.
t_v: learnable bot/user category embedding.

The CSV preprocessing stage computes only non-learnable inputs. The MLPs and
embeddings remain inside the PyTorch model and are trained end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BOT_TYPE_ID = 0
USER_TYPE_ID = 1
SRC = "__src_uid"
DST = "__dst_uid"


def normalize_id(s):
    return s.astype(str).str.strip().str.strip("'").str.strip('"')


def read_csv(path):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


def find_uid_col(df):
    for c in ("uid", "uid_1", "seed_uid"):
        if c in df.columns:
            return c
    raise ValueError("Node CSV needs uid, uid_1, or seed_uid.")


def infer_edge_cols(df):
    for a, b in (
        ("source", "target"),
        ("src_uid", "dst_uid"),
        ("source_uid", "target_uid"),
        ("from_uid", "to_uid"),
        ("uid", "id"),
    ):
        if a in df.columns and b in df.columns:
            return a, b
    raise ValueError("Cannot infer source/target columns from edge CSV.")


def merge_node_csvs(primary_nodes_csv: str,
                    extra_nodes_csvs: Optional[Sequence[str]] = None) -> pd.DataFrame:
    frames = []
    paths = [primary_nodes_csv] + list(extra_nodes_csvs or [])
    for order, path in enumerate(paths):
        df = read_csv(path).copy()
        uid_col = find_uid_col(df)
        if uid_col != "uid":
            if "uid" in df.columns:
                df["uid"] = df["uid"].where(df["uid"].notna(), df[uid_col])
            else:
                df = df.rename(columns={uid_col: "uid"})
        df["uid"] = normalize_id(df["uid"])
        df = df[(df["uid"] != "") & (df["uid"].str.lower() != "nan")].copy()
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df["__order"] = order
        df["__has_type"] = df["node_type"].notna().astype(int) if "node_type" in df.columns else 0
        df["__nonnull"] = df.notna().sum(axis=1)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    dup = len(merged) - merged["uid"].nunique()
    merged = merged.sort_values(
        ["uid", "__has_type", "__order", "__nonnull"],
        ascending=[True, False, True, False],
        kind="mergesort",
    )
    nodes = merged.groupby("uid", as_index=False, sort=False).first()
    nodes = nodes.drop(columns=["__order", "__has_type", "__nonnull"], errors="ignore")
    nodes = nodes.reset_index(drop=True)
    print(f"[INFO] merged nodes={len(nodes)} collapsed_duplicate_uid_rows={dup}")
    return nodes


def merge_edge_csvs(edge_csvs: Sequence[str]) -> pd.DataFrame:
    frames = []
    for path in edge_csvs:
        df = read_csv(path)
        a, b = infer_edge_cols(df)
        e = df[[a, b]].copy()
        e.columns = [SRC, DST]
        e[SRC] = normalize_id(e[SRC])
        e[DST] = normalize_id(e[DST])
        e = e[(e[SRC] != "") & (e[DST] != "")]
        frames.append(e)
    edges = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"[INFO] merged directed edges={len(edges)}")
    return edges


def infer_node_types(nodes: pd.DataFrame,
                     bot_uids: Optional[Iterable[str]] = None) -> np.ndarray:
    ids = np.full(len(nodes), USER_TYPE_ID, dtype=np.int64)
    uid = normalize_id(nodes["uid"])
    known_bots = {str(x).strip().strip("'").strip('"') for x in (bot_uids or [])}

    resolved = np.zeros(len(nodes), dtype=bool)
    if known_bots:
        m = uid.isin(known_bots).to_numpy()
        ids[m] = BOT_TYPE_ID
        resolved[m] = True

    if "node_type" in nodes.columns:
        raw = nodes["node_type"].fillna("").astype(str).str.strip().str.lower()
        bot_tokens = {"ai", "bot", "social_bot", "social bot", "seed"}
        user_tokens = {"user", "human", "human_user", "human user"}
        mb = raw.isin(bot_tokens).to_numpy() & ~resolved
        mu = raw.isin(user_tokens).to_numpy() & ~resolved
        ids[mb] = BOT_TYPE_ID
        ids[mu] = USER_TYPE_ID
        resolved |= mb | mu

    if "seed_uid" in nodes.columns:
        ms = (uid == normalize_id(nodes["seed_uid"])).to_numpy() & ~resolved
        ids[ms] = BOT_TYPE_ID
        resolved |= ms

    if "depth" in nodes.columns:
        depth = pd.to_numeric(nodes["depth"], errors="coerce")
        md = (depth == 0).fillna(False).to_numpy() & ~resolved
        ids[md] = BOT_TYPE_ID
        resolved |= md

    print(f"[INFO] node types: bot={(ids == BOT_TYPE_ID).sum()} user={(ids == USER_TYPE_ID).sum()}")
    return ids


def build_edge_index(nodes: pd.DataFrame, edges: pd.DataFrame):
    uid_to_idx = {u: i for i, u in enumerate(normalize_id(nodes["uid"]).tolist())}
    missing = sorted((set(edges[SRC]) | set(edges[DST])) - set(uid_to_idx))
    if missing:
        raise ValueError(f"Edges contain UIDs absent from nodes, e.g. {missing[:10]}")

    src = edges[SRC].map(uid_to_idx).to_numpy(dtype=np.int64)
    dst = edges[DST].map(uid_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.vstack([src, dst]) if len(edges) else np.zeros((2, 0), dtype=np.int64)

    out_deg = np.bincount(src, minlength=len(nodes)).astype(np.float32)
    in_deg = np.bincount(dst, minlength=len(nodes)).astype(np.float32)
    return edge_index, in_deg, out_deg


def centrality_features(in_deg, out_deg, n):
    den = float(max(1, n - 1))
    x = np.stack([in_deg / den, out_deg / den], axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Centrality features contain NaN/Inf.")
    return x


def activity_features(nodes: pd.DataFrame, type_ids: np.ndarray):
    if "statuses_count" not in nodes.columns:
        raise ValueError("statuses_count is required for activity encoding.")

    s = pd.to_numeric(nodes["statuses_count"], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).clip(lower=0)
    user_mask = type_ids == USER_TYPE_ID

    med = float(s[user_mask].median()) if s[user_mask].notna().any() else 0.0
    s = s.fillna(med)
    log_s = np.log1p(s.to_numpy(dtype=np.float64))

    user_log = log_s[user_mask]
    mean = float(user_log.mean()) if user_log.size else 0.0
    std = float(user_log.std(ddof=0)) if user_log.size else 1.0
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    z = ((log_s - mean) / std).astype(np.float32).reshape(-1, 1)
    z[~user_mask] = 0.0  # ignored for bots by the learnable encoder
    if not np.isfinite(z).all():
        raise ValueError("Activity features contain NaN/Inf.")
    print(f"[INFO] activity log1p z-score: mean={mean:.6f} std={std:.6f}")
    return z, mean, std


@dataclass
class TopologyGraphData:
    nodes: pd.DataFrame
    edge_index: np.ndarray
    centrality_features: np.ndarray
    activity_features: np.ndarray
    node_type_ids: np.ndarray
    activity_mean: float
    activity_std: float


def build_graph_from_csv(primary_nodes_csv: str,
                         edge_csvs: Sequence[str],
                         extra_nodes_csvs: Optional[Sequence[str]] = None,
                         bot_uids: Optional[Iterable[str]] = None) -> TopologyGraphData:
    """
    Build paper-aligned, non-learnable topology inputs.
    """
    nodes = merge_node_csvs(primary_nodes_csv, extra_nodes_csvs)
    edges = merge_edge_csvs(edge_csvs)

    # Include edge-only nodes if necessary.
    node_set = set(normalize_id(nodes["uid"]))
    edge_set = set(edges[SRC]) | set(edges[DST])
    missing = sorted(edge_set - node_set)
    if missing:
        nodes = pd.concat([nodes, pd.DataFrame({"uid": missing})],
                          ignore_index=True, sort=False)
        print(f"[WARN] appended {len(missing)} edge-only nodes")

    type_ids = infer_node_types(nodes, bot_uids)
    edge_index, in_deg, out_deg = build_edge_index(nodes, edges)
    cent = centrality_features(in_deg, out_deg, len(nodes))
    act, mean, std = activity_features(nodes, type_ids)

    print(
        f"[INFO] graph ready: nodes={len(nodes)} edges={edge_index.shape[1]} "
        f"centrality={cent.shape} activity={act.shape}"
    )
    return TopologyGraphData(
        nodes=nodes.reset_index(drop=True),
        edge_index=edge_index.astype(np.int64),
        centrality_features=cent,
        activity_features=act,
        node_type_ids=type_ids,
        activity_mean=mean,
        activity_std=std,
    )


class TopologyFeatureEncoder(nn.Module):
    """
    Learnable implementation of:
        c_v = f_c([in-centrality, out-centrality])
        a_v = f_a(activity) for users, a_bot for bots
        t_v = Embedding(type)
        x_v = c_v + a_v + t_v
        h0  = W0 x_v + b0
    """
    def __init__(self, graph_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.centrality_encoder = nn.Sequential(
            nn.Linear(2, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.activity_encoder = nn.Sequential(
            nn.Linear(1, graph_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bot_activity = nn.Parameter(torch.empty(graph_dim))
        nn.init.normal_(self.bot_activity, mean=0.0, std=0.02)

        self.type_embedding = nn.Embedding(2, graph_dim)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)

        self.input_projection = nn.Linear(graph_dim, graph_dim)

    def forward(self, centrality, activity, node_type_ids, return_components=False):
        if not torch.isfinite(centrality).all():
            raise ValueError("centrality contains NaN/Inf")
        if not torch.isfinite(activity).all():
            raise ValueError("activity contains NaN/Inf")

        c_v = self.centrality_encoder(centrality)
        user_a = self.activity_encoder(activity)
        bot_a = self.bot_activity.unsqueeze(0).expand_as(user_a)
        a_v = torch.where(node_type_ids.eq(BOT_TYPE_ID).unsqueeze(1), bot_a, user_a)
        t_v = self.type_embedding(node_type_ids)

        x_v = c_v + a_v + t_v
        h0 = self.input_projection(x_v)

        if not torch.isfinite(h0).all():
            raise RuntimeError("TopologyFeatureEncoder produced NaN/Inf")

        if return_components:
            return h0, {"c_v": c_v, "a_v": a_v, "t_v": t_v, "x_v": x_v}
        return h0


def save_preprocessed(data: TopologyGraphData, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "centrality_features.npy"), data.centrality_features)
    np.save(os.path.join(out_dir, "activity_features.npy"), data.activity_features)
    np.save(os.path.join(out_dir, "node_type_ids.npy"), data.node_type_ids)
    np.save(os.path.join(out_dir, "edge_index.npy"), data.edge_index)

    df = data.nodes.copy()
    df["node_type_id"] = data.node_type_ids
    df["in_degree_centrality"] = data.centrality_features[:, 0]
    df["out_degree_centrality"] = data.centrality_features[:, 1]
    df["activity_z"] = data.activity_features[:, 0]
    df.to_csv(os.path.join(out_dir, "nodes_topology_encoded.csv"),
              index=False, encoding="utf-8-sig")

    with open(os.path.join(out_dir, "graph_feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "num_nodes": len(data.nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "node_type_mapping": {"bot": BOT_TYPE_ID, "user": USER_TYPE_ID},
            "centrality": ["in_degree/(N-1)", "out_degree/(N-1)"],
            "activity": "zscore(log1p(statuses_count)) for human users",
            "activity_mean": data.activity_mean,
            "activity_std": data.activity_std,
            "excluded_direct_features": [
                "followers_count", "friends_count", "verified", "depth",
                "norm_followers_count", "norm_statuses_count"
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved to {out_dir}")


def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes-csv", required=True)
    p.add_argument("--extra-nodes-csv", default="")
    p.add_argument("--edges-csv", required=True)
    p.add_argument("--bot-uids", default="")
    p.add_argument("--output-dir", default="topology_preprocessed")
    args = p.parse_args()

    data = build_graph_from_csv(
        primary_nodes_csv=os.path.abspath(args.nodes_csv),
        extra_nodes_csvs=[os.path.abspath(x) for x in split_csv_arg(args.extra_nodes_csv)],
        edge_csvs=[os.path.abspath(x) for x in split_csv_arg(args.edges_csv)],
        bot_uids=split_csv_arg(args.bot_uids) or None,
    )
    save_preprocessed(data, os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
