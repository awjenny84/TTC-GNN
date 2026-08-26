# -*- coding: utf-8 -*-
"""
End-to-end 鍙備笌搴﹀绫诲埆棰勬祴锛歍ext Encoder + Graph Encoder(鍙€?GT / GraphSAGE) + 铻嶅悎锛圖ual-Co-Attention锛?骞跺彲閫夊姞鍏モ€滃浘鑷洃鐫ｆ崯澶扁€濓紙闅忔満娓歌蛋 + 璐熼噰鏍凤級鏉ュ鐜颁綘鎴浘閲岀殑 GraphSAGE 鏃犵洃鐫ｇ洰鏍囥€?
鐢ㄦ硶绀轰緥锛?  # 绾鍒扮锛堝彧鐢ㄥ弬涓庡害鐩戠潱锛屼笉鍔犲浘鑷洃鐫ｏ級
  conda run python train_multiclass_engagement_e2e.py --lambda_unsup 0

  # 绔埌绔?+ 鍥捐嚜鐩戠潱锛堝鐜版埅鍥惧叕寮忔€濇兂锛歳andom-walk positives + negative sampling锛?  conda run python train_multiclass_engagement_e2e.py --lambda_unsup 0.1 --unsup_every 1 --graph_encoder sage

娉ㄦ剰锛?- 绔埌绔笉鏄€滃垎寮€璁粌涓や釜妯″瀷鈥濓紝鑰屾槸涓€涓暣浣撴ā鍨嬮噷鏈変袱涓彲瀛︿範瀛愭ā鍧楋紙text encoder / graph encoder锛夛紝
  鐢ㄥ悓涓€涓?loss锛堟垨 loss 鐨勫姞鏉冨拰锛変竴娆″弽浼犲悓鏃舵洿鏂板弬鏁般€?- 缁撴瀯鍚戦噺鍙栨硶锛氱敤 role_id 鏄犲皠鍒板浘涓€滄櫤鑳戒綋鑺傜偣鈥濈殑 node_index锛屽啀鍘绘嬁璇ヨ妭鐐圭殑 embedding銆?  浣犻渶瑕佷繚璇?CSV 鐨?role_id 鍜?nodes.csv 鐨?role_id 鑳藉榻愶紙鏈€鐪佷簨锛氫袱杈归兘鏄?0..7锛夈€?"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from torch_geometric.nn import GATConv, SAGEConv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

DEFAULT_PROFILE_CSV_NAME = "bot_profile_descriptions.csv"
BOT_DESCRIPTION_TEXT_COL = "bot_description"
BOT_DESCRIPTION_ID_COL = "bot_description_id"
DEFAULT_EXTRA_NODE_CSV_NAMES = (
    "fan_bfs_nodes.csv",
    "fans_bfs_nodes.csv",
    "fan_nodes3.csv",
    "fan_bfs_nodes3.csv",
    "fans_bfs_nodes3.csv",
)
DEFAULT_EXTRA_EDGE_CSV_NAMES = (
    "fan_bfs_edges.csv",
    "fans_bfs_edges.csv",
    "fan_edges3.csv",
    "fan_bfs_edges3.csv",
    "fans_bfs_edges3.csv",
)

# -------------------------
# 0) Utils
# -------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"acc": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def normalize_id(series):
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def is_disabled_path_list(value) -> bool:
    return bool(value) and str(value).strip().lower() in {"none", "null", "-"}


def split_path_list(value) -> List[str]:
    if not value or is_disabled_path_list(value):
        return []
    paths = []
    for path in str(value).replace(";", ",").split(","):
        path = path.strip().strip("'").strip('"')
        if path:
            paths.append(path)
    return paths


def resolve_relative_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def read_csv_with_fallback(path: str, encodings: List[str], **kwargs) -> pd.DataFrame:
    tried = []
    for encoding in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            tried.append(encoding)
    raise UnicodeDecodeError(
        encodings[0],
        b"",
        0,
        1,
        "Could not decode CSV {} with encodings: {}".format(path, tried),
    )


def find_uid_column(frame: pd.DataFrame, context: str) -> str:
    for column in ("uid", "uid_1", "seed_uid"):
        if column in frame.columns:
            return column
    raise ValueError("{} must contain a uid-like column: uid, uid_1, or seed_uid.".format(context))


def resolve_extra_nodes_csvs(args) -> List[str]:
    explicit_paths = split_path_list(getattr(args, "extra_nodes_csv", ""))
    base_dir = os.path.dirname(os.path.abspath(args.nodes_csv)) or "."
    primary = os.path.abspath(args.nodes_csv)

    if explicit_paths:
        paths = [resolve_relative_path(path, base_dir) for path in explicit_paths]
    else:
        paths = [
            os.path.abspath(os.path.join(base_dir, name))
            for name in DEFAULT_EXTRA_NODE_CSV_NAMES
            if os.path.isfile(os.path.join(base_dir, name))
        ]

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Cannot find extra nodes CSV(s): {}".format(missing))
    return [path for path in dict.fromkeys(paths) if path != primary]


def resolve_extra_edges_csvs(args) -> List[str]:
    explicit_paths = split_path_list(getattr(args, "extra_edges_csv", ""))
    base_dir = os.path.dirname(os.path.abspath(args.edges_csv)) or "."
    primary = os.path.abspath(args.edges_csv)

    if explicit_paths:
        paths = [resolve_relative_path(path, base_dir) for path in explicit_paths]
    else:
        paths = [
            os.path.abspath(os.path.join(base_dir, name))
            for name in DEFAULT_EXTRA_EDGE_CSV_NAMES
            if os.path.isfile(os.path.join(base_dir, name))
        ]

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Cannot find extra edges CSV(s): {}".format(missing))
    return [path for path in dict.fromkeys(paths) if path != primary]


def read_nodes_csvs(args) -> pd.DataFrame:
    node_paths = [os.path.abspath(args.nodes_csv)] + resolve_extra_nodes_csvs(args)
    frames = []
    for source_order, path in enumerate(node_paths):
        frame = read_csv_with_fallback(
            path,
            [args.nodes_encoding, "utf-8-sig", "utf-8", "gbk"],
            dtype=str,
        ).copy()
        uid_col = find_uid_column(frame, path)
        if uid_col != "uid":
            frame = frame.rename(columns={uid_col: "uid"})
        frame["uid"] = normalize_id(frame["uid"])
        frame = frame[frame["uid"] != ""].copy()
        frame["__source_order"] = source_order
        frame["__source_path"] = path
        frame["__row_order"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)

    nodes = pd.concat(frames, ignore_index=True, sort=False)
    before = len(nodes)
    nodes = (
        nodes.sort_values(["__source_order", "__row_order"], kind="mergesort")
        .drop_duplicates("uid", keep="first")
        .reset_index(drop=True)
    )
    print(
        "[INFO] Loaded node CSVs: {} -> merged_nodes={} duplicate_uid_rows_collapsed={}".format(
            ", ".join("{}({})".format(os.path.basename(path), len(frame)) for path, frame in zip(node_paths, frames)),
            len(nodes),
            before - len(nodes),
        )
    )
    return nodes


def find_edge_columns(edges: pd.DataFrame, context: str) -> Tuple[str, str]:
    candidates = [
        ("source", "target"),
        ("src_uid", "dst_uid"),
        ("src", "dst"),
        ("from_uid", "to_uid"),
    ]
    for src_col, dst_col in candidates:
        if src_col in edges.columns and dst_col in edges.columns:
            return src_col, dst_col
    raise ValueError("{} must contain edge columns such as source/target or src_uid/dst_uid.".format(context))


def read_edges_csvs(args, uid_to_node: Dict[str, int]) -> np.ndarray:
    edge_paths = [os.path.abspath(args.edges_csv)] + resolve_extra_edges_csvs(args)
    pieces = []
    total_rows = 0
    dropped_rows = 0
    for path in edge_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError("Cannot find edges CSV: {}".format(path))
        edges = read_csv_with_fallback(
            path,
            [args.edges_encoding, "utf-8-sig", "utf-8", "gbk"],
            dtype=str,
        )
        src_col, dst_col = find_edge_columns(edges, path)
        total_rows += len(edges)
        src = normalize_id(edges[src_col])
        dst = normalize_id(edges[dst_col])
        mask = src.isin(uid_to_node) & dst.isin(uid_to_node)
        dropped_rows += int((~mask).sum())
        if mask.any():
            row = src[mask].map(uid_to_node).to_numpy(dtype=np.int64)
            col = dst[mask].map(uid_to_node).to_numpy(dtype=np.int64)
            pieces.append(np.vstack([row, col]))

    if pieces:
        edge_index = np.concatenate(pieces, axis=1)
        edge_index = np.unique(edge_index.T, axis=0).T.astype(np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    print(
        "[INFO] Loaded edge CSVs: {} total_rows={} kept_edges={} dropped_unknown_uid_edges={}".format(
            ", ".join(os.path.basename(path) for path in edge_paths),
            total_rows,
            edge_index.shape[1],
            dropped_rows,
        )
    )
    return edge_index


def build_graph_inputs_from_csv(args) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    nodes = read_nodes_csvs(args)
    required_cols = ["followers_count", "statuses_count", "verified"]
    missing = [column for column in required_cols if column not in nodes.columns]
    if missing:
        raise ValueError("Merged nodes CSVs are missing graph feature columns: {}".format(missing))

    numeric = nodes[["followers_count", "statuses_count"]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(numeric_only=True)
    numeric = numeric.fillna(medians).fillna(0.0)

    verified = nodes["verified"].fillna("").astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    raw_features = np.column_stack(
        [
            np.log1p(np.maximum(numeric["followers_count"].to_numpy(dtype=float), 0.0)),
            np.log1p(np.maximum(numeric["statuses_count"].to_numpy(dtype=float), 0.0)),
            verified.to_numpy(dtype=float),
        ]
    )
    graph_x = StandardScaler().fit_transform(raw_features).astype(np.float32)
    uid_to_node = {uid: idx for idx, uid in enumerate(nodes["uid"].astype(str).tolist())}
    edge_index = read_edges_csvs(args, uid_to_node)
    print("[INFO] Built graph inputs from CSVs: graph_x={} edge_index={}".format(graph_x.shape, edge_index.shape))
    return nodes, graph_x, edge_index


def resolve_profile_csv(args) -> str:
    raw_path = getattr(args, "profile_csv", DEFAULT_PROFILE_CSV_NAME)
    if os.path.isabs(raw_path):
        return raw_path

    candidates = [os.path.abspath(raw_path)]
    for base_path in (getattr(args, "csv_path", ""), getattr(args, "nodes_csv", ""), __file__):
        base_dir = os.path.dirname(os.path.abspath(base_path)) if base_path else ""
        if base_dir:
            candidates.append(os.path.abspath(os.path.join(base_dir, raw_path)))

    for path in dict.fromkeys(candidates):
        if os.path.isfile(path):
            return path
    return candidates[0]


def load_bot_profile_descriptions(args):
    profile_csv = resolve_profile_csv(args)
    encodings = [args.profile_encoding, "utf-8-sig", "utf-8", "gbk"]
    profiles = read_csv_with_fallback(profile_csv, encodings)
    uid_col = find_uid_column(profiles, profile_csv)
    desc_col = args.profile_desc_col
    if desc_col not in profiles.columns:
        fallback_cols = [
            column
            for column in ("role_description", "description", "desc_text")
            if column in profiles.columns
        ]
        if not fallback_cols:
            raise ValueError("{} must contain a bot description column.".format(profile_csv))
        desc_col = fallback_cols[0]

    profiles = profiles[[uid_col, desc_col]].copy()
    profiles.columns = ["uid", "__bot_description"]
    profiles["uid"] = normalize_id(profiles["uid"])
    profiles["__bot_description"] = profiles["__bot_description"].fillna("").astype(str).str.strip()
    profiles = profiles[(profiles["uid"] != "") & (profiles["__bot_description"] != "")]
    if profiles.empty:
        raise ValueError("{} has no usable uid/description rows.".format(profile_csv))

    conflict = profiles.groupby("uid")["__bot_description"].nunique()
    conflict = conflict[conflict > 1]
    if not conflict.empty:
        raise ValueError(
            "{} has conflicting descriptions for uid(s): {}".format(
                profile_csv, conflict.index.astype(str).tolist()[:5]
            )
        )

    profiles = profiles.drop_duplicates("uid", keep="first").reset_index(drop=True)
    uid_to_description = {}
    uid_to_description_id = {}
    id_to_description = {}
    for description_id, row in profiles.iterrows():
        uid = row["uid"]
        description = row["__bot_description"]
        uid_to_description[uid] = description
        uid_to_description_id[uid] = int(description_id)
        id_to_description[str(description_id)] = description

    print(
        "[INFO] Loaded bot profile descriptions: profiles={} unique_descriptions={} file={}".format(
            len(profiles), profiles["__bot_description"].nunique(), profile_csv
        )
    )
    return profile_csv, uid_to_description, uid_to_description_id, id_to_description


def build_chronological_per_bot_split(args) -> pd.DataFrame:
    """Use the same per-bot chronological 8/1/1 post split as the main experiment."""
    features = read_csv_with_fallback(
        args.csv_path,
        [args.csv_encoding, "utf-8-sig", "utf-8", "gbk"],
        dtype={"uid": str, "wid": str},
    )
    required = {"uid", "wid", args.label_col, args.time_col}
    missing = required - set(features.columns)
    if missing:
        raise ValueError("{} is missing columns {}".format(args.csv_path, sorted(missing)))

    source = features[["uid", "wid", args.label_col, args.time_col]].copy()
    source["uid"] = normalize_id(source["uid"])
    source["wid"] = normalize_id(source["wid"])
    source = source[(source["uid"] != "") & (source["wid"] != "")].copy()
    source["__time"] = pd.to_datetime(source[args.time_col], errors="coerce")

    bad_time = source["__time"].isna()
    if bad_time.any():
        examples = source.loc[bad_time, ["uid", "wid", args.time_col]].head().to_dict("records")
        raise ValueError(
            "Invalid or missing timestamps in '{}'; examples: {}".format(
                args.time_col, examples
            )
        )

    duplicate_mask = source.duplicated(["uid", "wid", args.label_col], keep="first")
    if duplicate_mask.any():
        print(
            "[WARN] {}: dropped {} exact duplicate post rows before splitting.".format(
                args.csv_path, int(duplicate_mask.sum())
            )
        )
        source = source.loc[~duplicate_mask].copy()

    conflicting = source.duplicated(["uid", "wid"], keep=False)
    if conflicting.any():
        examples = source.loc[conflicting, ["uid", "wid", args.label_col]].head().to_dict("records")
        raise ValueError("Duplicate (uid, wid) rows with conflicting labels: {}".format(examples))

    parts = []
    bot_stats = []
    for uid, group in source.groupby("uid", sort=True):
        group = (
            group.copy()
            .sort_values(["__time", "wid"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )
        n = len(group)
        if n < 3:
            raise ValueError(
                "Bot {} has only {} posts; at least 3 are required for an 8/1/1 split.".format(
                    uid, n
                )
            )

        n_train = int(np.floor(0.8 * n))
        n_val = int(np.floor(0.1 * n))
        n_test = n - n_train - n_val
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        if n_train <= 0:
            raise ValueError("Bot {} does not have enough posts for non-empty 8/1/1 splits.".format(uid))

        split_indices = {
            "train": np.arange(0, n_train),
            "val": np.arange(n_train, n_train + n_val),
            "test": np.arange(n_train + n_val, n),
        }
        for split_name in ("train", "val", "test"):
            piece = group.iloc[split_indices[split_name]].copy()
            piece["__split"] = split_name
            parts.append(piece)

        train_last = group.iloc[n_train - 1]["__time"]
        val_first = group.iloc[n_train]["__time"]
        val_last = group.iloc[n_train + n_val - 1]["__time"]
        test_first = group.iloc[n_train + n_val]["__time"]
        if not (train_last <= val_first <= val_last <= test_first):
            raise AssertionError("Chronological split order failed for bot {}.".format(uid))

        bot_stats.append((uid, n, n_train, n_val, n_test, group.iloc[0]["__time"], group.iloc[-1]["__time"]))

    membership = pd.concat(parts, ignore_index=True)
    split_post_sets = {
        split_name: set(
            zip(
                membership.loc[membership["__split"] == split_name, "uid"],
                membership.loc[membership["__split"] == split_name, "wid"],
            )
        )
        for split_name in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_post_sets[left] & split_post_sets[right]
        if overlap:
            raise AssertionError("Post leakage between {} and {}: {}".format(left, right, list(overlap)[:5]))

    features = features.copy()
    features["uid"] = normalize_id(features["uid"])
    features["wid"] = normalize_id(features["wid"])
    exact_feature_duplicates = features.duplicated(keep="first")
    if exact_feature_duplicates.any():
        print(
            "[WARN] {}: dropped {} exact duplicate feature rows.".format(
                args.csv_path, int(exact_feature_duplicates.sum())
            )
        )
        features = features.loc[~exact_feature_duplicates].copy()
    duplicated_keys = features.duplicated(["uid", "wid"], keep=False)
    if duplicated_keys.any():
        examples = features.loc[duplicated_keys, ["uid", "wid"]].head().to_dict("records")
        raise ValueError(
            "Feature CSV still has duplicate (uid, wid) keys after exact-row deduplication: {}".format(
                examples
            )
        )

    membership = membership.drop(columns=[args.time_col], errors="ignore")
    features = features.drop(columns=[args.label_col], errors="ignore")
    merged = membership.merge(features, on=["uid", "wid"], how="inner", validate="one_to_one")
    if len(merged) != len(membership):
        membership_keys = set(zip(membership["uid"], membership["wid"]))
        feature_keys = set(zip(features["uid"], features["wid"]))
        raise ValueError(
            "Some split posts cannot be found in the feature CSV: missing_in_features={}, "
            "extra_feature_rows={}.".format(
                len(membership_keys - feature_keys),
                len(feature_keys - membership_keys),
            )
        )

    print(
        "[INFO] Generated chronological per-bot post split using '{}': "
        "earliest 80% train / next 10% val / latest 10% test".format(args.time_col)
    )
    for split_name in ("train", "val", "test"):
        subset = merged[merged["__split"] == split_name]
        print("[INFO] {}: bots={} posts={}".format(split_name, subset["uid"].nunique(), len(subset)))
    print("[INFO] Same bot identities are shared across train/val/test; posts are mutually disjoint.")
    print(
        "[INFO] Per-bot split examples "
        "(uid,total,train,val,test,earliest,latest): {}".format(bot_stats[:5])
    )
    return merged


def attach_bot_profile_descriptions(df: pd.DataFrame, args):
    profile_csv, uid_to_description, uid_to_description_id, id_to_description = load_bot_profile_descriptions(args)
    df = df.copy()
    df["uid"] = normalize_id(df["uid"])
    missing_uids = sorted(set(df["uid"].unique()) - set(uid_to_description))
    if missing_uids:
        raise ValueError(
            "{} is missing bot descriptions for dataset uid(s): {}".format(
                profile_csv, missing_uids[:10]
            )
        )

    df[args.bot_description_col] = df["uid"].map(uid_to_description)
    df[args.bot_description_id_col] = df["uid"].map(uid_to_description_id).astype(int)
    print(
        "[INFO] Using bot profile description as text input: text_col={} id_col={} used_bots={}".format(
            args.bot_description_col,
            args.bot_description_id_col,
            df[args.bot_description_id_col].nunique(),
        )
    )
    return df, uid_to_description_id, id_to_description


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """edge_index: [2, E] (torch.long, CPU)
    杩斿洖姣忎釜鑺傜偣鐨勯偦灞呭垪琛紙褰撲綔鏃犲悜鍥撅細鍙屽悜鍔犲叆锛?    """
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def random_walk_one(start: int, adj: List[List[int]], walk_len: int, rng: np.random.Generator) -> List[int]:
    """Run one random walk from start and return visited nodes."""
    walk = [start]
    cur = start
    for _ in range(walk_len):
        neigh = adj[cur]
        if len(neigh) == 0:
            break
        cur = int(rng.choice(neigh))
        walk.append(cur)
    return walk


def sample_rw_positives(
    roots: torch.Tensor,
    adj: List[List[int]],
    walk_len: int,
    num_pos: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """瀵规瘡涓?root 鍋氫竴娆?random walk锛屼粠璁块棶鍒扮殑鑺傜偣閲岄噰鏍?num_pos 涓?positive context
    杩斿洖 pos_nodes: [B, num_pos]
    """
    roots_np = roots.detach().cpu().numpy().tolist()
    pos = []
    for r in roots_np:
        walk = random_walk_one(int(r), adj, walk_len, rng)
        ctx = [v for v in walk[1:]] if len(walk) > 1 else []
        if len(ctx) == 0:
            ctx = [int(r)]
        chosen = rng.choice(ctx, size=num_pos, replace=True).tolist()
        pos.append(chosen)
    return torch.tensor(pos, dtype=torch.long, device=roots.device)


def unsup_rw_neg_sampling_loss(
    z: torch.Tensor,
    roots: torch.Tensor,
    pos_nodes: torch.Tensor,
    num_neg: int,
) -> torch.Tensor:
    """澶嶇幇鎴浘閲岀殑鏃犵洃鐫ｇ洰鏍囷紙random-walk positives + negative sampling锛夛細
    L = - E_r [ sum_{vp in P_r} log 蟽(z_r^T z_vp) + sum_{vn in N_r} log 蟽(- z_r^T z_vn) ]
    杩欓噷鐢?mean 杩戜技銆?    """
    device = z.device
    B, P = pos_nodes.shape
    N = z.size(0)

    z_r = z[roots]  # [B, D]
    z_p = z[pos_nodes]  # [B, P, D]
    pos_score = (z_r.unsqueeze(1) * z_p).sum(dim=-1)  # [B, P]
    pos_loss = -F.logsigmoid(pos_score).mean()

    neg_nodes = torch.randint(low=0, high=N, size=(B, num_neg), device=device)
    z_n = z[neg_nodes]  # [B, K, D]
    neg_score = (z_r.unsqueeze(1) * z_n).sum(dim=-1)  # [B, K]
    neg_loss = -F.logsigmoid(-neg_score).mean()

    return pos_loss + neg_loss


# -------------------------
# 1) Dataset
# -------------------------


class WeiboTensorDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        role_ids: torch.Tensor,
        style_feats: torch.Tensor,
        labels: torch.Tensor,
        tokenizer,
        max_len: int = 128,
    ):
        self.texts = texts
        self.role_ids = role_ids
        self.style_feats = style_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tok = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "role_id": self.role_ids[idx],
            "style_vec": self.style_feats[idx],
            "label": self.labels[idx],
        }


# -------------------------
# 2) Text Encoder
# -------------------------


class StyleOnlyFiLMTextEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        style_in_dim: int,
        style_proj_dim: int = 64,
        film_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size

        self.style_mlp = nn.Sequential(
            nn.Linear(style_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, style_proj_dim),
            nn.ReLU(),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(style_proj_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden * 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, style_vec):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state  # [B, L, hidden]

        z_style = self.style_mlp(style_vec)
        gamma_beta = self.film_gen(z_style)
        Hdim = H.size(-1)
        gamma, beta = gamma_beta[:, :Hdim], gamma_beta[:, Hdim:]
        H_tilde = gamma.unsqueeze(1) * H + beta.unsqueeze(1)
        return self.dropout(H_tilde)


# -------------------------
# 3) Graph Encoders
# -------------------------


class GraphTransformerEncoder(nn.Module):
    """GATConv 鍫嗗彔 + FFN + LN锛屼綔涓虹粨鏋?encoder锛岃緭鍑烘墍鏈夎妭鐐?embedding"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.dropout = dropout


        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
            self.ln1.append(nn.LayerNorm(hidden_dim))
            self.ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            )
            self.ln2.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for conv, ln1, ffn, ln2 in zip(self.convs, self.ln1, self.ffn, self.ln2):
            out = conv(x, edge_index)
            h = F.gelu(out)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = ln1(x + h)
            x = ln2(x + ffn(x))
        return x


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -------------------------
# 4) Fusion + classifier
# -------------------------


class DualCoAttentionClassifier(nn.Module):
    def __init__(self, dim_text: int, dim_graph: int, num_classes: int, d_att: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_t = nn.Linear(dim_text, d_att)
        self.kg = nn.Linear(dim_graph, d_att)
        self.vg = nn.Linear(dim_graph, d_att)

        self.q_g = nn.Linear(dim_graph, d_att)
        self.kt = nn.Linear(dim_text, d_att)
        self.vt = nn.Linear(dim_text, d_att)

        self.att_t2g = nn.MultiheadAttention(d_att, nhead, dropout=dropout, batch_first=True)
        self.att_g2t = nn.MultiheadAttention(d_att, nhead, dropout=dropout, batch_first=True)

        self.ln_t = nn.LayerNorm(d_att)
        self.ln_g = nn.LayerNorm(d_att)

        self.pred = nn.Sequential(
            nn.Linear(2 * d_att, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, H_text: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
        if g_vec.dim() == 2:
            G = g_vec.unsqueeze(1)
        else:
            G = g_vec

        # Text -> Graph
        H_t2g, _ = self.att_t2g(self.q_t(H_text), self.kg(G), self.vg(G), need_weights=False)
        H_t2g = self.ln_t(H_t2g)

        # Graph -> Text
        G_g2t, _ = self.att_g2t(self.q_g(G), self.kt(H_text), self.vt(H_text), need_weights=False)
        G_g2t = self.ln_g(G_g2t)

        fused = torch.cat([H_t2g[:, 0, :], G_g2t.mean(dim=1)], dim=-1)
        return self.pred(fused)


# -------------------------
# 5) E2E wrapper
# -------------------------


class EngagementE2EModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        graph_in_dim: int,
        num_classes: int,
        graph_hidden: int = 64,
    ):
        super().__init__()
        self.text_enc = StyleOnlyFiLMTextEncoder(backbone, style_in_dim)
        dt = self.text_enc.bert.config.hidden_size
        self.graph_enc = GraphSAGEEncoder(graph_in_dim, hidden_dim=graph_hidden, num_layers=2)

        self.fuser = DualCoAttentionClassifier(dim_text=dt, dim_graph=graph_hidden, num_classes=num_classes)
        self.g_null = nn.Parameter(torch.zeros(graph_hidden))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_id: torch.Tensor,
        style_vec: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        role_to_node: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_text = self.text_enc(input_ids, attention_mask, style_vec)
        z_nodes = self.graph_enc(graph_x, graph_edge_index)
        node_idx = role_to_node[role_id]
        mask = (node_idx >= 0)  # [B] bool

        safe_idx = node_idx.clamp(min=0)  # 闃叉绱㈠紩鎶ラ敊
        g_vec = z_nodes[safe_idx]  # 鍏堝彇涓€涓€滃崰浣嶁€濈殑
        g_vec = torch.where(mask.unsqueeze(1), g_vec, self.g_null.expand_as(g_vec))
        logits = self.fuser(H_text, g_vec)
        return logits, z_nodes


# -------------------------
# 6) Train/Eval
# -------------------------


@torch.no_grad()
def evaluate(
    model,
    dl: DataLoader,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    role_to_node: torch.Tensor,
    device: str,
    criterion: nn.Module,
    average: str = "macro",
    return_preds: bool = False,
):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            role_id = batch["role_id"].to(device)
            style_vec = batch["style_vec"].to(device)
            y = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask, role_id, style_vec,
                              graph_x, graph_edge_index, role_to_node)
            loss = criterion(logits, y)

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n += bs

            y_pred = torch.argmax(logits, dim=-1)
            y_true_all.extend(y.detach().cpu().tolist())
            y_pred_all.extend(y_pred.detach().cpu().tolist())

    avg_loss = total_loss / max(1, n)
    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)
    metrics = compute_metrics(y_true_np, y_pred_np, average=average)

    if return_preds:
        return avg_loss, metrics, y_true_np, y_pred_np
    return avg_loss, metrics


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Ablation training with bot profile descriptions as text input and "
            "the same per-bot chronological 8/1/1 split as the main experiment."
        )
    )
    ap.add_argument("--csv_path", type=str, default="aigc_new_with_style_features_with_engagement_class.csv")
    ap.add_argument("--csv_encoding", type=str, default="utf-8-sig")
    ap.add_argument("--label_col", type=str, default="engagement_class")
    ap.add_argument("--time_col", type=str, default="create_time")
    ap.add_argument(
        "--text_col",
        type=str,
        default=BOT_DESCRIPTION_TEXT_COL,
        help="Compatibility alias; text input is generated from bot profile descriptions.",
    )
    ap.add_argument(
        "--role_col",
        type=str,
        default=BOT_DESCRIPTION_ID_COL,
        help="Compatibility alias; internally generated from bot profile descriptions.",
    )
    ap.add_argument("--nodes_csv", type=str, default="nodes1.csv")
    ap.add_argument("--nodes_encoding", type=str, default="gbk")
    ap.add_argument("--extra_nodes_csv", type=str, default="")
    ap.add_argument("--edges_csv", type=str, default="edges1.csv")
    ap.add_argument("--edges_encoding", type=str, default="utf-8-sig")
    ap.add_argument("--extra_edges_csv", type=str, default="")
    ap.add_argument("--profile_csv", type=str, default=DEFAULT_PROFILE_CSV_NAME)
    ap.add_argument("--profile_desc_col", type=str, default="description")
    ap.add_argument("--profile_encoding", type=str, default="utf-8-sig")
    ap.add_argument("--bot_description_col", type=str, default=BOT_DESCRIPTION_TEXT_COL)
    ap.add_argument("--bot_description_id_col", type=str, default=BOT_DESCRIPTION_ID_COL)
    ap.add_argument(
        "--feature_cols",
        type=str,
        default="length,emoji_count,is_qa,sentiment_score,TTR,RTTR,MTLD,MSTTR,common_ratio,stop_ratio",
    )
    ap.add_argument("--backbone", type=str, default="chinese-roberta-wwm-ext")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_list", type=str, default="42,52,62,72,82")
    ap.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Kept only for backward-compatible checkpoint metadata; chronological split does not use it.",
    )
    ap.add_argument("--metric_average", type=str, default="macro")
    ap.add_argument("--graph_hidden", type=int, default=64)
    ap.add_argument("--early_patience", type=int, default=99, help="val_f1 patience")
    ap.add_argument("--early_min_delta", type=float, default=1e-4, help="min val_f1 improvement")
    ap.add_argument("--lambda_unsup", type=float, default=0.0)
    ap.add_argument("--unsup_every", type=int, default=1)
    ap.add_argument("--rw_len", type=int, default=5)
    ap.add_argument("--rw_pos", type=int, default=5)
    ap.add_argument("--rw_neg", type=int, default=10)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Ablation setting: GraphSAGE + bot-profile-description text + style-feature-only FiLM modulation")
    seed_values = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    if len(seed_values) != 5:
        raise ValueError(f"--seed_list must contain exactly 5 seeds, got {len(seed_values)}: {seed_values}")
    print(
        "[INFO] training seeds={} split=chronological per-bot 8/1/1 time_col={}".format(
            seed_values, args.time_col
        )
    )

    df = build_chronological_per_bot_split(args)
    df, uid_to_description_id, id_to_description = attach_bot_profile_descriptions(df, args)
    args.text_col = args.bot_description_col
    args.role_col = args.bot_description_id_col

    if args.label_col not in df.columns:
        raise ValueError(f"CSV must include label column: {args.label_col}")
    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    role_ids_all = torch.tensor(
        pd.to_numeric(df[args.role_col], errors="raise").astype(int).values,
        dtype=torch.long,
    )
    num_roles = max(int(role_ids_all.max().item() + 1), len(id_to_description))

    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for c in feat_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing feature column: {c}")
    feats = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True))

    split_arrays = {
        split_name: np.flatnonzero(df["__split"].to_numpy() == split_name)
        for split_name in ("train", "val", "test")
    }
    idx_train = split_arrays["train"]
    idx_val = split_arrays["val"]
    idx_test = split_arrays["test"]

    le = LabelEncoder()
    le.fit(df.iloc[idx_train][args.label_col].astype(str).fillna("NA").values)
    unseen_labels = set(df[args.label_col].astype(str).fillna("NA").values) - set(le.classes_)
    if unseen_labels:
        raise ValueError(
            "Non-training split(s) contain labels absent from training: {}".format(
                sorted(unseen_labels)
            )
        )
    y_all = torch.tensor(
        le.transform(df[args.label_col].astype(str).fillna("NA").values),
        dtype=torch.long,
    )
    num_classes = int(len(le.classes_))
    print(f"[INFO] num_classes={num_classes}, classes={list(le.classes_)}")
    print(f"[INFO] Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    scaler = StandardScaler()
    scaler.fit(feats.iloc[idx_train].values)
    style_all = scaler.transform(feats.values)
    style_all_t = torch.tensor(style_all, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    def make_ds(idxs: np.ndarray) -> WeiboTensorDataset:
        idxs_t = torch.as_tensor(idxs, dtype=torch.long)
        return WeiboTensorDataset(
            texts=df.iloc[idxs][args.text_col].tolist(),
            role_ids=role_ids_all[idxs_t],
            style_feats=style_all_t[idxs_t],
            labels=y_all[idxs_t],
            tokenizer=tokenizer,
            max_len=args.max_len,
        )
    ds_train = make_ds(idx_train)
    ds_val = make_ds(idx_val)
    ds_test = make_ds(idx_test)
    dl_train_eval = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if not os.path.exists(args.nodes_csv):
        raise FileNotFoundError(f"Cannot find {args.nodes_csv}")
    if not os.path.exists(args.edges_csv):
        raise FileNotFoundError(f"Cannot find {args.edges_csv}")
    nodes, X, edge_index = build_graph_inputs_from_csv(args)
    graph_x = torch.tensor(X, dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    N, graph_in_dim = X.shape
    node_uids = normalize_id(nodes["uid"])
    role_to_node = torch.full((num_roles,), -1, dtype=torch.long)
    for node_idx, uid in enumerate(node_uids.tolist()):
        if node_idx >= N:
            continue
        description_id = uid_to_description_id.get(uid)
        if description_id is not None and 0 <= description_id < num_roles:
            role_to_node[description_id] = int(node_idx)
    used_description_ids = sorted(set(role_ids_all.tolist()))
    missing = [description_id for description_id in used_description_ids if role_to_node[description_id] < 0]
    if missing:
        print(
            f"[WARN] bot_description_id {missing} not found in nodes.csv by uid, "
            "g_null will be used for them."
        )
    role_to_node = role_to_node.to(device)
    adj = build_adj_list(torch.tensor(edge_index, dtype=torch.long), num_nodes=N)
    criterion = nn.CrossEntropyLoss()
    train_metric_runs: List[Dict[str, float]] = []
    test_metric_runs: List[Dict[str, float]] = []
    ckpt_paths: List[str] = []
    for run_idx, seed in enumerate(seed_values, start=1):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        print(f"\n[SEED {run_idx}/{len(seed_values)}] seed={seed}")
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        model = EngagementE2EModel(
            backbone=args.backbone,
            num_roles=num_roles,
            style_in_dim=len(feat_cols),
            graph_in_dim=graph_in_dim,
            num_classes=num_classes,
            graph_hidden=args.graph_hidden,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = args.epochs * max(1, len(dl_train))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        best_val_f1 = -1.0
        global_step = 0
        no_improve = 0
        best_epoch = -1
        ckpt_path = os.path.join(
            "ckpt",
            f"ablation_style_only_desc_sage_maxlen{args.max_len}_gh{args.graph_hidden}_seed{seed}_chronological.pt",
        )
        ckpt_paths.append(ckpt_path)
        for ep in range(args.epochs):
            model.train()
            total_loss = 0.0
            n = 0
            for batch in dl_train:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                role_id = batch["role_id"].to(device)
                style_vec = batch["style_vec"].to(device)
                y = batch["label"].to(device)
                logits, z_nodes = model(
                    input_ids,
                    attention_mask,
                    role_id,
                    style_vec,
                    graph_x,
                    graph_edge_index,
                    role_to_node,
                )
                loss = criterion(logits, y)
                if args.lambda_unsup > 0 and (global_step % max(1, args.unsup_every) == 0):
                    roots = role_to_node[torch.arange(num_roles, device=device)]
                    roots = roots[roots >= 0]
                    if roots.numel() > 0:
                        pos_nodes = sample_rw_positives(roots, adj, args.rw_len, args.rw_pos, rng)
                        loss_unsup = unsup_rw_neg_sampling_loss(z_nodes, roots, pos_nodes, args.rw_neg)
                        loss = loss + args.lambda_unsup * loss_unsup
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                bs = input_ids.size(0)
                total_loss += loss.item() * bs
                n += bs
                global_step += 1
            train_loss = total_loss / max(1, n)
            val_loss, val_metrics = evaluate(
                model,
                dl_val,
                graph_x,
                graph_edge_index,
                role_to_node,
                device,
                criterion,
                average=args.metric_average,
            )
            print(
                f"[Seed {seed} | Epoch {ep+1}/{args.epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_P={val_metrics['precision']:.4f} "
                f"val_R={val_metrics['recall']:.4f} "
                f"val_F1={val_metrics['f1']:.4f} ({args.metric_average})"
            )
            improved = val_metrics["f1"] > (best_val_f1 + args.early_min_delta)
            if improved:
                best_val_f1 = val_metrics["f1"]
                best_epoch = ep + 1
                no_improve = 0
                os.makedirs("ckpt", exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "backbone": args.backbone,
                        "num_roles": num_roles,
                        "num_classes": num_classes,
                        "label_classes": le.classes_.tolist(),
                        "feature_cols": feat_cols,
                        "graph_encoder": "sage",
                        "text_encoder_ablation": "style_feature_only_film",
                        "text_input": "bot_profile_description",
                        "split_method": "per-bot chronological post-level 80/10/10",
                        "time_col": args.time_col,
                        "profile_csv": resolve_profile_csv(args),
                        "profile_desc_col": args.profile_desc_col,
                        "bot_description_col": args.bot_description_col,
                        "bot_description_id_col": args.bot_description_id_col,
                        "bot_description_mapping": id_to_description,
                        "graph_hidden": args.graph_hidden,
                        "role_col": args.role_col,
                        "seed": seed,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1
                print(
                    f"[Seed {seed} | EarlyStop] no_improve={no_improve}/{args.early_patience} "
                    f"(best_f1={best_val_f1:.4f} @ epoch {best_epoch})"
                )
                if no_improve >= args.early_patience:
                    print(
                        f"[Seed {seed} | EarlyStop] Stop at epoch {ep + 1}. "
                        f"Best epoch={best_epoch}, best_val_f1={best_val_f1:.4f}"
                    )
                    break
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        train_eval_loss, train_eval_metrics = evaluate(
            model,
            dl_train_eval,
            graph_x,
            graph_edge_index,
            role_to_node,
            device,
            criterion,
            average=args.metric_average,
        )
        train_metric_runs.append(train_eval_metrics)
        test_loss, test_metrics, y_true, y_pred = evaluate(
            model,
            dl_test,
            graph_x,
            graph_edge_index,
            role_to_node,
            device,
            criterion,
            average=args.metric_average,
            return_preds=True,
        )
        test_metric_runs.append(test_metrics)
        print(
            f"[Seed {seed} | TRAIN] loss={train_eval_loss:.4f} "
            f"acc={train_eval_metrics['acc']:.4f} "
            f"P={train_eval_metrics['precision']:.4f} "
            f"R={train_eval_metrics['recall']:.4f} "
            f"F1={train_eval_metrics['f1']:.4f}"
        )
        print(
            f"[Seed {seed} | TEST] loss={test_loss:.4f} "
            f"acc={test_metrics['acc']:.4f} "
            f"P={test_metrics['precision']:.4f} "
            f"R={test_metrics['recall']:.4f} "
            f"F1={test_metrics['f1']:.4f}"
        )
        num_classes_eval = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
        labels = np.arange(num_classes_eval)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        np.savetxt(f"cm_test_counts_seed{seed}.csv", cm, fmt="%d", delimiter=",")
        np.savetxt(f"cm_test_norm_true_seed{seed}.csv", cm_norm, fmt="%.6f", delimiter=",")
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            f"test_predictions_ablation_style_only_seed{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        class_names = [str(i) for i in labels]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(values_format="d", cmap=plt.cm.Blues)
        for text in disp.text_.ravel():
            text.set_color("black")
            text.set_fontsize(11)
        plt.tight_layout()
        plt.savefig(f"cm_test_seed{seed}.png", dpi=300)
        plt.close()
        print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4))
    metric_keys = ["acc", "precision", "recall", "f1"]
    train_summary: Dict[str, Dict[str, float]] = {}
    test_summary: Dict[str, Dict[str, float]] = {}
    pm = "\u00B1"
    print(f"\n[SUMMARY] mean {pm} std over 5 seeds")
    for k in metric_keys:
        tr_vals = np.array([m[k] for m in train_metric_runs], dtype=np.float64)
        te_vals = np.array([m[k] for m in test_metric_runs], dtype=np.float64)
        train_summary[k] = {"mean": float(tr_vals.mean()), "std": float(tr_vals.std(ddof=0))}
        test_summary[k] = {"mean": float(te_vals.mean()), "std": float(te_vals.std(ddof=0))}
        print(f"[TRAIN] {k}: {train_summary[k]['mean']:.4f} {pm} {train_summary[k]['std']:.4f}")
        print(f"[TEST ] {k}: {test_summary[k]['mean']:.4f} {pm} {test_summary[k]['std']:.4f}")
    os.makedirs("ckpt", exist_ok=True)
    with open("ckpt/style_scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, ensure_ascii=False)
    with open("ckpt/label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump({"label_col": args.label_col, "classes": le.classes_.tolist()}, f, ensure_ascii=False)
    with open("ckpt/ablation_style_only_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seed_values,
                "split_method": "per-bot chronological post-level 80/10/10",
                "time_col": args.time_col,
                "text_input": "bot_profile_description",
                "profile_csv": resolve_profile_csv(args),
                "profile_desc_col": args.profile_desc_col,
                "bot_description_col": args.bot_description_col,
                "bot_description_id_col": args.bot_description_id_col,
                "bot_description_mapping": id_to_description,
                "metric_average": args.metric_average,
                "train": train_summary,
                "test": test_summary,
                "ckpt_paths": ckpt_paths,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        "[DONE] Saved per-seed ckpt/prediction/confusion files, "
        "ckpt/style_scaler.json, ckpt/label_encoder_classes.json, "
        "ckpt/ablation_style_only_seed_summary.json"
    )
if __name__ == "__main__":
    main()


