# -*- coding: utf-8 -*-
"""
Build RAW node features (no SPE) and edge_index from nodes.csv / edges.csv.

Output:
  - node_features_raw.npy: [N, d0]  (raw node features only, NO spectral PE)
  - edge_index.npy:        [2, E]   (directed edge list for PyG)
  - idx2uid.csv            uid list in node order

Run:
  python build_node_features_raw.py --nodes_csv nodes.csv --edges_csv edges.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def name_hash_embedding(names, dim=8, base=1315423911):
    arr = np.zeros((len(names), dim), dtype=np.float32)
    for i, s in enumerate(names.astype(str).tolist()):
        h = base
        for ch in s:
            h ^= ((h << 5) + ord(ch) + (h >> 2)) & 0xFFFFFFFF
        for d in range(dim):
            arr[i, d] = ((h >> (d * 3)) ^ (h * (d + 1))) & 0xFFFF
        if arr[i].std() > 0:
            arr[i] = (arr[i] - arr[i].mean()) / (arr[i].std() + 1e-9)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", type=str, default="nodes1.csv")
    ap.add_argument("--edges_csv", type=str, default="edges1.csv")
    ap.add_argument("--encoding", type=str, default="gbk")
    ap.add_argument("--use_screen_name_hash", action="store_true")
    ap.add_argument("--hash_dim", type=int, default=8)
    ap.add_argument("--save_x", type=str, default="node_features_raw.npy")
    ap.add_argument("--save_edge_index", type=str, default="edge_index.npy")
    ap.add_argument("--save_idx2uid", type=str, default="idx2uid.csv")
    args = ap.parse_args()

    # ---------- nodes ----------
    nodes = pd.read_csv(args.nodes_csv, encoding=args.encoding)

    required_cols = {"uid", "followers_count", "statuses_count", "verified"}
    missing = required_cols - set(nodes.columns)
    if missing:
        raise ValueError(f"nodes.csv 缺少必要列: {missing}. 当前列: {nodes.columns.tolist()}")

    uid2idx = {uid: i for i, uid in enumerate(nodes["uid"].astype(str).tolist())}
    idx2uid = np.array(list(uid2idx.keys()))
    N = len(uid2idx)

    # verified -> {0,1}
    if nodes["verified"].dtype == bool:
        nodes["verified"] = nodes["verified"].astype(int)
    else:
        nodes["verified"] = nodes["verified"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

    # numeric features: log1p + standardize
    num_feats = nodes[["followers_count", "statuses_count", "verified"]].copy()
    num_feats = np.log1p(num_feats.values.astype(float))
    scaler = StandardScaler()
    num_feats = scaler.fit_transform(num_feats)

    X_init = num_feats

    # optional: screen_name hash embedding
    if args.use_screen_name_hash and ("screen_name" in nodes.columns):
        name_emb = name_hash_embedding(nodes["screen_name"].values, dim=args.hash_dim)
        X_init = np.concatenate([X_init, name_emb], axis=1)

    # optional: node_type one-hot if exists
    if "node_type" in nodes.columns:
        t = nodes["node_type"].astype(str).values
        t_oh = np.stack([(t == "ai").astype(np.float32), (t == "user").astype(np.float32)], axis=1)
        X_init = np.concatenate([X_init, t_oh], axis=1)

    X_init = X_init.astype(np.float32)
    print(f"[INFO] RAW node feature shape = {X_init.shape}")

    # ---------- edges ----------
    edges = pd.read_csv(args.edges_csv)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    mask = edges["source"].isin(uid2idx) & edges["target"].isin(uid2idx)
    if mask.sum() < len(edges):
        print(f"[WARN] drop {len(edges) - mask.sum()} edges with unknown uid.")
    edges = edges[mask]

    row = edges["source"].map(uid2idx).values.astype(np.int64)
    col = edges["target"].map(uid2idx).values.astype(np.int64)
    edge_index = np.vstack([row, col])  # [2, E]
    print(f"[INFO] edge_index shape = {edge_index.shape}")

    # ---------- save ----------
    np.save(args.save_x, X_init)
    np.save(args.save_edge_index, edge_index)
    pd.Series(idx2uid).to_csv(args.save_idx2uid, index=False, header=["uid"])
    print(f"[DONE] saved: {args.save_x}, {args.save_edge_index}, {args.save_idx2uid}")


if __name__ == "__main__":
    main()
