import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import train_multiclass_engagement_e2e_network_only as base
from topology_feature_encoder import build_graph_from_csv as build_paper_topology_graph


DEFAULT_EXTRA_NODES_CSV = "fans_bfs_nodes3.csv"
DEFAULT_EXTRA_EDGES_CSV = "fans_bfs_edges3.csv"
TIME_COL_CANDIDATES = (
    "create_time",
    "created_time",
    "publish_time",
    "post_time",
    "timestamp",
    "time",
    "date",
)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def split_path_list(value: str):
    if not value:
        return []
    paths = []
    for path in str(value).replace(";", ",").split(","):
        path = path.strip().strip("'").strip('"')
        if path:
            paths.append(path)
    return paths


def resolve_relative(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def resolve_required_paths(value: str, base_dir: str, description: str):
    paths = [resolve_relative(p, base_dir) for p in split_path_list(value)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Cannot find {description}: {missing}")
    return list(dict.fromkeys(paths))


def read_csv_with_fallback(path: str, encodings):
    last_error = None
    for encoding in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"uid": str, "wid": str})
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {path} with encodings {list(dict.fromkeys(encodings))}; last={last_error}",
    )


def resolve_time_col(df: pd.DataFrame, requested: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"--time_col='{requested}' does not exist in the post CSV. "
                f"Available columns include: {df.columns.tolist()[:40]}"
            )
        return requested

    for candidate in TIME_COL_CANDIDATES:
        if candidate in df.columns:
            print(f"[INFO] Auto-detected chronological column: {candidate}")
            return candidate

    raise ValueError(
        "No chronological column could be auto-detected. Pass --time_col explicitly. "
        f"Tried: {TIME_COL_CANDIDATES}"
    )


def make_time_order_key(series: pd.Series, column_name: str) -> pd.Series:
    # If the entire column is numeric, sorting numerically is deterministic and
    # preserves temporal order for Unix timestamps and already-ordered counters.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        print(f"[INFO] Using numeric order for time column '{column_name}'.")
        return numeric.astype(np.float64)

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    bad = parsed.isna()
    if bad.any():
        examples = series.loc[bad].astype(str).head(10).tolist()
        raise ValueError(
            f"Time column '{column_name}' contains {int(bad.sum())} values that cannot be parsed. "
            f"Examples: {examples}"
        )
    print(f"[INFO] Using datetime order for time column '{column_name}'.")
    return parsed


def build_role_dca_context(adj, role_to_node: torch.Tensor, max_context_nodes: int):
    if max_context_nodes < 1:
        raise ValueError("--dca_context_size must be >= 1.")

    rows = []
    for node_idx in role_to_node.tolist():
        node_idx = int(node_idx)
        if node_idx < 0:
            context = []
        else:
            neighbors = sorted(int(nbr) for nbr in adj[node_idx] if int(nbr) != node_idx)
            context = [node_idx] + neighbors

        context = context[:max_context_nodes]
        context += [-1] * (max_context_nodes - len(context))
        rows.append(context)

    return torch.tensor(rows, dtype=torch.long)


def load_and_split_posts(args):
    df = read_csv_with_fallback(
        args.csv_path,
        [args.csv_encoding, "utf-8-sig", "utf-8", "gbk"],
    )

    required = {"uid", "wid", args.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Post CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["uid"] = normalize_id(df["uid"])
    df["wid"] = normalize_id(df["wid"])
    df = df[(df["uid"] != "") & (df["wid"] != "")].copy()

    exact_dups = df.duplicated(keep="first")
    if exact_dups.any():
        print(f"[WARN] Dropping {int(exact_dups.sum())} exact duplicate post rows.")
        df = df.loc[~exact_dups].copy()

    duplicate_keys = df.duplicated(["uid", "wid"], keep=False)
    if duplicate_keys.any():
        examples = df.loc[duplicate_keys, ["uid", "wid"]].head(10).to_dict("records")
        raise ValueError(
            "Post CSV contains duplicate (uid, wid) keys after exact-row deduplication. "
            f"Examples: {examples}"
        )

    time_col = resolve_time_col(df, args.time_col)
    args.time_col = time_col
    df["__time_order"] = make_time_order_key(df[time_col], time_col)

    # No random permutation: sort within each bot by time, then wid as a stable
    # tie-breaker. The earliest 80% are training, the next 10% validation, and
    # the latest 10% testing.
    df = df.sort_values(
        ["uid", "__time_order", "wid"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["__split"] = ""

    per_bot_rows = []
    for uid, group in df.groupby("uid", sort=False):
        indices = group.index.to_numpy()
        n = len(indices)
        n_train = int(np.floor(0.8 * n))
        n_val = int(np.floor(0.1 * n))
        n_test = n - n_train - n_val

        if min(n_train, n_val, n_test) < 1:
            raise ValueError(
                f"Bot uid={uid} has only {n} posts, which cannot produce non-empty "
                "chronological train/val/test partitions under 8/1/1."
            )

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        df.loc[train_idx, "__split"] = "train"
        df.loc[val_idx, "__split"] = "val"
        df.loc[test_idx, "__split"] = "test"

        per_bot_rows.append(
            {
                "uid": uid,
                "total": n,
                "train": len(train_idx),
                "val": len(val_idx),
                "test": len(test_idx),
            }
        )

    if (df["__split"] == "").any():
        raise RuntimeError("Internal error: some posts were not assigned to a split.")

    # Explicit leakage/order checks.
    if df.groupby(["uid", "wid"])["__split"].nunique().max() != 1:
        raise AssertionError("A post appears in more than one split.")

    for uid, group in df.groupby("uid", sort=False):
        train_times = group.loc[group["__split"] == "train", "__time_order"]
        val_times = group.loc[group["__split"] == "val", "__time_order"]
        test_times = group.loc[group["__split"] == "test", "__time_order"]
        if train_times.max() > val_times.min() or val_times.max() > test_times.min():
            raise AssertionError(f"Chronological order check failed for uid={uid}.")

    per_bot = pd.DataFrame(per_bot_rows)
    totals = per_bot[["train", "val", "test"]].sum().to_dict()
    print(
        "[INFO] Per-bot chronological 8/1/1 completed: "
        f"bots={len(per_bot)} train={totals['train']} val={totals['val']} test={totals['test']}"
    )
    print(per_bot.to_string(index=False))
    return df, per_bot


def prepare_graph_only_data(args):
    df, per_bot_counts = load_and_split_posts(args)

    split_arrays = {
        split: np.flatnonzero(df["__split"].to_numpy() == split)
        for split in ("train", "val", "test")
    }

    le = LabelEncoder()
    train_labels = df.iloc[split_arrays["train"]][args.label_col].astype(str)
    le.fit(train_labels)
    unseen = set(df[args.label_col].astype(str)) - set(le.classes_)
    if unseen:
        raise ValueError(
            "Validation/test contain engagement classes absent from training: "
            f"{sorted(unseen)}"
        )
    if len(le.classes_) != 3:
        raise ValueError(
            f"Expected three engagement classes in training, found {le.classes_.tolist()}"
        )
    y_all = torch.tensor(
        le.transform(df[args.label_col].astype(str)), dtype=torch.long
    )

    # role_id is only an internal index for bot -> graph-node alignment. It is
    # not a persona feature and no profile-description file is used.
    bot_uids = sorted(df["uid"].astype(str).unique().tolist())
    uid_to_role_id = {uid: idx for idx, uid in enumerate(bot_uids)}
    role_ids_all = torch.tensor(
        df["uid"].map(uid_to_role_id).astype(int).to_numpy(), dtype=torch.long
    )
    num_roles = len(bot_uids)

    def make_dataset(indices):
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return base.GraphOnlyPostDataset(
            role_ids=role_ids_all[index_tensor],
            labels=y_all[index_tensor],
        )

    node_dir = os.path.dirname(args.nodes_csv) or os.getcwd()
    extra_node_paths = resolve_required_paths(
        args.extra_nodes_csv,
        node_dir,
        "extra node CSV(s)",
    )
    edge_paths = resolve_required_paths(
        args.edges_csv,
        node_dir,
        "primary edge CSV(s)",
    ) + resolve_required_paths(
        args.extra_edges_csv,
        node_dir,
        "extra edge CSV(s)",
    )
    edge_paths = list(dict.fromkeys(edge_paths))

    topology = build_paper_topology_graph(
        primary_nodes_csv=args.nodes_csv,
        extra_nodes_csvs=extra_node_paths,
        edge_csvs=edge_paths,
        bot_uids=bot_uids,
    )

    graph_nodes = topology.nodes.copy()
    if "uid" not in graph_nodes.columns:
        raise ValueError("topology_feature_encoder output nodes must contain 'uid'.")
    graph_nodes["uid"] = normalize_id(graph_nodes["uid"])
    graph_uid_to_idx = {
        uid: idx for idx, uid in enumerate(graph_nodes["uid"].astype(str).tolist())
    }

    role_to_node = torch.full((num_roles,), -1, dtype=torch.long)
    missing_bot_uids = []
    for uid, role_id in uid_to_role_id.items():
        node_idx = graph_uid_to_idx.get(uid)
        if node_idx is None:
            missing_bot_uids.append(uid)
        else:
            role_to_node[role_id] = int(node_idx)

    if missing_bot_uids:
        raise ValueError(
            "The following target bots are absent from the topology graph: "
            f"{missing_bot_uids[:20]}"
        )

    edge_index = topology.edge_index
    node_count = len(graph_nodes)
    adj = base.build_adj_list(
        torch.tensor(edge_index, dtype=torch.long), num_nodes=node_count
    )
    role_to_dca_context = build_role_dca_context(
        adj, role_to_node, args.dca_context_size
    )
    dca_context_lengths = (role_to_dca_context >= 0).sum(dim=1).numpy()

    print(
        "[INFO] TOPOLOGY SOURCES: "
        f"nodes=[{args.nodes_csv}, {', '.join(extra_node_paths)}] "
        f"edges=[{', '.join(edge_paths)}]"
    )
    print(
        "[INFO] Paper-aligned topology tensors: "
        f"centrality={topology.centrality_features.shape} "
        f"activity={topology.activity_features.shape} "
        f"node_types={topology.node_type_ids.shape} "
        f"nodes={node_count} edges={edge_index.shape[1]}"
    )
    print(
        "[INFO] Graph-context DCA neighborhoods: "
        f"shape={tuple(role_to_dca_context.shape)} "
        f"min_len={int(dca_context_lengths.min())} "
        f"max_len={int(dca_context_lengths.max())} "
        f"mean_len={float(dca_context_lengths.mean()):.2f}"
    )
    print(f"[INFO] classes={le.classes_.tolist()}")

    return {
        "df": df,
        "per_bot_counts": per_bot_counts,
        "feat_cols": [],
        "le": le,
        "num_roles": num_roles,
        "num_classes": len(le.classes_),
        "graph_in_dim": args.graph_hidden,
        "role_to_node_cpu": role_to_node,
        "role_to_dca_context_cpu": role_to_dca_context,
        "centrality_features_np": topology.centrality_features,
        "activity_features_np": topology.activity_features,
        "node_type_ids_np": topology.node_type_ids,
        "edge_index_np": edge_index,
        "adj": adj,
        "datasets": {
            split: make_dataset(indices) for split, indices in split_arrays.items()
        },
        "split_names": ["train", "val", "test"],
        "topology_nodes": graph_nodes,
        "topology_activity_mean": topology.activity_mean,
        "topology_activity_std": topology.activity_std,
        "uid_to_role_id": uid_to_role_id,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Graph-only engagement prediction with per-bot chronological 8/1/1: "
            "TopologyFeatureEncoder + selectable GN/GAT encoder + graph-context DCA classifier."
        )
    )
    parser.add_argument(
        "--csv_path",
        default="aigc_new_with_style_features_with_engagement_class.csv",
        help="Post table containing uid, wid, engagement label, and time column.",
    )
    parser.add_argument("--csv_encoding", default="utf-8-sig")
    parser.add_argument("--label_col", default="engagement_class")
    parser.add_argument(
        "--time_col",
        default="",
        help="Chronological post column. If omitted, common names are auto-detected.",
    )

    parser.add_argument("--nodes_csv", default="nodes1.csv")
    parser.add_argument(
        "--extra_nodes_csv",
        default=DEFAULT_EXTRA_NODES_CSV,
        help="Additional topology node CSV(s), comma-separated.",
    )
    parser.add_argument(
        "--edges_csv",
        default="edges1.csv",
        help="Primary topology edge CSV(s), comma-separated.",
    )
    parser.add_argument(
        "--extra_edges_csv",
        default=DEFAULT_EXTRA_EDGES_CSV,
        help="Additional topology edge CSV(s), comma-separated.",
    )

    parser.add_argument("--output_dir", default="graph_only_training_outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--run_multi_seed", action="store_true")
    parser.add_argument("--metric_average", default="macro")

    parser.add_argument(
        "--graph_encoder",
        default="gn",
        choices=["gn", "gat", "sage"],
        help="Graph encoder: gn=GraphConv stack, gat=GATConv stack, sage=legacy GraphSAGE.",
    )
    parser.add_argument("--graph_hidden", type=int, default=64)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pred_hidden", type=int, default=256)
    parser.add_argument("--dca_dim", type=int, default=256)
    parser.add_argument("--dca_heads", type=int, default=4)
    parser.add_argument(
        "--dca_context_size",
        type=int,
        default=32,
        help="Maximum graph context nodes per bot for the DCA classifier.",
    )

    parser.add_argument("--early_patience", type=int, default=99)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--lambda_unsup", type=float, default=0.0)
    parser.add_argument("--unsup_every", type=int, default=1)
    parser.add_argument("--rw_len", type=int, default=5)
    parser.add_argument("--rw_pos", type=int, default=5)
    parser.add_argument("--rw_neg", type=int, default=10)
    parser.add_argument("--save_each_seed_cm", action="store_true")
    parser.add_argument("--keep_checkpoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.graph_encoder = base.normalize_graph_encoder_name(args.graph_encoder)
    graph_encoder_label = base.graph_encoder_display_name(args.graph_encoder)

    for attr in ("csv_path", "nodes_csv", "output_dir"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    node_dir = os.path.dirname(args.nodes_csv) or os.getcwd()

    # Resolve the two requested node sources and the corresponding two edge sources
    # before changing the working directory.
    args.extra_nodes_csv = ",".join(
        resolve_required_paths(args.extra_nodes_csv, node_dir, "extra node CSV(s)")
    )
    args.edges_csv = ",".join(
        resolve_required_paths(args.edges_csv, node_dir, "primary edge CSV(s)")
    )
    args.extra_edges_csv = ",".join(
        resolve_required_paths(args.extra_edges_csv, node_dir, "extra edge CSV(s)")
    )

    args.split_method = "per-bot chronological post-level 80/10/10"
    args.split_tag = f"chronological_811_dca_{args.graph_encoder}"

    print(f"[INFO] csv_path={args.csv_path}")
    print(
        "[INFO] MODEL VARIANT = NETWORK ONLY + GRAPH-CONTEXT DCA "
        f"(TopologyFeatureEncoder + {graph_encoder_label} + DCA classifier; "
        "no Text/Persona/Style)"
    )
    print(f"[INFO] graph_encoder={args.graph_encoder} ({graph_encoder_label})")
    print("[INFO] SPLIT = per-bot chronological 8/1/1; no random post split")

    bundle = prepare_graph_only_data(args)

    run_output = os.path.join(args.output_dir, args.split_tag)
    os.makedirs(run_output, exist_ok=True)

    # Save the exact deterministic post membership used in this run.
    manifest_cols = ["uid", "wid", args.time_col, args.label_col, "__split"]
    bundle["df"][manifest_cols].to_csv(
        os.path.join(run_output, "split_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    bundle["per_bot_counts"].to_csv(
        os.path.join(run_output, "split_counts_per_bot.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    os.chdir(run_output)
    os.makedirs("ckpt", exist_ok=True)

    run_seeds = base.parse_seeds(args.seeds) if args.run_multi_seed else [args.seed]
    print(f"[INFO] run_seeds={run_seeds}")

    results_csv = "multi_seed_results.csv"
    results = []
    completed_seeds = set()

    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        if "seed" not in existing_df.columns:
            raise ValueError("Existing multi_seed_results.csv has no 'seed' column.")
        existing_df = existing_df.drop_duplicates(subset=["seed"], keep="last")
        for _, row in existing_df.iterrows():
            record = row.to_dict()
            record["seed"] = int(record["seed"])
            results.append(record)
            completed_seeds.add(int(record["seed"]))
        print(
            f"[INFO] Resume enabled: completed seeds found: {sorted(completed_seeds)}"
        )

    pending_seeds = [seed for seed in run_seeds if seed not in completed_seeds]
    skipped_seeds = [seed for seed in run_seeds if seed in completed_seeds]
    if skipped_seeds:
        print(f"[INFO] Skipping already completed seeds: {skipped_seeds}")

    for seed in pending_seeds:
        print(f"[INFO] Starting seed {seed} ...")
        result = base.run_one_seed(args, seed, bundle)
        results = [r for r in results if int(r["seed"]) != int(seed)]
        results.append(result)

        ckpt_path = base.default_checkpoint_path(args, seed)
        if args.keep_checkpoints:
            print(f"[INFO] Keeping seed {seed} checkpoint: {ckpt_path}")
        elif os.path.exists(ckpt_path):
            try:
                ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
                os.remove(ckpt_path)
                print(
                    f"[INFO] Deleted seed {seed} checkpoint after test metrics were obtained: "
                    f"{ckpt_path} ({ckpt_size_mb:.1f} MB freed)"
                )
            except OSError as exc:
                print(
                    f"[WARN] Could not delete checkpoint for seed {seed}: "
                    f"{ckpt_path} ({exc})"
                )

        results = sorted(results, key=lambda r: int(r["seed"]))
        base.save_summary(results, results_csv)
        print(
            "[INFO] Progress saved: completed seeds = "
            f"{[int(r['seed']) for r in results]}"
        )

    result_by_seed = {int(r["seed"]): r for r in results}
    missing_requested = [seed for seed in run_seeds if seed not in result_by_seed]
    if missing_requested:
        raise RuntimeError(
            f"Requested seeds are still missing after training: {missing_requested}"
        )

    results = [result_by_seed[seed] for seed in run_seeds]
    base.save_summary(results, results_csv)
    summary = base.compute_mean_std(
        [
            {key: value for key, value in result.items() if key not in ("seed", "best_epoch")}
            for result in results
        ]
    )
    summary["acc"] = summary["test_acc"]
    summary["precision"] = summary["test_precision"]
    summary["recall"] = summary["test_recall"]
    summary["f1"] = summary["test_f1"]

    with open("multi_seed_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "split_method": args.split_method,
                "time_col": args.time_col,
                "model_variant": f"network_only_{args.graph_encoder}_dca",
                "graph_encoder": graph_encoder_label,
                "uses_dca": True,
                "gat_heads": args.gat_heads,
                "dca_dim": args.dca_dim,
                "dca_heads": args.dca_heads,
                "dca_context_size": args.dca_context_size,
                "run_seeds": run_seeds,
                "summary": {
                    key: {"mean": value[0], "std": value[1]}
                    for key, value in summary.items()
                },
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with open("ckpt/label_encoder_classes.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {"label_col": args.label_col, "classes": bundle["le"].classes_.tolist()},
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with open("ckpt/bot_graph_mapping.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "uid_to_role_id": bundle["uid_to_role_id"],
                "role_to_node": bundle["role_to_node_cpu"].tolist(),
                "role_to_dca_context": bundle["role_to_dca_context_cpu"].tolist(),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    with open("ckpt/topology_feature_meta.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "feature_definition": "x_v = c_v + a_v + t_v",
                "centrality": "directed in-degree/(N-1) and out-degree/(N-1)",
                "activity": "zscore(log1p(statuses_count)) for human users; learnable vector for bots",
                "node_type": "learnable bot/user embedding",
                "graph_encoder": graph_encoder_label,
                "gat_heads": args.gat_heads,
                "classifier": "graph-context DCA",
                "dca_dim": args.dca_dim,
                "dca_heads": args.dca_heads,
                "dca_context_size": args.dca_context_size,
                "nodes_csv": args.nodes_csv,
                "extra_nodes_csv": split_path_list(args.extra_nodes_csv),
                "edges_csv": split_path_list(args.edges_csv),
                "extra_edges_csv": split_path_list(args.extra_edges_csv),
                "activity_mean": bundle["topology_activity_mean"],
                "activity_std": bundle["topology_activity_std"],
                "num_nodes": int(len(bundle["topology_nodes"])),
                "num_edges": int(bundle["edge_index_np"].shape[1]),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] Outputs saved in {run_output}")


if __name__ == "__main__":
    main()
