import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer

import train_multiclass_engagement_e2e_5seeds_with_train_metrics as base
from topology_feature_encoder import build_graph_from_csv as build_paper_topology_graph


DEFAULT_EXTRA_NODE_CSV_NAMES = (
    "fan_bfs_nodes.csv",
    "fans_bfs_nodes.csv",
    "fan_nodes3.csv",
    "fans_bfs_nodes3.csv",
)
DEFAULT_EXTRA_EDGE_CSV_NAMES = (
    "fan_bfs_edges.csv",
    "fans_bfs_edges.csv",
    "fan_edges3.csv",
    "fan_bfs_edges3.csv",
    "fans_bfs_edges3.csv",
)
EDGE_SRC_COL = "__src_uid"
EDGE_DST_COL = "__dst_uid"
DEFAULT_PROFILE_CSV_NAME = "bot_profile_descriptions.csv"
ROLE_DESCRIPTION_TEXT_COL = "role_description"
ROLE_DESCRIPTION_ID_COL = "role_description_id"


def chronological_checkpoint_path(args, seed):
    """Checkpoint path for chronological per-bot 8/1/1 training.

    This deliberately does not use ``split_seed`` because the data split is
    deterministic and is defined only by chronological ordering within each bot.
    """
    return os.path.join(
        "ckpt",
        "best_TPS_maxlen{}_seed{}_chronological.pt".format(args.max_len, seed),
    )


def normalize_id(series):
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def report_nonfinite_frame(frame, name):
    """Report NaN/+Inf/-Inf counts for a numeric pandas DataFrame."""
    values = frame.to_numpy(dtype=np.float64, copy=False)
    nan_count = int(np.isnan(values).sum())
    posinf_count = int(np.isposinf(values).sum())
    neginf_count = int(np.isneginf(values).sum())
    print(
        "[CHECK] {} non-finite values: NaN={} +Inf={} -Inf={}".format(
            name, nan_count, posinf_count, neginf_count
        )
    )
    return nan_count, posinf_count, neginf_count


def sanitize_frame_with_reference_medians(frame, reference_indices, name):
    """Replace +/-Inf with NaN, then impute from reference rows only.

    The reference rows are normally the training split. This avoids fitting
    imputation statistics on validation/test posts. If one feature is entirely
    non-finite in the reference rows, 0.0 is used for that feature and a warning
    is emitted.
    """
    clean = frame.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    report_nonfinite_frame(clean, name + " after inf->nan")

    reference = clean.iloc[reference_indices]
    medians = reference.median(numeric_only=True)

    all_missing = medians[medians.isna()].index.tolist()
    if all_missing:
        print(
            "[WARN] {} columns are entirely non-finite in the training split: {}. "
            "Using 0.0 as the fallback imputation value for these columns.".format(
                name, all_missing
            )
        )
        medians = medians.fillna(0.0)

    clean = clean.fillna(medians)

    remaining = ~np.isfinite(clean.to_numpy(dtype=np.float64, copy=False))
    if remaining.any():
        bad = np.argwhere(remaining)
        raise ValueError(
            "{} still contains non-finite values after imputation; first bad positions: {}".format(
                name, bad[:10].tolist()
            )
        )

    print("[INFO] {}: all numerical values are finite after imputation.".format(name))
    return clean, medians


def sanitize_numpy_features(array, name):
    """Make a 2-D numpy feature matrix finite using per-column medians.

    Non-finite values are replaced by the median of finite values in the same
    column. A column containing no finite values falls back to 0.0.
    """
    values = np.asarray(array, dtype=np.float32).copy()
    if values.ndim != 2:
        raise ValueError("{} must be a 2-D feature matrix, got shape {}.".format(name, values.shape))

    finite_mask = np.isfinite(values)
    bad_count = int((~finite_mask).sum())
    if bad_count == 0:
        print("[CHECK] {}: no NaN/Inf values found.".format(name))
        return values

    print("[WARN] {} contains {} NaN/Inf values; applying per-column median imputation.".format(
        name, bad_count
    ))

    for col in range(values.shape[1]):
        col_values = values[:, col]
        finite_col = np.isfinite(col_values)
        if finite_col.any():
            fill_value = float(np.median(col_values[finite_col]))
        else:
            fill_value = 0.0
            print(
                "[WARN] {} column {} has no finite values; using 0.0.".format(
                    name, col
                )
            )
        col_values[~finite_col] = fill_value
        values[:, col] = col_values

    if not np.isfinite(values).all():
        raise ValueError("{} still contains NaN/Inf after sanitization.".format(name))

    print("[INFO] {}: all numerical values are finite after imputation.".format(name))
    return values


def is_disabled_path_list(value):
    return bool(value) and str(value).strip().lower() in {"none", "null", "-"}


def split_path_list(value):
    if not value:
        return []
    if is_disabled_path_list(value):
        return []
    paths = []
    for path in str(value).replace(";", ",").split(","):
        path = path.strip().strip("'").strip('"')
        if path:
            paths.append(path)
    return paths


def resolve_relative_path(path, base_dir):
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def normalize_path_list_arg(value, base_dir):
    if is_disabled_path_list(value):
        return "none"
    paths = split_path_list(value)
    if not paths:
        return ""
    return ",".join(os.path.abspath(resolve_relative_path(path, base_dir)) for path in paths)


def get_role_code_col(args):
    return getattr(args, "role_description_code_col", ROLE_DESCRIPTION_ID_COL)


def get_role_description_col(args):
    return getattr(args, "role_description_col", ROLE_DESCRIPTION_TEXT_COL)


def resolve_profile_csv(args):
    raw_path = getattr(args, "profile_csv", DEFAULT_PROFILE_CSV_NAME)
    if os.path.isabs(raw_path):
        return raw_path

    candidates = [os.path.abspath(raw_path)]
    for base_path in (
        getattr(args, "csv_path", ""),
        getattr(args, "nodes_csv", ""),
        __file__,
    ):
        base_dir = os.path.dirname(os.path.abspath(base_path)) if base_path else ""
        if base_dir:
            candidates.append(os.path.abspath(os.path.join(base_dir, raw_path)))

    for path in dict.fromkeys(candidates):
        if os.path.isfile(path):
            return path
    return candidates[0]


def read_csv_with_fallback(path, encodings):
    tried = []
    for encoding in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            tried.append(encoding)
    raise UnicodeDecodeError(
        encodings[0],
        b"",
        0,
        1,
        "Could not decode CSV {} with encodings: {}".format(path, tried),
    )


def find_uid_column(frame, context):
    for column in ("uid", "uid_1", "seed_uid"):
        if column in frame.columns:
            return column
    raise ValueError("{} must contain a uid-like column: uid, uid_1, or seed_uid.".format(context))


def resolve_extra_nodes_csvs(args):
    if is_disabled_path_list(getattr(args, "extra_nodes_csv", "")):
        return []
    explicit_paths = split_path_list(getattr(args, "extra_nodes_csv", ""))
    base_dir = os.path.dirname(args.nodes_csv) or "."
    primary = os.path.abspath(args.nodes_csv)
    if explicit_paths:
        paths = [os.path.abspath(resolve_relative_path(path, base_dir)) for path in explicit_paths]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Cannot find extra nodes CSV(s): {}".format(missing))
        return [path for path in dict.fromkeys(paths) if path != primary]

    paths = []
    for name in DEFAULT_EXTRA_NODE_CSV_NAMES:
        path = os.path.abspath(os.path.join(base_dir, name))
        if os.path.isfile(path) and path != primary:
            paths.append(path)
    return list(dict.fromkeys(paths))


def prepare_node_frame(path, frame, source_order):
    uid_col = find_uid_column(frame, "nodes CSV {}".format(path))
    frame = frame.copy()
    if uid_col != "uid":
        if "uid" in frame.columns:
            frame["uid"] = frame["uid"].where(frame["uid"].notna(), frame[uid_col])
        else:
            frame = frame.rename(columns={uid_col: "uid"})
    frame["uid"] = normalize_id(frame["uid"])
    frame = frame[frame["uid"] != ""].copy()
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame["__nodes_source_order"] = source_order
    frame["__nodes_source_path"] = path
    return frame


def merge_nodes_frames(node_frames, args):
    prepared = [
        prepare_node_frame(path, frame, source_order)
        for source_order, (path, frame) in enumerate(node_frames)
    ]
    combined = pd.concat(prepared, ignore_index=True, sort=False)

    role_code_col = get_role_code_col(args)
    if role_code_col in combined.columns:
        role_values = pd.to_numeric(combined[role_code_col], errors="coerce")
        valid_roles = combined.loc[role_values >= 0, ["uid"]].copy()
        valid_roles["__role_value"] = role_values.loc[role_values >= 0].astype(int).values
        conflict = valid_roles.groupby("uid")["__role_value"].nunique()
        conflict = conflict[conflict > 1]
        if not conflict.empty:
            raise ValueError("nodes CSVs have conflicting role description ids for uid(s): {}".format(
                conflict.index.astype(str).tolist()[:5]
            ))
        combined["__has_role"] = (role_values >= 0).astype(int)
    else:
        combined["__has_role"] = 0

    meta_cols = {"__nodes_source_order", "__nodes_source_path", "__has_role", "__non_null_count"}
    combined["__non_null_count"] = combined.drop(columns=list(meta_cols), errors="ignore").notna().sum(axis=1)

    duplicate_uids = int(len(combined) - combined["uid"].nunique())
    combined = combined.sort_values(
        ["uid", "__has_role", "__nodes_source_order", "__non_null_count"],
        ascending=[True, False, True, False],
        kind="mergesort",
    )
    nodes = combined.groupby("uid", as_index=False, sort=False).first()
    nodes = nodes.drop(columns=list(meta_cols), errors="ignore").reset_index(drop=True)
    source_text = ", ".join(
        "{}({})".format(os.path.basename(path), len(frame))
        for path, frame in node_frames
    )
    print("[INFO] Loaded nodes CSVs: {} -> merged_nodes={} duplicate_rows_collapsed={}".format(
        source_text, len(nodes), duplicate_uids
    ))
    return nodes


def read_nodes_csv(args):
    encodings = [args.nodes_encoding, "utf-8-sig", "utf-8", "gbk"]
    node_paths = [args.nodes_csv] + resolve_extra_nodes_csvs(args)
    node_frames = [
        (path, read_csv_with_fallback(path, encodings))
        for path in node_paths
    ]
    if len(node_frames) == 1:
        return node_frames[0][1]
    return merge_nodes_frames(node_frames, args)


def load_role_description_profiles(args):
    profile_csv = resolve_profile_csv(args)
    encodings = [
        getattr(args, "profile_encoding", "utf-8-sig"),
        "utf-8-sig",
        "utf-8",
        "gbk",
    ]
    profiles = read_csv_with_fallback(profile_csv, encodings)
    uid_col = find_uid_column(profiles, profile_csv)
    desc_col = getattr(args, "profile_desc_col", "description")
    if desc_col not in profiles.columns:
        fallback_cols = [col for col in ("role_description", "description", "desc_text") if col in profiles.columns]
        if not fallback_cols:
            raise ValueError("{} must contain a profile description column.".format(profile_csv))
        desc_col = fallback_cols[0]

    profiles = profiles[[uid_col, desc_col]].copy()
    profiles.columns = ["uid", "__profile_description"]
    profiles["uid"] = normalize_id(profiles["uid"])
    profiles["__profile_description"] = profiles["__profile_description"].fillna("").astype(str).str.strip()
    profiles = profiles[(profiles["uid"] != "") & (profiles["__profile_description"] != "")]
    if profiles.empty:
        raise ValueError("{} has no usable uid/description rows.".format(profile_csv))

    conflict = profiles.groupby("uid")["__profile_description"].nunique()
    conflict = conflict[conflict > 1]
    if not conflict.empty:
        raise ValueError("{} has conflicting descriptions for uid(s): {}".format(
            profile_csv, conflict.index.astype(str).tolist()[:5]
        ))

    profiles = profiles.drop_duplicates("uid", keep="first").reset_index(drop=True)
    uid_to_description = {}
    uid_to_code = {}
    id_to_description = {}
    for role_id, row in profiles.iterrows():
        description = row["__profile_description"]
        uid = row["uid"]
        uid_to_description[uid] = description
        uid_to_code[uid] = int(role_id)
        id_to_description[str(role_id)] = description

    print("[INFO] Loaded role descriptions: profiles={} unique_descriptions={} file={}".format(
        len(profiles), profiles["__profile_description"].nunique(), profile_csv
    ))
    return profile_csv, uid_to_description, uid_to_code, id_to_description


def ensure_role_description_ids(df, nodes, args):
    if "uid" not in df.columns:
        raise ValueError("Merged data must contain uid so role descriptions can be inferred.")

    code_col = get_role_code_col(args)
    text_col = get_role_description_col(args)
    profile_csv, uid_to_description, uid_to_code, id_to_description = load_role_description_profiles(args)

    df = df.copy()
    nodes = nodes.copy()
    df["uid"] = normalize_id(df["uid"])

    missing_uids = sorted(set(df["uid"].unique()) - set(uid_to_description))
    if missing_uids:
        raise ValueError("{} is missing role descriptions for dataset uid(s): {}".format(
            profile_csv, missing_uids[:10]
        ))

    df[text_col] = df["uid"].map(uid_to_description)
    df[code_col] = df["uid"].map(uid_to_code).astype(int)

    try:
        node_uid_col = find_uid_column(nodes, "nodes CSV")
    except ValueError:
        node_uid_col = None

    nodes[text_col] = ""
    nodes[code_col] = -1
    if node_uid_col is not None:
        node_uids = normalize_id(nodes[node_uid_col])
        mapped_descriptions = node_uids.map(uid_to_description)
        mapped_codes = node_uids.map(uid_to_code)
        mapped_mask = mapped_codes.notna()
        nodes.loc[mapped_mask, text_col] = mapped_descriptions.loc[mapped_mask].values
        nodes.loc[mapped_mask, code_col] = mapped_codes.loc[mapped_mask].astype(int).values

    args.role_col = code_col
    args.profile_csv = profile_csv
    args.role_description_col = text_col
    args.role_description_mapping = id_to_description
    print("[INFO] Encoded role descriptions into {}: used_by_fold={} total_codes={}".format(
        code_col, df[code_col].nunique(), len(id_to_description)
    ))
    return df, nodes


def edge_candidate_from_nodes_csv(nodes_csv):
    node_dir = os.path.dirname(nodes_csv) or "."
    node_name = os.path.basename(nodes_csv)
    if "nodes" not in node_name:
        return ""
    return os.path.join(node_dir, node_name.replace("nodes", "edges", 1))


def resolve_edges_csvs(args, required):
    candidates = []
    node_dir = os.path.dirname(args.nodes_csv) or "."
    extra_nodes_csvs = resolve_extra_nodes_csvs(args)

    if args.edges_csv:
        candidates.extend(split_path_list(args.edges_csv))
    else:
        primary_edge = edge_candidate_from_nodes_csv(args.nodes_csv)
        if primary_edge:
            candidates.append(primary_edge)

    for extra_nodes_csv in extra_nodes_csvs:
        extra_edge = edge_candidate_from_nodes_csv(extra_nodes_csv)
        if extra_edge:
            candidates.append(extra_edge)

    if getattr(args, "extra_edges_csv", ""):
        candidates.extend(split_path_list(args.extra_edges_csv))
    elif extra_nodes_csvs:
        candidates.extend(os.path.join(node_dir, name) for name in DEFAULT_EXTRA_EDGE_CSV_NAMES)

    fallback_candidates = [
        os.path.join(node_dir, "edges3.csv"),
        os.path.join(node_dir, "edges1.csv"),
        os.path.join(node_dir, "edges.csv"),
    ]

    paths = []
    explicit = split_path_list(args.edges_csv) + split_path_list(getattr(args, "extra_edges_csv", ""))
    explicit = [os.path.abspath(resolve_relative_path(path, node_dir)) for path in explicit]
    missing_explicit = [path for path in explicit if not os.path.isfile(path)]
    if missing_explicit:
        raise FileNotFoundError("Cannot find explicit edges CSV(s): {}".format(missing_explicit))

    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        path = os.path.abspath(resolve_relative_path(candidate, node_dir))
        if os.path.isfile(path):
            paths.append(path)
    paths = list(dict.fromkeys(paths))
    if paths:
        return paths
    for candidate in fallback_candidates:
        path = os.path.abspath(resolve_relative_path(candidate, node_dir))
        if os.path.isfile(path):
            return [path]
    if required:
        raise FileNotFoundError("Cannot find edges CSV; tried: {}".format(candidates + fallback_candidates))
    return []


def resolve_edges_csv(args, required):
    paths = resolve_edges_csvs(args, required)
    return paths[0] if paths else ""


def infer_edge_columns(edges):
    candidates = [
        ("source", "target"),
        ("src_uid", "dst_uid"),
        ("source_uid", "target_uid"),
        ("from_uid", "to_uid"),
        ("uid", "id"),
    ]
    for src_col, dst_col in candidates:
        if src_col in edges.columns and dst_col in edges.columns:
            return src_col, dst_col
    raise ValueError("Edges CSV must contain one of these source/target column pairs: {}".format(candidates))


def read_edges_csvs(args, required):
    edge_paths = resolve_edges_csvs(args, required=required)
    frames = []
    for edge_path in edge_paths:
        edges = read_csv_with_fallback(edge_path, [args.nodes_encoding, "utf-8-sig", "utf-8", "gbk"])
        src_col, dst_col = infer_edge_columns(edges)
        edge_frame = edges[[src_col, dst_col]].copy()
        edge_frame.columns = [EDGE_SRC_COL, EDGE_DST_COL]
        edge_frame[EDGE_SRC_COL] = normalize_id(edge_frame[EDGE_SRC_COL])
        edge_frame[EDGE_DST_COL] = normalize_id(edge_frame[EDGE_DST_COL])
        edge_frame = edge_frame[(edge_frame[EDGE_SRC_COL] != "") & (edge_frame[EDGE_DST_COL] != "")]
        frames.append(edge_frame)

    if not frames:
        return pd.DataFrame(columns=[EDGE_SRC_COL, EDGE_DST_COL]), edge_paths

    edges = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print("[INFO] Loaded edges CSVs: {} -> merged_edges={}".format(
        ", ".join(os.path.basename(path) for path in edge_paths), len(edges)
    ))
    return edges, edge_paths


def build_graph_from_csv(nodes, args):
    uid_col = find_uid_column(nodes, "nodes CSV")
    nodes = nodes.copy()
    nodes[uid_col] = normalize_id(nodes[uid_col])
    nodes = nodes[nodes[uid_col] != ""].drop_duplicates(uid_col, keep="first").reset_index(drop=True)

    edges, edge_paths = read_edges_csvs(args, required=True)

    edge_uids = set(edges[EDGE_SRC_COL]) | set(edges[EDGE_DST_COL])
    node_uids = set(nodes[uid_col])
    missing_uids = sorted(uid for uid in edge_uids if uid and uid not in node_uids)
    if missing_uids:
        add_nodes = pd.DataFrame({uid_col: missing_uids})
        nodes = pd.concat([nodes, add_nodes], ignore_index=True, sort=False)

    uid_to_idx = {uid: idx for idx, uid in enumerate(nodes[uid_col].astype(str).tolist())}
    edge_mask = edges[EDGE_SRC_COL].isin(uid_to_idx) & edges[EDGE_DST_COL].isin(uid_to_idx)
    if edge_mask.sum() < len(edges):
        print("[WARN] dropped {} edges with missing uid endpoints.".format(int(len(edges) - edge_mask.sum())))
    edges = edges.loc[edge_mask]

    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        row = edges[EDGE_SRC_COL].map(uid_to_idx).values.astype(np.int64)
        col = edges[EDGE_DST_COL].map(uid_to_idx).values.astype(np.int64)
        edge_index = np.vstack([row, col])

    if "verified" in nodes.columns:
        nodes["verified"] = nodes["verified"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

    preferred_features = [col.strip() for col in args.graph_feature_cols.split(",") if col.strip()]
    feature_cols = [col for col in preferred_features if col in nodes.columns]
    if not feature_cols:
        numeric_cols = nodes.apply(pd.to_numeric, errors="coerce").columns.tolist()
        excluded = {
            uid_col,
            "uid_1",
            "seed_uid",
            "role_id",
            get_role_code_col(args),
            get_role_description_col(args),
        }
        feature_cols = [col for col in numeric_cols if col not in excluded]

    if feature_cols:
        feat_frame = nodes[feature_cols].apply(pd.to_numeric, errors="coerce")
        # Treat +/-Inf exactly like missing values before graph-feature imputation.
        feat_frame = feat_frame.replace([np.inf, -np.inf], np.nan)
        report_nonfinite_frame(feat_frame, "graph CSV features")
        medians = feat_frame.median(numeric_only=True).fillna(0.0)
        feat_frame = feat_frame.fillna(medians)
        feat_values = sanitize_numpy_features(
            feat_frame.values.astype(np.float32),
            "graph CSV features after imputation",
        )
        if args.scale_graph_features:
            feat_values = StandardScaler().fit_transform(feat_values).astype(np.float32)
            feat_values = sanitize_numpy_features(
                feat_values,
                "scaled graph CSV features",
            )
    else:
        feature_cols = ["constant"]
        feat_values = np.ones((len(nodes), 1), dtype=np.float32)

    print("[INFO] Built graph from CSV: nodes={} edges={} features={} edges_csvs={}".format(
        len(nodes), edge_index.shape[1], feature_cols, ", ".join(edge_paths)
    ))
    return nodes, feat_values.astype(np.float32), edge_index.astype(np.int64)


def load_graph_inputs(nodes, args):
    extra_nodes = resolve_extra_nodes_csvs(args)
    custom_nodes = os.path.basename(args.nodes_csv) not in {"nodes.csv", "nodes1.csv"}
    guessed_edges = resolve_edges_csvs(args, required=False)
    should_build = (
        args.build_graph_from_csv
        or bool(extra_nodes)
        or not (os.path.exists(args.graph_x) and os.path.exists(args.edge_index))
        or (custom_nodes and bool(guessed_edges))
    )
    if should_build:
        if extra_nodes and not args.build_graph_from_csv:
            print("[INFO] Extra nodes CSV detected; building graph tensors from CSV to keep node order aligned.")
        elif custom_nodes and not args.build_graph_from_csv:
            print("[INFO] Custom nodes_csv detected; building graph tensors from CSV instead of old npy files.")
        return build_graph_from_csv(nodes, args)

    graph_x = np.load(args.graph_x)
    edge_index = np.load(args.edge_index)
    graph_x = sanitize_numpy_features(graph_x, "graph_x loaded from npy")
    print("[INFO] Loaded graph npy: graph_x={} edge_index={}".format(args.graph_x, args.edge_index))
    return nodes, graph_x, edge_index


def load_chronological_membership(args, splits=("train", "val", "test")):
    """Create a deterministic chronological 8/1/1 post split within every bot.

    Posts from each bot are sorted by ``args.time_col`` from earliest to latest.
    The earliest ~80% are used for training, the next ~10% for validation,
    and the latest remainder for testing. The same bot identities are
    intentionally present in all three splits, while individual posts remain
    mutually disjoint. No random seed is used for data splitting.
    """
    splits = tuple(splits)
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Per-bot chronological 8/1/1 splitting requires train, val, and test splits.")

    source = pd.read_csv(
        args.csv_path,
        encoding="utf-8-sig",
        dtype={"uid": str, "wid": str},
    )
    required = {"uid", "wid", args.label_col, args.time_col}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("{} is missing columns {}".format(args.csv_path, sorted(missing)))

    source = source[["uid", "wid", args.label_col, args.time_col]].copy()
    source["uid"] = normalize_id(source["uid"])
    source["wid"] = normalize_id(source["wid"])
    source = source[(source["uid"] != "") & (source["wid"] != "")].copy()

    # Parse once and keep a dedicated internal timestamp column so that the
    # original feature table can still retain its own create_time column.
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
        print("[WARN] {}: dropped {} exact duplicate post rows before splitting.".format(
            args.csv_path, int(duplicate_mask.sum())
        ))
        source = source.loc[~duplicate_mask].copy()

    conflicting = source.duplicated(["uid", "wid"], keep=False)
    if conflicting.any():
        examples = source.loc[conflicting, ["uid", "wid", args.label_col]].head().to_dict("records")
        raise ValueError("Duplicate (uid, wid) rows with conflicting labels: {}".format(examples))

    parts = []
    bot_stats = []
    for uid, group in source.groupby("uid", sort=True):
        # A stable chronological order is used. wid is a deterministic
        # tie-breaker when multiple posts share exactly the same timestamp.
        group = (
            group.copy()
            .sort_values(["__time", "wid"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )
        n = len(group)
        if n < 3:
            raise ValueError(
                "Bot {} has only {} posts; at least 3 are required for an 8/1/1 split.".format(uid, n)
            )

        # Use floor for train/val and assign the remainder to test. For normal
        # post counts this is exactly/approximately 80/10/10, while keeping
        # every post in exactly one split.
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

        boundaries = {
            "train": np.arange(0, n_train),
            "val": np.arange(n_train, n_train + n_val),
            "test": np.arange(n_train + n_val, n),
        }
        for split in ("train", "val", "test"):
            piece = group.iloc[boundaries[split]].copy()
            piece["__split"] = split
            parts.append(piece)

        # Explicitly verify the temporal ordering for this bot.
        train_last = group.iloc[n_train - 1]["__time"]
        val_first = group.iloc[n_train]["__time"]
        val_last = group.iloc[n_train + n_val - 1]["__time"]
        test_first = group.iloc[n_train + n_val]["__time"]
        if not (train_last <= val_first <= val_last <= test_first):
            raise AssertionError("Chronological split order failed for bot {}.".format(uid))

        bot_stats.append((
            uid, n, n_train, n_val, n_test,
            group.iloc[0]["__time"], group.iloc[-1]["__time"],
        ))

    membership = pd.concat(parts, ignore_index=True)

    # Post-level disjointness is mandatory; bot overlap is expected by design.
    split_post_sets = {
        split: set(zip(
            membership.loc[membership["__split"] == split, "uid"],
            membership.loc[membership["__split"] == split, "wid"],
        ))
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_post_sets[left] & split_post_sets[right]
        if overlap:
            raise AssertionError("Post leakage between {} and {}: {}".format(left, right, list(overlap)[:5]))

    print(
        "[INFO] Generated chronological per-bot post split using '{}': "
        "earliest 80% train / next 10% val / latest 10% test".format(args.time_col)
    )
    for split in ("train", "val", "test"):
        subset = membership[membership["__split"] == split]
        print("[INFO] {}: bots={} posts={}".format(split, subset["uid"].nunique(), len(subset)))
    print("[INFO] Same bot identities are intentionally shared across train/val/test; posts are mutually disjoint.")
    print(
        "[INFO] Per-bot split examples "
        "(uid,total,train,val,test,earliest,latest): {}".format(bot_stats[:5])
    )

    # The original timestamp text is not needed after __time has been created;
    # dropping it avoids a duplicate create_time column when merging features.
    membership = membership.drop(columns=[args.time_col], errors="ignore")
    return membership


def merge_membership_with_features(args, membership):
    features = pd.read_csv(args.csv_path, encoding="utf-8-sig", dtype={"uid": str, "wid": str})
    for key in ("uid", "wid"):
        if key not in features.columns:
            raise ValueError("Feature CSV must contain key column '{}'.".format(key))
        features[key] = normalize_id(features[key])
    exact_feature_duplicates = features.duplicated(keep="first")
    if exact_feature_duplicates.any():
        print("[WARN] {}: dropped {} exact duplicate feature rows.".format(
            args.csv_path, int(exact_feature_duplicates.sum())
        ))
        features = features.loc[~exact_feature_duplicates].copy()
    duplicated_keys = features.duplicated(["uid", "wid"], keep=False)
    if duplicated_keys.any():
        examples = features.loc[duplicated_keys, ["uid", "wid"]].head().to_dict("records")
        raise ValueError(
            "Feature CSV still has duplicate (uid, wid) keys after exact-row deduplication: {}".format(examples)
        )

    # Split-membership labels must replace any old/global label in the feature table.
    features = features.drop(columns=[args.label_col], errors="ignore")
    merged = membership.merge(features, on=["uid", "wid"], how="inner", validate="one_to_one")
    if len(merged) != len(membership):
        membership_keys = set(zip(membership["uid"], membership["wid"]))
        feature_keys = set(zip(features["uid"], features["wid"]))
        raise ValueError(
            "Some split posts cannot be found in the feature CSV: "
            "missing_in_features={}, extra_feature_rows={}. "
            "The feature CSV may contain many more posts than the current split membership, which is allowed, "
            "but every (uid, wid) pair in the split membership must exist in the feature CSV.".format(
                len(membership_keys - feature_keys), len(feature_keys - membership_keys)
            )
        )
    print(
        "[INFO] Merged split posts with feature table: matched_rows={} feature_rows_total={}".format(
            len(merged), len(features)
        )
    )
    return merged


def prepare_chronological_data(args, splits=("train", "val", "test")):
    splits = tuple(splits)
    membership = load_chronological_membership(args, splits=splits)
    df = merge_membership_with_features(args, membership)
    nodes = read_nodes_csv(args)
    df, nodes = ensure_role_description_ids(df, nodes, args)
    role_code_col = get_role_code_col(args)
    role_text_col = get_role_description_col(args)
    for column in (args.text_col, role_code_col, role_text_col, args.label_col):
        if column not in df.columns:
            raise ValueError("Merged data is missing required column '{}'.".format(column))

    feat_cols = [column.strip() for column in args.feature_cols.split(",") if column.strip()]
    missing_features = [column for column in feat_cols if column not in df.columns]
    if missing_features:
        raise ValueError("Merged data is missing style columns: {}".format(missing_features))

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    role_desc_ids_all = torch.tensor(
        pd.to_numeric(df[role_code_col], errors="raise").astype(int).values,
        dtype=torch.long,
    )
    max_role_id = int(role_desc_ids_all.max().item())
    if role_code_col in nodes.columns:
        node_role_ids = pd.to_numeric(nodes[role_code_col], errors="coerce")
        node_role_ids = node_role_ids[node_role_ids >= 0]
        if not node_role_ids.empty:
            max_role_id = max(max_role_id, int(node_role_ids.max()))
    num_roles = max(max_role_id + 1, len(getattr(args, "role_description_mapping", {})))
    feats = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    split_arrays = {split: np.flatnonzero(df["__split"].to_numpy() == split) for split in splits}

    # NaN/Inf-safe preprocessing. Importantly, the imputation statistics and
    # StandardScaler are still fitted on TRAINING POSTS ONLY, preserving the
    # chronological evaluation protocol and avoiding val/test leakage.
    report_nonfinite_frame(feats, "raw style features")
    feats, train_medians = sanitize_frame_with_reference_medians(
        feats,
        split_arrays["train"],
        "style features",
    )

    scaler = StandardScaler()
    train_style_np = feats.iloc[split_arrays["train"]].to_numpy(dtype=np.float64)
    scaler.fit(train_style_np)

    style_np = scaler.transform(feats.to_numpy(dtype=np.float64)).astype(np.float32)
    style_np = sanitize_numpy_features(style_np, "scaled style features")
    style_all_t = torch.tensor(style_np, dtype=torch.float32)

    if not torch.isfinite(style_all_t).all():
        raise ValueError("style_all_t contains NaN/Inf after preprocessing.")

    le = LabelEncoder()
    le.fit(df.iloc[split_arrays["train"]][args.label_col].astype(str))
    unseen = set(df[args.label_col].astype(str)) - set(le.classes_)
    if unseen:
        raise ValueError("Non-training split(s) contain labels absent from training: {}".format(sorted(unseen)))
    y_all = torch.tensor(le.transform(df[args.label_col].astype(str)), dtype=torch.long)
    num_classes = len(le.classes_)
    if num_classes != 3:
        raise ValueError("Expected all three engagement classes in training, found {}.".format(list(le.classes_)))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def make_dataset(indices):
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        return base.WeiboTensorDataset(
            texts=df.iloc[indices][args.text_col].tolist(),
            role_ids=role_desc_ids_all[index_tensor],
            style_feats=style_all_t[index_tensor],
            labels=y_all[index_tensor],
            tokenizer=tokenizer,
            max_len=args.max_len,
        )

    # ------------------------------------------------------------------
    # Paper-aligned topology feature construction.
    #
    # Instead of concatenating raw followers/statuses/friends/depth values,
    # delegate graph preprocessing to topology_feature_encoder.py:
    #   c_v <- in/out-degree centrality
    #   a_v <- log1p(statuses_count) for users; learnable vector for bots
    #   t_v <- bot/user type embedding
    # The learnable c_v/a_v/t_v encoders themselves live in the base model.
    # ------------------------------------------------------------------
    extra_node_paths = resolve_extra_nodes_csvs(args)
    edge_paths = resolve_edges_csvs(args, required=True)

    # All bot identities represented by the post table are known social bots.
    # Passing them explicitly avoids relying on depth/node_type heuristics
    # for bot identification.
    bot_uids = sorted(df["uid"].astype(str).unique().tolist())

    topology = build_paper_topology_graph(
        primary_nodes_csv=args.nodes_csv,
        extra_nodes_csvs=extra_node_paths,
        edge_csvs=edge_paths,
        bot_uids=bot_uids,
    )

    graph_nodes = topology.nodes.copy()
    graph_nodes["uid"] = normalize_id(graph_nodes["uid"])
    edge_index = topology.edge_index

    # Map the bot role-description IDs to their exact node indices in the
    # topology module's node ordering.
    uid_to_role_id = (
        df[["uid", role_code_col]]
        .drop_duplicates("uid")
        .set_index("uid")[role_code_col]
        .astype(int)
        .to_dict()
    )
    role_to_node = torch.full((num_roles,), -1, dtype=torch.long)
    for node_idx, uid in enumerate(graph_nodes["uid"].astype(str).tolist()):
        role_id = uid_to_role_id.get(uid)
        if role_id is not None and 0 <= int(role_id) < num_roles:
            role_to_node[int(role_id)] = int(node_idx)

    missing_roles = torch.where(role_to_node < 0)[0].tolist()
    used_role_ids = sorted(set(uid_to_role_id.values()))
    missing_used_roles = [rid for rid in used_role_ids if role_to_node[rid] < 0]
    if missing_used_roles:
        raise ValueError(
            "Some bot role IDs cannot be aligned to topology nodes: {}".format(
                missing_used_roles
            )
        )

    node_count = len(graph_nodes)
    # The learnable feature encoder outputs graph_hidden-dimensional h_v^(0).
    graph_in_dim = args.graph_hidden

    print(
        "[INFO] Paper-aligned topology features enabled: "
        "centrality={} activity={} node_types={} nodes={} edges={}".format(
            topology.centrality_features.shape,
            topology.activity_features.shape,
            topology.node_type_ids.shape,
            node_count,
            edge_index.shape[1],
        )
    )

    split_size_text = " ".join("{}={}".format(name, len(split_arrays[name])) for name in splits)
    print("[INFO] Chronological per-bot 8/1/1 split sizes: {}".format(split_size_text))
    print("[INFO] classes={}".format(list(le.classes_)))
    print("[INFO] role_code_col={} role_description_col={} num_roles={}".format(
        role_code_col, role_text_col, num_roles
    ))
    return {
        "df": df,
        "feat_cols": feat_cols,
        "le": le,
        "scaler": scaler,
        "num_roles": num_roles,
        "num_classes": num_classes,
        "graph_in_dim": graph_in_dim,
        "role_to_node_cpu": role_to_node,
        "centrality_features_np": topology.centrality_features,
        "activity_features_np": topology.activity_features,
        "node_type_ids_np": topology.node_type_ids,
        "edge_index_np": edge_index,
        "adj": base.build_adj_list(torch.tensor(edge_index, dtype=torch.long), num_nodes=node_count),
        "datasets": {name: make_dataset(indices) for name, indices in split_arrays.items()},
        "split_names": list(splits),
        "role_code_col": role_code_col,
        "role_description_col": role_text_col,
        "role_description_mapping": getattr(args, "role_description_mapping", {}),
        "topology_nodes": graph_nodes,
        "topology_activity_mean": topology.activity_mean,
        "topology_activity_std": topology.activity_std,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Three-class E2E training with chronological per-bot 8/1/1 post splitting and paper-aligned topology feature encoding.")
    parser.add_argument("--csv_path", default="aigc_new_with_style_features_with_engagement_class.csv", help="Feature-rich cleaned post table.")
    parser.add_argument("--output_dir", default="per_bot_811_training_outputs")
    parser.add_argument("--label_col", default="engagement_class")
    parser.add_argument("--text_col", default="text_raw")
    parser.add_argument(
        "--time_col",
        default="create_time",
        help="Timestamp column used to sort each bot's posts before the chronological 80/10/10 split.",
    )
    parser.add_argument("--role_col", default=ROLE_DESCRIPTION_ID_COL, help="Compatibility alias; internally generated from role descriptions.")
    parser.add_argument("--role_description_code_col", default=ROLE_DESCRIPTION_ID_COL)
    parser.add_argument("--role_description_col", default=ROLE_DESCRIPTION_TEXT_COL)
    parser.add_argument("--profile_csv", default=DEFAULT_PROFILE_CSV_NAME)
    parser.add_argument("--profile_desc_col", default="description")
    parser.add_argument("--profile_encoding", default="utf-8-sig")
    parser.add_argument("--nodes_csv", default="nodes1.csv")
    parser.add_argument(
        "--extra_nodes_csv",
        default="",
        help=(
            "Comma-separated extra node CSVs. If omitted, fan_bfs_nodes.csv, "
            "fans_bfs_nodes.csv, fan_nodes3.csv, or fans_bfs_nodes3.csv next "
            "to --nodes_csv is loaded when present; "
            "use 'none' to disable this auto-detection."
        ),
    )
    parser.add_argument("--nodes_encoding", default="gbk")
    parser.add_argument("--graph_x", default="node_features_raw.npy")
    parser.add_argument("--edge_index", default="edge_index.npy")
    parser.add_argument("--edges_csv", default="", help="Edges CSV used when graph npy files are missing or --build_graph_from_csv is set.")
    parser.add_argument(
        "--extra_edges_csv",
        default="",
        help=(
            "Comma-separated extra edge CSVs. Extra fan edge files are also "
            "auto-detected from extra node CSV names when present."
        ),
    )
    parser.add_argument("--build_graph_from_csv", action="store_true", help="Build graph_x and edge_index directly from nodes/edges CSV.")
    parser.add_argument("--graph_feature_cols", default="", help="Deprecated: topology_feature_encoder.py defines graph features.")
    parser.add_argument("--scale_graph_features", action="store_true", help="Deprecated and ignored: activity scaling is handled by topology_feature_encoder.py.")
    parser.add_argument("--feature_cols", default="length,emoji_count,is_qa,emotional_polarity,TTR,RTTR,MTLD,MSTTR,common_ratio,stop_ratio")
    parser.add_argument("--backbone", default="chinese-roberta-wwm-ext")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,52,62,72,82")
    parser.add_argument("--run_multi_seed", action="store_true")
    parser.add_argument("--metric_average", default="macro")
    parser.add_argument("--graph_encoder", default="sage", choices=["gt", "sage"])
    parser.add_argument("--graph_hidden", type=int, default=64)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--att_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--text_cond_dim", type=int, default=64)
    parser.add_argument("--style_proj_dim", type=int, default=64)
    parser.add_argument("--film_hidden", type=int, default=256)
    parser.add_argument("--att_hidden", type=int, default=256)
    parser.add_argument("--pred_hidden", type=int, default=256)
    parser.add_argument("--early_patience", type=int, default=99)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--lambda_unsup", type=float, default=0.0)
    parser.add_argument("--unsup_every", type=int, default=1)
    parser.add_argument("--rw_len", type=int, default=5)
    parser.add_argument("--rw_pos", type=int, default=5)
    parser.add_argument("--rw_neg", type=int, default=10)
    parser.add_argument("--save_each_seed_cm", action="store_true")
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Keep per-seed best .pt checkpoints instead of deleting them after test evaluation.",
    )
    args = parser.parse_args()
    # Internal compatibility attributes for functions imported from the
    # shared base trainer. They are not CLI options and are not used to
    # construct the chronological 8/1/1 data split.
    args.fold_id = 1
    return args


def main():
    args = parse_args()

    # The shared base trainer historically names checkpoints with
    # args.split_seed. Chronological 8/1/1 has no random split seed, so replace
    # only the checkpoint naming function with a chronological version.
    # base.run_one_seed() will therefore save/load the same checkpoint path
    # without reintroducing any random data split.
    base.default_checkpoint_path = chronological_checkpoint_path

    if args.att_hidden % args.att_heads != 0:
        raise ValueError("--att_hidden must be divisible by --att_heads.")
    # Resolve inputs before entering the experiment output directory.
    for attr in ("csv_path", "nodes_csv", "output_dir"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    node_dir = os.path.dirname(args.nodes_csv) or os.getcwd()
    args.extra_nodes_csv = normalize_path_list_arg(args.extra_nodes_csv, node_dir)
    args.edges_csv = normalize_path_list_arg(args.edges_csv, node_dir)
    args.extra_edges_csv = normalize_path_list_arg(args.extra_edges_csv, node_dir)
    args.profile_csv = resolve_profile_csv(args)
    if os.path.isdir(args.backbone):
        args.backbone = os.path.abspath(args.backbone)
    print("[INFO] csv_path={}".format(args.csv_path))
    print("[INFO] chronological per-bot 8/1/1 post split; time_col={}".format(args.time_col))
    print("[INFO] profile_csv={} profile_desc_col={}".format(args.profile_csv, args.profile_desc_col))
    print(
        "[INFO] topology_source=topology_feature_encoder.py "
        "nodes_csv={} extra_nodes_csv={} edges_csv={} extra_edges_csv={}".format(
            args.nodes_csv, args.extra_nodes_csv, args.edges_csv, args.extra_edges_csv
        )
    )
    bundle = prepare_chronological_data(args)

    experiment_output = os.path.join(args.output_dir, "per_bot_811_chronological")
    os.makedirs(experiment_output, exist_ok=True)
    os.chdir(experiment_output)
    os.makedirs("ckpt", exist_ok=True)

    run_seeds = base.parse_seeds(args.seeds) if args.run_multi_seed else [args.seed]
    print("[INFO] run_seeds={} split=chronological per-bot 8/1/1 time_col={}".format(run_seeds, args.time_col))

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
        print("[INFO] Resume enabled: completed seeds found: {}".format(sorted(completed_seeds)))

    pending_seeds = [seed for seed in run_seeds if seed not in completed_seeds]
    skipped_seeds = [seed for seed in run_seeds if seed in completed_seeds]
    if skipped_seeds:
        print("[INFO] Skipping already completed seeds: {}".format(skipped_seeds))
    if not pending_seeds:
        print("[INFO] All requested seeds are already completed for this per-bot split.")

    for seed in pending_seeds:
        print("[INFO] Starting seed {} ...".format(seed))
        result = base.run_one_seed(args, seed, bundle)
        results = [r for r in results if int(r["seed"]) != int(seed)]
        results.append(result)

        ckpt_path = base.default_checkpoint_path(args, seed)
        if args.keep_checkpoints:
            print("[INFO] Keeping seed {} checkpoint: {}".format(seed, ckpt_path))
        elif os.path.exists(ckpt_path):
            try:
                ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
                os.remove(ckpt_path)
                print("[INFO] Deleted seed {} checkpoint after test metrics were obtained: {} ({:.1f} MB freed)".format(seed, ckpt_path, ckpt_size_mb))
            except OSError as exc:
                print("[WARN] Could not delete checkpoint for seed {}: {} ({})".format(seed, ckpt_path, exc))

        results = sorted(results, key=lambda r: int(r["seed"]))
        base.save_summary(results, results_csv)
        print("[INFO] Progress saved: completed seeds = {}".format([int(r["seed"]) for r in results]))

    result_by_seed = {int(r["seed"]): r for r in results}
    missing_requested = [seed for seed in run_seeds if seed not in result_by_seed]
    if missing_requested:
        raise RuntimeError("Requested seeds are still missing after training: {}".format(missing_requested))

    results = [result_by_seed[seed] for seed in run_seeds]
    base.save_summary(results, results_csv)
    summary = base.compute_mean_std([
        {key: value for key, value in result.items() if key not in ("seed", "best_epoch")}
        for result in results
    ])
    summary["acc"] = summary["test_acc"]
    summary["precision"] = summary["test_precision"]
    summary["recall"] = summary["test_recall"]
    summary["f1"] = summary["test_f1"]

    with open("multi_seed_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump({
            "split_method": "per-bot chronological post-level 80/10/10",
            "time_col": args.time_col,
            "run_seeds": run_seeds,
            "summary": {key: {"mean": value[0], "std": value[1]} for key, value in summary.items()},
        }, file_obj, ensure_ascii=False, indent=2)
    with open("ckpt/style_scaler.json", "w", encoding="utf-8") as file_obj:
        json.dump({"mean": bundle["scaler"].mean_.tolist(), "scale": bundle["scaler"].scale_.tolist()}, file_obj)
    with open("ckpt/label_encoder_classes.json", "w", encoding="utf-8") as file_obj:
        json.dump({"label_col": args.label_col, "classes": bundle["le"].classes_.tolist()}, file_obj)
    with open("ckpt/role_description_mapping.json", "w", encoding="utf-8") as file_obj:
        json.dump({
            "profile_csv": args.profile_csv,
            "role_code_col": bundle["role_code_col"],
            "role_description_col": bundle["role_description_col"],
            "id_to_description": bundle["role_description_mapping"],
        }, file_obj, ensure_ascii=False, indent=2)

    with open("ckpt/topology_feature_meta.json", "w", encoding="utf-8") as file_obj:
        json.dump({
            "feature_definition": "x_v = c_v + a_v + t_v",
            "centrality": "directed in-degree/(N-1) and out-degree/(N-1)",
            "activity": "zscore(log1p(statuses_count)) for human users; learnable vector for bots",
            "node_type": "learnable bot/user embedding",
            "excluded_direct_features": [
                "followers_count",
                "friends_count",
                "verified",
                "depth",
                "norm_followers_count",

                "norm_statuses_count"
            ],
            "activity_mean": bundle["topology_activity_mean"],
            "activity_std": bundle["topology_activity_std"],
            "num_nodes": int(len(bundle["topology_nodes"])),
            "num_edges": int(bundle["edge_index_np"].shape[1]),
        }, file_obj, ensure_ascii=False, indent=2)

    print("[DONE] Chronological per-bot 8/1/1 outputs saved in {}".format(experiment_output))


if __name__ == "__main__":
    main()