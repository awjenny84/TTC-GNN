# -*- coding: utf-8 -*-
"""Chronological per-bot 8/1/1 E2E training with masked concat fusion.

This entrypoint reuses the same post split, role-conditioned text encoder,
paper-aligned topology feature encoder, graph encoder, and training loop as
train_multiclass_engagement_e2e_5seeds_per_bot_811.py.

The modeling difference is the prediction head: the CLS text representation is
concatenated with the bot topology representation. If a role cannot be aligned
to a graph node, a learned null graph vector is used before concatenation.
"""

import json
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

import train_multiclass_engagement_e2e_5seeds_per_bot_811 as per_bot
import train_multiclass_engagement_e2e_5seeds_with_train_metrics as base
from topology_feature_encoder import TopologyFeatureEncoder


def mask_concat_chronological_checkpoint_path(args, seed: int) -> str:
    return os.path.join(
        "ckpt",
        "best_TPS_mask_concat_maxlen{}_seed{}_chronological.pt".format(args.max_len, seed),
    )


class MaskConcatClassifier(nn.Module):
    """Concatenate CLS text embedding and graph embedding before prediction."""

    def __init__(
        self,
        dim_text: int,
        dim_graph: int,
        num_classes: int,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(dim_text + dim_graph, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, h_text: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
        text_cls = h_text[:, 0, :] if h_text.dim() == 3 else h_text
        fused = torch.cat([text_cls, g_vec], dim=-1)
        return self.pred(fused)


class MaskConcatEngagementE2EModel(nn.Module):
    """Same text/topology/graph encoders as per-bot 811, masked concat head."""

    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        graph_in_dim: int,  # retained for compatibility with base.make_model
        num_classes: int,
        graph_encoder: str = "gt",
        graph_hidden: int = 64,
        graph_layers: int = 2,
        att_heads: int = 4,
        dropout: float = 0.1,
        text_cond_dim: int = 64,
        style_proj_dim: int = 64,
        film_hidden: int = 256,
        att_hidden: int = 256,  # retained for CLI/checkpoint compatibility
        pred_hidden: int = 256,
    ):
        super().__init__()
        _ = graph_in_dim
        _ = att_hidden

        self.text_enc = base.RoleConditionedTextEncoder(
            backbone,
            num_roles,
            style_in_dim,
            cond_dim=text_cond_dim,
            style_proj_dim=style_proj_dim,
            film_hidden=film_hidden,
            dropout=dropout,
        )
        dim_text = self.text_enc.bert.config.hidden_size

        # Paper-aligned topology feature construction: x_v = c_v + a_v + t_v.
        self.topology_feature_enc = TopologyFeatureEncoder(
            graph_dim=graph_hidden,
            dropout=dropout,
        )

        if graph_encoder == "sage":
            self.graph_enc = base.GraphSAGEEncoder(
                graph_hidden,
                hidden_dim=graph_hidden,
                num_layers=graph_layers,
                dropout=dropout,
            )
        else:
            self.graph_enc = base.GraphTransformerEncoder(
                graph_hidden,
                hidden_dim=graph_hidden,
                num_layers=graph_layers,
                heads=att_heads,
                dropout=dropout,
            )

        self.fuser = MaskConcatClassifier(
            dim_text=dim_text,
            dim_graph=graph_hidden,
            num_classes=num_classes,
            hidden=pred_hidden,
            dropout=dropout,
        )
        self.g_null = nn.Parameter(torch.zeros(graph_hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_id: torch.Tensor,
        style_vec: torch.Tensor,
        centrality_features: torch.Tensor,
        activity_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        graph_edge_index: torch.Tensor,
        role_to_node: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_text = self.text_enc(input_ids, attention_mask, role_id, style_vec)

        graph_x = self.topology_feature_enc(
            centrality_features,
            activity_features,
            node_type_ids,
        )
        z_nodes = self.graph_enc(graph_x, graph_edge_index)

        node_idx = role_to_node[role_id]
        mask = node_idx >= 0
        safe_idx = node_idx.clamp(min=0)
        g_vec = z_nodes[safe_idx]
        g_vec = torch.where(mask.unsqueeze(1), g_vec, self.g_null.expand_as(g_vec))

        logits = self.fuser(h_text, g_vec)
        return logits, z_nodes


def make_mask_concat_model(args, bundle: Dict[str, Any]) -> MaskConcatEngagementE2EModel:
    return MaskConcatEngagementE2EModel(
        backbone=args.backbone,
        num_roles=bundle["num_roles"],
        style_in_dim=len(bundle["feat_cols"]),
        graph_in_dim=bundle["graph_in_dim"],
        num_classes=bundle["num_classes"],
        graph_encoder=args.graph_encoder,
        graph_hidden=args.graph_hidden,
        graph_layers=base.get_arg(args, "graph_layers", 2),
        att_heads=base.get_arg(args, "att_heads", 4),
        dropout=base.get_arg(args, "dropout", 0.1),
        text_cond_dim=base.get_arg(args, "text_cond_dim", 64),
        style_proj_dim=base.get_arg(args, "style_proj_dim", 64),
        film_hidden=base.get_arg(args, "film_hidden", 256),
        att_hidden=base.get_arg(args, "att_hidden", 256),
        pred_hidden=base.get_arg(args, "pred_hidden", 256),
    )


_base_checkpoint_payload = base.checkpoint_payload


def mask_concat_checkpoint_payload(args, bundle: Dict[str, Any], seed: int) -> Dict[str, Any]:
    payload = _base_checkpoint_payload(args, bundle, seed)
    payload.update(
        {
            "fusion": "mask_concat",
            "concat_hidden": base.get_arg(args, "pred_hidden", 256),
            "missing_graph_node": "learned g_null vector",
        }
    )
    return payload


def parse_args():
    return per_bot.parse_args()


def main():
    args = parse_args()

    # Patch only the extension points used by the shared training loop.
    base.make_model = make_mask_concat_model
    base.checkpoint_payload = mask_concat_checkpoint_payload
    base.default_checkpoint_path = mask_concat_chronological_checkpoint_path

    for attr in ("csv_path", "nodes_csv", "output_dir"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    node_dir = os.path.dirname(args.nodes_csv) or os.getcwd()
    args.extra_nodes_csv = per_bot.normalize_path_list_arg(args.extra_nodes_csv, node_dir)
    args.edges_csv = per_bot.normalize_path_list_arg(args.edges_csv, node_dir)
    args.extra_edges_csv = per_bot.normalize_path_list_arg(args.extra_edges_csv, node_dir)
    args.profile_csv = per_bot.resolve_profile_csv(args)
    if os.path.isdir(args.backbone):
        args.backbone = os.path.abspath(args.backbone)

    print("[INFO] csv_path={}".format(args.csv_path))
    print("[INFO] chronological per-bot 8/1/1 post split; time_col={}".format(args.time_col))
    print("[INFO] fusion=mask_concat concat_hidden={}".format(args.pred_hidden))
    print("[INFO] profile_csv={} profile_desc_col={}".format(args.profile_csv, args.profile_desc_col))
    print(
        "[INFO] topology_source=topology_feature_encoder.py "
        "nodes_csv={} extra_nodes_csv={} edges_csv={} extra_edges_csv={}".format(
            args.nodes_csv, args.extra_nodes_csv, args.edges_csv, args.extra_edges_csv
        )
    )

    bundle = per_bot.prepare_chronological_data(args)

    experiment_output = os.path.join(args.output_dir, "per_bot_811_chronological_mask_concat")
    os.makedirs(experiment_output, exist_ok=True)
    os.chdir(experiment_output)
    os.makedirs("ckpt", exist_ok=True)

    run_seeds = base.parse_seeds(args.seeds) if args.run_multi_seed else [args.seed]
    print(
        "[INFO] run_seeds={} split=chronological per-bot 8/1/1 "
        "time_col={} fusion=mask_concat".format(run_seeds, args.time_col)
    )

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
        print("[INFO] All requested seeds are already completed for this mask-concat per-bot split.")

    for seed in pending_seeds:
        print("[INFO] Starting mask-concat seed {} ...".format(seed))
        result = base.run_one_seed(args, seed, bundle)
        results = [row for row in results if int(row["seed"]) != int(seed)]
        results.append(result)

        ckpt_path = base.default_checkpoint_path(args, seed)
        if args.keep_checkpoints:
            print("[INFO] Keeping seed {} checkpoint: {}".format(seed, ckpt_path))
        elif os.path.exists(ckpt_path):
            try:
                ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
                os.remove(ckpt_path)
                print(
                    "[INFO] Deleted seed {} checkpoint after test metrics were obtained: "
                    "{} ({:.1f} MB freed)".format(seed, ckpt_path, ckpt_size_mb)
                )
            except OSError as exc:
                print("[WARN] Could not delete checkpoint for seed {}: {} ({})".format(seed, ckpt_path, exc))

        results = sorted(results, key=lambda row: int(row["seed"]))
        base.save_summary(results, results_csv)
        print("[INFO] Progress saved: completed seeds = {}".format([int(row["seed"]) for row in results]))

    result_by_seed = {int(row["seed"]): row for row in results}
    missing_requested = [seed for seed in run_seeds if seed not in result_by_seed]
    if missing_requested:
        raise RuntimeError("Requested seeds are still missing after training: {}".format(missing_requested))

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
                "split_method": "per-bot chronological post-level 80/10/10",
                "time_col": args.time_col,
                "fusion": "mask_concat",
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
    with open("ckpt/style_scaler.json", "w", encoding="utf-8") as file_obj:
        json.dump({"mean": bundle["scaler"].mean_.tolist(), "scale": bundle["scaler"].scale_.tolist()}, file_obj)
    with open("ckpt/label_encoder_classes.json", "w", encoding="utf-8") as file_obj:
        json.dump({"label_col": args.label_col, "classes": bundle["le"].classes_.tolist()}, file_obj)
    with open("ckpt/role_description_mapping.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "profile_csv": args.profile_csv,
                "role_code_col": bundle["role_code_col"],
                "role_description_col": bundle["role_description_col"],
                "id_to_description": bundle["role_description_mapping"],
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
                "excluded_direct_features": [
                    "followers_count",
                    "friends_count",
                    "verified",
                    "depth",
                    "norm_followers_count",
                    "norm_statuses_count",
                ],
                "activity_mean": bundle["topology_activity_mean"],
                "activity_std": bundle["topology_activity_std"],
                "num_nodes": int(len(bundle["topology_nodes"])),
                "num_edges": int(bundle["edge_index_np"].shape[1]),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print("[DONE] Mask-concat chronological per-bot 8/1/1 outputs saved in {}".format(experiment_output))


if __name__ == "__main__":
    main()
