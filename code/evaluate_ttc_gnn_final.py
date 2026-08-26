# -*- coding: utf-8 -*-
"""Final test evaluation for the selected TTC-GNN grid-search configuration."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import torch
import torch.nn as nn

import train_multiclass_engagement_e2e_5seeds_bot_disjoint_fixed as fixed
import train_multiclass_engagement_e2e_5seeds_with_train_metrics as base
from grid_search_ttc_gnn import add_common_args, config_id, output_path, parse_list


FINAL_FIELDS = [
    "config_id",
    "learning_rate",
    "graph_hidden",
    "graph_layers",
    "att_heads",
    "dropout",
    "seed",
    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "best_val_precision",
    "best_val_recall",
    "best_val_f1",
    "best_val_macro_f1",
    "test_loss",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_macro_f1",
    "training_time",
    "checkpoint_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Final TTC-GNN test evaluation after grid search.")
    add_common_args(parser)
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--summary_csv", default="grid_search_summary.csv")
    parser.add_argument("--grid_results_csv", default="grid_search_results.csv")
    parser.add_argument("--final_results_csv", default="final_test_results.csv")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--graph_hidden", type=int, default=None)
    parser.add_argument("--graph_layers", type=int, default=None)
    parser.add_argument("--att_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the selected configuration instead of evaluating grid-search checkpoints.",
    )
    return parser.parse_args()


def absolutize_args(args) -> None:
    for attr in ("csv_path", "fold_dir", "nodes_csv", "graph_x", "edge_index", "output_dir"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    if args.edges_csv:
        args.edges_csv = os.path.abspath(args.edges_csv)
    if os.path.isdir(args.backbone):
        args.backbone = os.path.abspath(args.backbone)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def selected_config(args, summary_csv: Path, expected_n_seeds: int) -> Dict[str, Any]:
    manual_values = [
        args.learning_rate,
        args.graph_hidden,
        args.graph_layers,
        args.att_heads,
        args.dropout,
    ]
    if any(value is not None for value in manual_values):
        if not all(value is not None for value in manual_values):
            raise ValueError(
                "Manual final evaluation requires all of --learning_rate, --graph_hidden, "
                "--graph_layers, --att_heads, and --dropout."
            )
        return {
            "lr": args.learning_rate,
            "graph_hidden": args.graph_hidden,
            "graph_layers": args.graph_layers,
            "att_heads": args.att_heads,
            "dropout": args.dropout,
        }

    rows = load_csv_rows(summary_csv)
    if not rows:
        raise ValueError("No rows found in summary CSV: {}".format(summary_csv))
    complete_rows = [
        row for row in rows
        if "n_seeds" not in row or int(float(row["n_seeds"])) == expected_n_seeds
    ]
    if not complete_rows:
        raise ValueError(
            "No complete configuration with {} seeds found in summary CSV: {}".format(
                expected_n_seeds,
                summary_csv,
            )
        )
    best = complete_rows[0]
    return {
        "lr": float(best["learning_rate"]),
        "graph_hidden": int(best["graph_hidden"]),
        "graph_layers": int(best["graph_layers"]),
        "att_heads": int(best["att_heads"]),
        "dropout": float(best["dropout"]),
    }


def apply_config(args, config: Dict[str, Any], checkpoint_path: Path):
    run_args = copy.copy(args)
    run_args.lr = config["lr"]
    run_args.graph_hidden = config["graph_hidden"]
    run_args.graph_layers = config["graph_layers"]
    run_args.att_heads = config["att_heads"]
    run_args.dropout = config["dropout"]
    run_args.checkpoint_path = str(checkpoint_path)
    run_args.save_each_seed_cm = False
    return run_args


def grid_checkpoint_rows(grid_results_csv: Path, cid: str) -> Dict[int, Dict[str, str]]:
    rows = load_csv_rows(grid_results_csv)
    selected = {}
    for row in rows:
        if row.get("config_id") == cid and row.get("seed"):
            selected[int(row["seed"])] = row
    return selected


def evaluate_checkpoint(args, seed: int, bundle: Dict[str, Any], grid_row: Dict[str, str]) -> Dict[str, Any]:
    checkpoint_path = grid_row["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Missing grid-search checkpoint: {}".format(checkpoint_path))

    base.set_seed(seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, dl_test = base.make_dataloaders(args, bundle["datasets"], seed, include_test=True)
    graph_x = torch.tensor(bundle["graph_x_np"], dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(bundle["edge_index_np"], dtype=torch.long, device=device)
    role_to_node = bundle["role_to_node_cpu"].to(device)

    model = base.make_model(args, bundle).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = base.evaluate(
        model,
        dl_test,
        graph_x,
        graph_edge_index,
        role_to_node,
        device,
        criterion,
        average=args.metric_average,
        return_preds=False,
    )

    return {
        "seed": seed,
        "best_epoch": int(float(grid_row["best_epoch"])),
        "best_val_loss": float(grid_row["val_loss"]),
        "best_val_acc": float(grid_row["val_acc"]),
        "best_val_precision": float(grid_row["val_precision"]),
        "best_val_recall": float(grid_row["val_recall"]),
        "best_val_f1": float(grid_row["val_f1"]),
        "best_val_macro_f1": float(grid_row["val_macro_f1"]),
        "test_loss": float(test_loss),
        "test_acc": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "training_time": float(grid_row["training_time"]),
        "checkpoint_path": checkpoint_path,
    }


def final_row(cid: str, config: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config_id": cid,
        "learning_rate": config["lr"],
        "graph_hidden": config["graph_hidden"],
        "graph_layers": config["graph_layers"],
        "att_heads": config["att_heads"],
        "dropout": config["dropout"],
        **result,
    }


def write_final_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FINAL_FIELDS} for row in rows)


def print_metric_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n[FINAL TEST SUMMARY]")
    for label, field in [
        ("Accuracy", "test_acc"),
        ("Precision", "test_precision"),
        ("Recall", "test_recall"),
        ("F1", "test_f1"),
        ("Macro-F1", "test_macro_f1"),
    ]:
        values = [float(row[field]) for row in rows]
        spread = stdev(values) if len(values) > 1 else 0.0
        print("{}: {:.4f} +/- {:.4f}".format(label, mean(values), spread))


def main() -> None:
    args = parse_args()
    if args.graph_encoder != "sage":
        raise ValueError("This TTC-GNN final evaluation protocol is defined for --graph_encoder sage.")
    absolutize_args(args)
    seeds = parse_list(args.seeds, int)
    fold_output = Path(args.output_dir) / "fold_{}".format(args.fold_id)
    summary_csv = output_path(args.summary_csv, fold_output)
    grid_results_csv = output_path(args.grid_results_csv, fold_output)
    final_results_csv = output_path(args.final_results_csv, fold_output)

    config = selected_config(args, summary_csv, expected_n_seeds=len(seeds))
    if args.att_hidden % config["att_heads"] != 0:
        raise ValueError("--att_hidden must be divisible by the selected number of attention heads.")
    cid = config_id(config)

    print("[INFO] Final test evaluation is now loading train/val/test split files.")
    print("[INFO] selected_config={}".format(json.dumps(config, sort_keys=True)))
    print("[INFO] config_id={}".format(cid))

    bundle = fixed.prepare_bot_disjoint_data(args, splits=("train", "val", "test"))
    rows = []

    if args.retrain:
        checkpoint_root = fold_output / "final_test_checkpoints" / cid
        for seed in seeds:
            checkpoint_path = checkpoint_root / "seed_{}.pt".format(seed)
            run_args = apply_config(args, config, checkpoint_path)
            result = base.run_one_seed(run_args, seed, bundle)
            rows.append(final_row(cid, config, result))
    else:
        checkpoints_by_seed = grid_checkpoint_rows(grid_results_csv, cid)
        for seed in seeds:
            if seed not in checkpoints_by_seed:
                raise FileNotFoundError(
                    "No grid-search result row found for config_id={} seed={}. "
                    "Run grid search first or pass --retrain.".format(cid, seed)
                )
            run_args = apply_config(args, config, Path(checkpoints_by_seed[seed]["checkpoint_path"]))
            result = evaluate_checkpoint(run_args, seed, bundle, checkpoints_by_seed[seed])
            rows.append(final_row(cid, config, result))

    write_final_results(final_results_csv, rows)
    print_metric_summary(rows)
    print("[DONE] Wrote {}".format(final_results_csv))


if __name__ == "__main__":
    main()
