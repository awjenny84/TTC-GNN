# -*- coding: utf-8 -*-
"""Validation-only grid search for the TTC-GNN bot-disjoint trainer.

The test fold is intentionally not loaded by this script. It prepares only the
training and validation datasets, checkpoints each seed by validation Macro-F1,
and writes resumable per-run and per-config CSV outputs.
"""

import argparse
import copy
import csv
import itertools
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_multiclass_engagement_e2e_5seeds_bot_disjoint_fixed as fixed
import train_multiclass_engagement_e2e_5seeds_with_train_metrics as base


RESULT_FIELDS = [
    "config_id",
    "learning_rate",
    "graph_hidden",
    "graph_layers",
    "att_heads",
    "dropout",
    "seed",
    "best_epoch",
    "val_loss",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_macro_f1",
    "training_time",
    "checkpoint_path",
]

SUMMARY_FIELDS = [
    "config_id",
    "learning_rate",
    "graph_hidden",
    "graph_layers",
    "att_heads",
    "dropout",
    "n_seeds",
    "mean_val_macro_f1",
    "std_val_macro_f1",
    "mean_acc",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "average_best_epoch",
]


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def slug_float(value: float) -> str:
    text = "{:.12g}".format(float(value))
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def config_id(config: Dict[str, Any]) -> str:
    return "lr{}_gh{}_gl{}_heads{}_drop{}".format(
        slug_float(config["lr"]),
        config["graph_hidden"],
        config["graph_layers"],
        config["att_heads"],
        slug_float(config["dropout"]),
    )


def build_grid(args) -> List[Dict[str, Any]]:
    lr_values = parse_list(args.lr_values, float)
    graph_hidden_values = parse_list(args.graph_hidden_values, int)
    graph_layers_values = parse_list(args.graph_layers_values, int)
    att_heads_values = parse_list(args.att_heads_values, int)
    dropout_values = parse_list(args.dropout_values, float)

    for heads in att_heads_values:
        if args.att_hidden % heads != 0:
            raise ValueError(
                "--att_hidden ({}) must be divisible by every --att_heads_values entry; got {}.".format(
                    args.att_hidden,
                    heads,
                )
            )

    configs = []
    for lr, graph_hidden, graph_layers, att_heads, dropout in itertools.product(
        lr_values,
        graph_hidden_values,
        graph_layers_values,
        att_heads_values,
        dropout_values,
    ):
        configs.append({
            "lr": lr,
            "graph_hidden": graph_hidden,
            "graph_layers": graph_layers,
            "att_heads": att_heads,
            "dropout": dropout,
        })
    return configs


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv_path", default="aigc_new_with_style_features.csv")
    parser.add_argument("--fold_dir", default=".")
    parser.add_argument("--fold_id", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--fold_suffix", default="")
    parser.add_argument("--output_dir", default="ttc_gnn_grid_search_outputs")
    parser.add_argument("--label_col", default="engagement_class")
    parser.add_argument("--text_col", default="text_raw")
    parser.add_argument("--role_col", default="role_id")
    parser.add_argument("--nodes_csv", default="nodes1.csv")
    parser.add_argument("--nodes_encoding", default="gbk")
    parser.add_argument("--graph_x", default="node_features_raw.npy")
    parser.add_argument("--edge_index", default="edge_index.npy")
    parser.add_argument("--edges_csv", default="")
    parser.add_argument("--build_graph_from_csv", action="store_true")
    parser.add_argument(
        "--graph_feature_cols",
        default="followers_count,statuses_count,verified,friends_count,norm_followers_count,norm_statuses_count,depth",
    )
    parser.add_argument("--scale_graph_features", action="store_true")
    parser.add_argument(
        "--feature_cols",
        default="length,emoji_count,is_qa,sentiment,ttr,rttr,mtld,msttr,common_ratio,stop_ratio",
    )
    parser.add_argument("--backbone", default="chinese-roberta-wwm-ext")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--metric_average", default="macro")
    parser.add_argument("--graph_encoder", default="sage", choices=["gt", "sage"])
    parser.add_argument("--early_patience", type=int, default=99)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--lambda_unsup", type=float, default=0.0)
    parser.add_argument("--unsup_every", type=int, default=1)
    parser.add_argument("--rw_len", type=int, default=5)
    parser.add_argument("--rw_pos", type=int, default=5)
    parser.add_argument("--rw_neg", type=int, default=10)
    parser.add_argument("--text_cond_dim", type=int, default=64)
    parser.add_argument("--style_proj_dim", type=int, default=64)
    parser.add_argument("--film_hidden", type=int, default=256)
    parser.add_argument("--att_hidden", type=int, default=256)
    parser.add_argument("--pred_hidden", type=int, default=256)


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only TTC-GNN grid search.")
    add_common_args(parser)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--lr_values", default="1e-3,5e-4")
    parser.add_argument("--graph_hidden_values", default="64,128")
    parser.add_argument("--graph_layers_values", default="1,2")
    parser.add_argument("--att_heads_values", default="2,4")
    parser.add_argument("--dropout_values", default="0.1,0.3")
    parser.add_argument("--results_csv", default="grid_search_results.csv")
    parser.add_argument("--summary_csv", default="grid_search_summary.csv")
    parser.add_argument("--best_config_json", default="best_config.json")
    return parser.parse_args()


def absolutize_args(args) -> None:
    for attr in ("csv_path", "fold_dir", "nodes_csv", "graph_x", "edge_index", "output_dir"):
        setattr(args, attr, os.path.abspath(getattr(args, attr)))
    if args.edges_csv:
        args.edges_csv = os.path.abspath(args.edges_csv)
    if os.path.isdir(args.backbone):
        args.backbone = os.path.abspath(args.backbone)


def output_path(raw_path: str, fold_output: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return fold_output / path


def load_result_rows(results_csv: Path) -> List[Dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def completed_keys(rows: Iterable[Dict[str, str]]) -> set:
    keys = set()
    for row in rows:
        checkpoint_path = row.get("checkpoint_path", "")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            continue
        if row.get("config_id") and row.get("seed"):
            keys.add((row["config_id"], int(row["seed"])))
    return keys


def append_result(results_csv: Path, row: Dict[str, Any]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def to_float(row: Dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else math.nan


def summarize_results(results_csv: Path, summary_csv: Path) -> List[Dict[str, Any]]:
    rows = load_result_rows(results_csv)
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["config_id"],
            row["learning_rate"],
            row["graph_hidden"],
            row["graph_layers"],
            row["att_heads"],
            row["dropout"],
        )
        grouped[key].append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        macro_values = [to_float(row, "val_macro_f1") for row in group_rows]
        acc_values = [to_float(row, "val_acc") for row in group_rows]
        precision_values = [to_float(row, "val_precision") for row in group_rows]
        recall_values = [to_float(row, "val_recall") for row in group_rows]
        f1_values = [to_float(row, "val_f1") for row in group_rows]
        epoch_values = [to_float(row, "best_epoch") for row in group_rows]
        summary_rows.append({
            "config_id": key[0],
            "learning_rate": key[1],
            "graph_hidden": key[2],
            "graph_layers": key[3],
            "att_heads": key[4],
            "dropout": key[5],
            "n_seeds": len(group_rows),
            "mean_val_macro_f1": mean(macro_values),
            "std_val_macro_f1": stdev(macro_values) if len(macro_values) > 1 else 0.0,
            "mean_acc": mean(acc_values),
            "mean_precision": mean(precision_values),
            "mean_recall": mean(recall_values),
            "mean_f1": mean(f1_values),
            "average_best_epoch": mean(epoch_values),
        })

    summary_rows.sort(key=lambda row: row["mean_val_macro_f1"], reverse=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


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


def get_arg(args, name: str, default):
    return getattr(args, name, default)


def make_validation_dataloaders(args, datasets, seed: int):
    if hasattr(base, "make_dataloaders"):
        try:
            return base.make_dataloaders(args, datasets, seed, include_test=False)
        except TypeError as exc:
            if "include_test" not in str(exc):
                raise
            if "test" in datasets:
                loaders = base.make_dataloaders(args, datasets, seed)
                return loaders[0], loaders[1]
            print("[WARN] Base make_dataloaders requires a test split; building train/val loaders locally.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init_fn = getattr(base, "seed_worker", None)
    dl_train = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    dl_val = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    return dl_train, dl_val


def make_grid_model(args, bundle):
    if hasattr(base, "make_model"):
        return base.make_model(args, bundle)

    kwargs = {
        "backbone": args.backbone,
        "num_roles": bundle["num_roles"],
        "style_in_dim": len(bundle["feat_cols"]),
        "graph_in_dim": bundle["graph_in_dim"],
        "num_classes": bundle["num_classes"],
        "graph_encoder": args.graph_encoder,
        "graph_hidden": args.graph_hidden,
        "graph_layers": get_arg(args, "graph_layers", 2),
        "att_heads": get_arg(args, "att_heads", 4),
        "dropout": get_arg(args, "dropout", 0.1),
        "text_cond_dim": get_arg(args, "text_cond_dim", 64),
        "style_proj_dim": get_arg(args, "style_proj_dim", 64),
        "film_hidden": get_arg(args, "film_hidden", 256),
        "att_hidden": get_arg(args, "att_hidden", 256),
        "pred_hidden": get_arg(args, "pred_hidden", 256),
    }
    try:
        return base.EngagementE2EModel(**kwargs)
    except TypeError:
        for key in ("graph_layers", "att_heads", "dropout", "text_cond_dim", "style_proj_dim", "film_hidden", "att_hidden", "pred_hidden"):
            kwargs.pop(key, None)
        return base.EngagementE2EModel(**kwargs)


def checkpoint_payload(args, bundle, seed: int):
    if hasattr(base, "checkpoint_payload"):
        return base.checkpoint_payload(args, bundle, seed)

    return {
        "backbone": args.backbone,
        "num_roles": bundle["num_roles"],
        "num_classes": bundle["num_classes"],
        "label_classes": bundle["le"].classes_.tolist(),
        "feature_cols": bundle["feat_cols"],
        "graph_encoder": args.graph_encoder,
        "graph_hidden": args.graph_hidden,
        "seed": seed,
        "split_seed": args.split_seed,
    }


def ensure_macro_f1(metrics: Dict[str, float], average: str) -> Dict[str, float]:
    metrics = dict(metrics)
    if "macro_f1" not in metrics:
        if average != "macro":
            print("[WARN] evaluate() returned no macro_f1; using f1 from average='{}' as val_macro_f1.".format(average))
        metrics["macro_f1"] = metrics["f1"]
    return metrics


def run_one_seed_validation_only_fallback(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    print("\n{} Grid-search seed={} {}".format("=" * 20, seed, "=" * 20))
    started_at = time.perf_counter()
    base.set_seed(seed, deterministic=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_train, dl_val = make_validation_dataloaders(args, bundle["datasets"], seed)

    graph_x = torch.tensor(bundle["graph_x_np"], dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(bundle["edge_index_np"], dtype=torch.long, device=device)
    role_to_node = bundle["role_to_node_cpu"].to(device)
    adj = bundle["adj"]
    rng = np.random.default_rng(seed)

    missing = torch.where(bundle["role_to_node_cpu"] < 0)[0].tolist()
    if missing:
        print("[WARN] role_id {} missing from nodes; using g_null for those roles.".format(missing))

    model = make_grid_model(args, bundle).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = base.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_loss = None
    best_val_metrics = None
    global_step = 0
    no_improve = 0
    ckpt_path = str(get_arg(args, "checkpoint_path", "grid_search_seed_{}.pt".format(seed)))

    for epoch in range(args.epochs):
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
                if not hasattr(base, "sample_rw_positives") or not hasattr(base, "unsup_rw_neg_sampling_loss"):
                    raise AttributeError("Base trainer lacks random-walk unsupervised loss helpers; set --lambda_unsup 0.")
                roots = role_to_node[torch.arange(bundle["num_roles"], device=device)]
                roots = roots[roots >= 0]
                if roots.numel() > 0:
                    pos_nodes = base.sample_rw_positives(roots, adj, args.rw_len, args.rw_pos, rng)
                    loss_unsup = base.unsup_rw_neg_sampling_loss(z_nodes, roots, pos_nodes, args.rw_neg)
                    loss = loss + args.lambda_unsup * loss_unsup

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size
            global_step += 1

        train_loss = total_loss / max(1, n)
        val_loss, val_metrics = base.evaluate(
            model,
            dl_val,
            graph_x,
            graph_edge_index,
            role_to_node,
            device,
            criterion,
            average=args.metric_average,
        )
        val_metrics = ensure_macro_f1(val_metrics, args.metric_average)

        print(
            "[Grid seed={} | Epoch {}/{}] train_loss={:.4f} | val_loss={:.4f} | "
            "val_acc={:.4f} val_P={:.4f} val_R={:.4f} val_F1={:.4f} ({}) val_MacroF1={:.4f}".format(
                seed,
                epoch + 1,
                args.epochs,
                train_loss,
                val_loss,
                val_metrics["acc"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                args.metric_average,
                val_metrics["macro_f1"],
            )
        )

        improved = val_metrics["macro_f1"] > (best_val_macro_f1 + args.early_min_delta)
        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_val_loss = float(val_loss)
            best_val_metrics = dict(val_metrics)
            no_improve = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            payload = checkpoint_payload(args, bundle, seed)
            payload.update({
                "model": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_metrics": best_val_metrics,
            })
            torch.save(payload, ckpt_path)
        else:
            no_improve += 1
            print("[EarlyStop] no_improve={}/{} (best_macro_f1={:.4f} @ epoch {})".format(
                no_improve,
                args.early_patience,
                best_val_macro_f1,
                best_epoch,
            ))
            if no_improve >= args.early_patience:
                print("[EarlyStop] Stop at epoch {}. Best epoch={}, best_val_macro_f1={:.4f}".format(
                    epoch + 1,
                    best_epoch,
                    best_val_macro_f1,
                ))
                break

    if best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was saved. Check that --epochs is greater than 0.")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_loss": float(best_val_loss),
        "val_acc": float(best_val_metrics["acc"]),
        "val_precision": float(best_val_metrics["precision"]),
        "val_recall": float(best_val_metrics["recall"]),
        "val_f1": float(best_val_metrics["f1"]),
        "val_macro_f1": float(best_val_macro_f1),
        "training_time": float(time.perf_counter() - started_at),
        "checkpoint_path": ckpt_path,
    }


def run_validation_only(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    runner = getattr(base, "run_one_seed_validation_only", None)
    if callable(runner):
        return runner(args, seed, bundle)
    print("[WARN] Base trainer has no run_one_seed_validation_only; using grid_search_ttc_gnn fallback runner.")
    return run_one_seed_validation_only_fallback(args, seed, bundle)


def main() -> None:
    args = parse_args()
    if args.graph_encoder != "sage":
        raise ValueError("This TTC-GNN grid-search protocol is defined for --graph_encoder sage.")
    absolutize_args(args)

    seeds = parse_list(args.seeds, int)
    configs = build_grid(args)
    fold_output = Path(args.output_dir) / "fold_{}".format(args.fold_id)
    results_csv = output_path(args.results_csv, fold_output)
    summary_csv = output_path(args.summary_csv, fold_output)
    best_config_json = output_path(args.best_config_json, fold_output)
    checkpoint_root = fold_output / "checkpoints"

    print("[INFO] Grid search uses only train and validation split files.")
    print("[INFO] fold_dir={} fold_id={} fold_suffix='{}'".format(args.fold_dir, args.fold_id, args.fold_suffix))
    print("[INFO] configs={} seeds={}".format(len(configs), seeds))
    print("[INFO] results_csv={}".format(results_csv))
    print("[INFO] summary_csv={}".format(summary_csv))

    bundle = fixed.prepare_bot_disjoint_data(args, splits=("train", "val"))
    rows = load_result_rows(results_csv)
    done = completed_keys(rows)

    total_runs = len(configs) * len(seeds)
    completed_count = len(done)
    for config_index, config in enumerate(configs, start=1):
        cid = config_id(config)
        for seed in seeds:
            run_key = (cid, seed)
            if run_key in done:
                print("[SKIP] completed config={} seed={}".format(cid, seed))
                continue

            checkpoint_path = checkpoint_root / cid / "seed_{}.pt".format(seed)
            print(
                "[RUN] {}/{} config={} seed={} lr={} graph_hidden={} graph_layers={} heads={} dropout={}".format(
                    completed_count + 1,
                    total_runs,
                    cid,
                    seed,
                    config["lr"],
                    config["graph_hidden"],
                    config["graph_layers"],
                    config["att_heads"],
                    config["dropout"],
                )
            )
            run_args = apply_config(args, config, checkpoint_path)
            result = run_validation_only(run_args, seed, bundle)
            row = {
                "config_id": cid,
                "learning_rate": config["lr"],
                "graph_hidden": config["graph_hidden"],
                "graph_layers": config["graph_layers"],
                "att_heads": config["att_heads"],
                "dropout": config["dropout"],
                **result,
            }
            append_result(results_csv, row)
            done.add(run_key)
            completed_count += 1
            summarize_results(results_csv, summary_csv)

    summary_rows = summarize_results(results_csv, summary_csv)
    complete_rows = [row for row in summary_rows if int(row["n_seeds"]) == len(seeds)]
    if not complete_rows:
        print("[WARN] No complete hyperparameter configuration has all {} seeds yet.".format(len(seeds)))
        return

    best = complete_rows[0]
    with best_config_json.open("w", encoding="utf-8") as file_obj:
        json.dump(best, file_obj, indent=2)

    print("\n[BEST CONFIG]")
    print(json.dumps(best, indent=2))
    print("[DONE] Wrote {} and {}".format(results_csv, summary_csv))


if __name__ == "__main__":
    main()
