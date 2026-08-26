# -*- coding: utf-8 -*-
"""
Hyperparameter sweep runner + log parser (Macro-F1).

This script runs:
  (1) Text-side: max_len in {64,128,256} with graph_hidden fixed (default 64)
  (2) Graph-side: graph_hidden in {32,64,128,256} with max_len fixed (default 128)
Each configuration is repeated for 5 seeds (0..4 by default). The paired
training script uses the deterministic per-bot chronological 8/1/1 post split
from train_multiclass_engagement_e2e_5seeds_per_bot_811.py.

It expects the training script to print lines like:
  [TEST] loss=... | acc=... | p=... | r=... | f1=...

Outputs:
  - results/hparam_runs.csv  (per-run)
  - results/hparam_summary.csv (mean±std across seeds)
  - results/plot_maxlen.png, results/plot_graph_hidden.png

Usage:
  python run_hyperparam_sweep_macro_f1.py --train_script train_multiclass_engagement_e2e_hparam.py
"""
import argparse
import csv
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

TEST_LINE_RE = re.compile(r"\[TEST\].*?f1=([0-9]*\.?[0-9]+)")

def run_one(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode

def parse_test_f1(log_path: str) -> float:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # match: [TEST] ... F1=0.7373 or f1=0.7373
    matches = re.findall(r"\[TEST\].*?\bF1\s*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        raise ValueError(f"Cannot find [TEST] ... F1=... in log: {log_path}")
    return float(matches[-1])  # use the last occurrence

def summarize(group_rows, key_fields):
    # group_rows: list[dict] with f1
    f1s = [r["macro_f1"] for r in group_rows]
    return {
        **{k: group_rows[0][k] for k in key_fields},
        "n_seeds": len(f1s),
        "macro_f1_mean": mean(f1s),
        "macro_f1_std": pstdev(f1s) if len(f1s) > 1 else 0.0,
    }

def plot_curve(xs, ys_mean, ys_std, xlabel, out_path):
    plt.figure(figsize=(9, 5))
    plt.errorbar(xs, ys_mean, yerr=ys_std, fmt='-o', capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel("Macro-F1")
    plt.grid(True, which="both", linestyle="-", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, default="train_multiclass_engagement_e2e_hparam.py")
    ap.add_argument("--python", type=str, default="python")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--time_col", type=str, default="create_time")
    ap.add_argument("--split_seed", type=int, default=None, help="Deprecated; ignored because the per-bot chronological split is deterministic.")

    ap.add_argument("--fixed_max_len", type=int, default=128)
    ap.add_argument("--fixed_graph_hidden", type=int, default=64)

    ap.add_argument("--max_len_values", type=str, default="64,128,256")
    ap.add_argument("--graph_hidden_values", type=str, default="32,64,128,256")

    ap.add_argument("--extra_args", type=str, default="", help="extra args passed to training script, e.g. \"--epochs 10\"")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    train_script = Path(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    max_lens = [int(x) for x in args.max_len_values.split(",") if x.strip() != ""]
    ghs = [int(x) for x in args.graph_hidden_values.split(",") if x.strip() != ""]

    extra = args.extra_args.strip().split() if args.extra_args.strip() else []

    out_dir = Path("results")
    log_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    per_run_rows = []

    # Sweep 1: max_len
    for L in max_lens:
        for S in seeds:
            log_path = log_dir / f"maxlen_sweep_maxlen_{L}_gh_{args.fixed_graph_hidden}_seed_{S}_perbot811.log"
            cmd = [args.python, str(train_script),
                   "--max_len", str(L),
                   "--graph_hidden", str(args.fixed_graph_hidden),
                   "--time_col", str(args.time_col),
                   "--seed", str(S)] + extra
            if args.dry_run:
                print("DRY:", " ".join(cmd))
                continue
            rc = run_one(cmd, log_path)
            if rc != 0:
                raise RuntimeError(f"Training failed (rc={rc}). See {log_path}")
            f1 = parse_test_f1(log_path)
            per_run_rows.append({
                "sweep": "max_len",
                "max_len": L,
                "graph_hidden": args.fixed_graph_hidden,
                "seed": S,
                "macro_f1": f1,
                "log_path": str(log_path),
            })

    # Sweep 2: graph_hidden
    for H in ghs:
        for S in seeds:
            log_path = log_dir / f"graphhidden_sweep_maxlen_{args.fixed_max_len}_gh_{H}_seed_{S}_perbot811.log"
            cmd = [args.python, str(train_script),
                   "--max_len", str(args.fixed_max_len),
                   "--graph_hidden", str(H),
                   "--time_col", str(args.time_col),
                   "--seed", str(S)] + extra
            if args.dry_run:
                print("DRY:", " ".join(cmd))
                continue
            rc = run_one(cmd, log_path)
            if rc != 0:
                raise RuntimeError(f"Training failed (rc={rc}). See {log_path}")
            f1 = parse_test_f1(log_path)
            per_run_rows.append({
                "sweep": "graph_hidden",
                "max_len": args.fixed_max_len,
                "graph_hidden": H,
                "seed": S,
                "macro_f1": f1,
                "log_path": str(log_path),
            })

    if args.dry_run:
        return

    # Write per-run CSV
    per_run_csv = out_dir / "hparam_runs.csv"
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"[OK] wrote {per_run_csv}")

    # Summaries
    summary_rows = []

    # max_len summary
    for L in max_lens:
        rows = [r for r in per_run_rows if r["sweep"] == "max_len" and r["max_len"] == L]
        summary_rows.append(summarize(rows, key_fields=["sweep", "max_len", "graph_hidden"]))

    # graph_hidden summary
    for H in ghs:
        rows = [r for r in per_run_rows if r["sweep"] == "graph_hidden" and r["graph_hidden"] == H]
        summary_rows.append(summarize(rows, key_fields=["sweep", "max_len", "graph_hidden"]))

    summary_csv = out_dir / "hparam_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[OK] wrote {summary_csv}")

    # Plots
    # max_len
    xs = max_lens
    means = [next(r for r in summary_rows if r["sweep"] == "max_len" and r["max_len"] == L)["macro_f1_mean"] for L in xs]
    stds = [next(r for r in summary_rows if r["sweep"] == "max_len" and r["max_len"] == L)["macro_f1_std"] for L in xs]
    plot_curve(xs, means, stds, xlabel="max_len", out_path=out_dir / "plot_maxlen.png")
    print(f"[OK] saved {out_dir / 'plot_maxlen.png'}")

    # graph_hidden
    xs = ghs
    means = [next(r for r in summary_rows if r["sweep"] == "graph_hidden" and r["graph_hidden"] == H)["macro_f1_mean"] for H in xs]
    stds = [next(r for r in summary_rows if r["sweep"] == "graph_hidden" and r["graph_hidden"] == H)["macro_f1_std"] for H in xs]
    plot_curve(xs, means, stds, xlabel="graph_hidden", out_path=out_dir / "plot_graph_hidden.png")
    print(f"[OK] saved {out_dir / 'plot_graph_hidden.png'}")

if __name__ == "__main__":
    main()
