# -*- coding: utf-8 -*-
"""
Plot validation metrics vs. epoch for the max_len comparison.

This script only reads training logs and draws figures. It does not retrain a
model, change model layers, or change the data split. Defaults are aligned with
the main experiments:

- split: per-bot chronological 8/1/1
- seeds: 42,52,62,72,82
- fixed graph_hidden: 64
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MAX_LENS = [64, 128, 256]
DEFAULT_GRAPH_HIDDEN = 64
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_LOG_DIR = "logs"
DEFAULT_OUT_DIR = "results"
DEFAULT_LOG_TEMPLATE = "maxlen_{max_len}_gh_{graph_hidden}_seed_{seed}.log"
SPLIT_NOTE = "per-bot chronological 8/1/1"


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    try:
        parsed = [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated integer list"
        ) from exc
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot validation curves from logs while keeping the model structure "
            "and per-bot chronological 8/1/1 split consistent with the main "
            "experiments."
        )
    )
    parser.add_argument(
        "--max_lens",
        type=parse_int_list,
        default=DEFAULT_MAX_LENS,
        help="Comma-separated max_len values to compare. Default: 64,128,256",
    )
    parser.add_argument(
        "--graph_hidden",
        type=int,
        default=DEFAULT_GRAPH_HIDDEN,
        help="Fixed graph_hidden used by all plotted runs. Default: 64",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated run seeds. Default: 42,52,62,72,82",
    )
    parser.add_argument(
        "--log_dir",
        default=DEFAULT_LOG_DIR,
        help="Directory containing training logs. Default: logs",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for the generated figure. Default: results",
    )
    parser.add_argument(
        "--log_template",
        default=DEFAULT_LOG_TEMPLATE,
        help=(
            "Log filename template. Available fields: max_len, graph_hidden, seed. "
            "Default: maxlen_{max_len}_gh_{graph_hidden}_seed_{seed}.log"
        ),
    )
    return parser.parse_args()


def build_log_name(template: str, max_len: int, graph_hidden: int, seed: int) -> str:
    return template.format(max_len=max_len, graph_hidden=graph_hidden, seed=seed)


def build_pattern(metric_key: str):
    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    return re.compile(
        rf"\[(?:Seed\s+\d+\s+\|\s*)?Epoch\s+(\d+)/(\d+)\].*?"
        rf"{re.escape(metric_key)}={number}",
        re.IGNORECASE,
    )


def parse_curve(log_path: Path, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    pattern = build_pattern(metric_key)
    epochs, values = [], []
    with log_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            match = pattern.search(line)
            if not match:
                continue
            epochs.append(int(match.group(1)))
            values.append(float(match.group(3)))

    if not epochs:
        raise ValueError("No {} values found in {}".format(metric_key, log_path))
    return np.array(epochs, dtype=int), np.array(values, dtype=float)


def align_curves(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    max_epoch = max(int(epochs.max()) for epochs, _ in curves)
    epoch_grid = np.arange(1, max_epoch + 1)
    matrix = np.full((len(curves), max_epoch), np.nan, dtype=float)

    for row_idx, (epochs, values) in enumerate(curves):
        for epoch, value in zip(epochs, values):
            if 1 <= epoch <= max_epoch:
                matrix[row_idx, epoch - 1] = value

    return epoch_grid, matrix


def nanmean_sample_std(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix, axis=0)
    counts = np.sum(np.isfinite(matrix), axis=0)
    centered = matrix - mean
    sum_square = np.nansum(centered * centered, axis=0)
    std = np.zeros_like(mean)
    valid = counts > 1
    std[valid] = np.sqrt(sum_square[valid] / (counts[valid] - 1))
    return mean, std


def collect_missing_logs(
    log_dir: Path,
    log_template: str,
    max_lens: Iterable[int],
    graph_hidden: int,
    seeds: Iterable[int],
) -> List[Path]:
    missing = []
    for max_len in max_lens:
        for seed in seeds:
            name = build_log_name(log_template, max_len, graph_hidden, seed)
            path = log_dir / name
            if not path.exists():
                missing.append(path)
    return missing


def output_filename(max_lens: Sequence[int], graph_hidden: int, seeds: Sequence[int]) -> str:
    max_len_part = "-".join(str(value) for value in max_lens)
    seed_part = "-".join(str(seed) for seed in seeds)
    return (
        "val_metrics_2x2_per_bot_811_"
        "gh{}_maxlen{}_seeds{}.png".format(graph_hidden, max_len_part, seed_part)
    )


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_logs = collect_missing_logs(
        log_dir=log_dir,
        log_template=args.log_template,
        max_lens=args.max_lens,
        graph_hidden=args.graph_hidden,
        seeds=args.seeds,
    )
    if missing_logs:
        preview = "\n".join("  {}".format(path) for path in missing_logs[:20])
        more = "" if len(missing_logs) <= 20 else "\n  ... {} more".format(len(missing_logs) - 20)
        raise FileNotFoundError(
            "Missing {} expected log files for split={} and seeds={}:\n{}{}".format(
                len(missing_logs),
                SPLIT_NOTE,
                ",".join(str(seed) for seed in args.seeds),
                preview,
                more,
            )
        )

    metrics = [
        ("val_acc", "Accuracy"),
        ("val_P", "Precision"),
        ("val_R", "Recall"),
        ("val_MacroF1", "Macro-F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for axis, (metric_key, metric_name) in zip(axes, metrics):
        for max_len in args.max_lens:
            curves = []
            for seed in args.seeds:
                log_name = build_log_name(
                    args.log_template,
                    max_len=max_len,
                    graph_hidden=args.graph_hidden,
                    seed=seed,
                )
                curves.append(parse_curve(log_dir / log_name, metric_key))

            epoch_grid, matrix = align_curves(curves)
            mean, std = nanmean_sample_std(matrix)

            axis.plot(epoch_grid, mean, label="max_len={}".format(max_len))
            axis.fill_between(epoch_grid, mean - std, mean + std, alpha=0.15)

        axis.set_title(metric_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation {}".format(metric_name))
        axis.grid(True, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(args.max_lens))
    fig.suptitle(
        "{}; graph_hidden={}; seeds={}".format(
            SPLIT_NOTE,
            args.graph_hidden,
            ",".join(str(seed) for seed in args.seeds),
        ),
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig_path = out_dir / output_filename(args.max_lens, args.graph_hidden, args.seeds)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)


if __name__ == "__main__":
    main()
