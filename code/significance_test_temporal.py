# -*- coding: utf-8 -*-
"""
Paired significance tests for TTC-GNN experiments.

This script is specifically for experiments that use the same per-bot
chronological 8/1/1 split and the same random seeds across Full TTC-GNN and
all baselines/ablations. Because the results are paired by seed, this script
uses paired comparisons only. It never uses an independent t-test.

It does not retrain any model and does not modify training code. It only reads
existing multi_seed_results.csv files.
"""

import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


FULL_MODEL_NAME = "TTC-GNN"
DEFAULT_EXPECTED_SEEDS = "42"
REQUIRED_COLUMNS = [
    "seed",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_macro_f1",
]
TEST_METRICS = [
    "test_acc",
    "test_precision",
    "test_recall",
    "test_macro_f1",
]
ALPHA = 0.05
ZERO_TOL = 1e-12


def parse_seed_list(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise ValueError("--expected_seeds must contain at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--expected_seeds contains duplicates: {}".format(seeds))
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paired significance tests for identical chronological 8/1/1 split "
            "and identical random seeds."
        )
    )
    parser.add_argument("--full", required=True, help="Full TTC-GNN multi_seed_results.csv")
    parser.add_argument("--pce", help="PCE baseline multi_seed_results.csv")
    parser.add_argument("--tce", help="TCE baseline multi_seed_results.csv")
    parser.add_argument("--bert", help="BERT baseline multi_seed_results.csv")
    parser.add_argument("--roberta", help="RoBERTa baseline multi_seed_results.csv")
    parser.add_argument(
        "--baseline",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        default=[],
        help="Additional baseline/ablation, e.g. --baseline STYLE /path/to/multi_seed_results.csv",
    )
    parser.add_argument("--output_dir", default="significance_results")
    parser.add_argument(
        "--expected_seeds",
        default=DEFAULT_EXPECTED_SEEDS,
        help=(
            "Comma-separated paired seeds to require. Default is 42 for the "
            "initial one-seed check; use 42,52,62,72,82 for the full five-seed test."
        ),
    )
    parser.add_argument(
        "--strict_seed_match",
        action="store_true",
        help=(
            "Require each CSV to contain exactly --expected_seeds and no extra seeds. "
            "By default, extra seeds are ignored so a five-seed CSV can be used for "
            "a seed=42-only check."
        ),
    )
    return parser.parse_args()


def normalize_seed_column(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    seed_numeric = pd.to_numeric(frame["seed"], errors="coerce")
    bad_seed = seed_numeric.isna()
    if bad_seed.any():
        bad_values = frame.loc[bad_seed, "seed"].tolist()
        raise ValueError(
            "{} has non-numeric seed values: {}".format(model_name, bad_values)
        )

    not_integer = ~np.isclose(seed_numeric, np.round(seed_numeric), atol=ZERO_TOL)
    if not_integer.any():
        bad_values = frame.loc[not_integer, "seed"].tolist()
        raise ValueError(
            "{} has non-integer seed values: {}".format(model_name, bad_values)
        )

    out = frame.copy()
    out["seed"] = seed_numeric.round().astype(int)
    return out


def read_results_csv(
    model_name: str,
    csv_path: str,
    expected_seeds: List[int],
    strict_seed_match: bool = False,
) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("{} CSV not found: {}".format(model_name, csv_path))

    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "{} is missing required columns {} in {}".format(
                model_name, missing_columns, csv_path
            )
        )

    frame = normalize_seed_column(frame, model_name)
    selected_mask = frame["seed"].isin(expected_seeds)
    duplicate_selected_seeds = frame.loc[
        selected_mask & frame["seed"].duplicated(keep=False), "seed"
    ].tolist()
    if duplicate_selected_seeds:
        raise ValueError(
            "{} has duplicate selected seed rows: {}".format(
                model_name, sorted(set(duplicate_selected_seeds))
            )
        )

    actual_seeds = set(frame["seed"].tolist())
    expected_set = set(expected_seeds)
    missing_seeds = sorted(expected_set - actual_seeds)
    extra_seeds = sorted(actual_seeds - expected_set)
    if missing_seeds or (strict_seed_match and extra_seeds):
        raise ValueError(
            "{} seed mismatch. Expected {}; missing={}; extra={}".format(
                model_name, expected_seeds, missing_seeds, extra_seeds
            )
        )

    for metric in REQUIRED_COLUMNS:
        if metric == "seed":
            continue
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        bad_metric = frame[metric].isna() | ~np.isfinite(frame[metric])
        if bad_metric.any():
            bad_rows = frame.loc[bad_metric, ["seed", metric]].to_dict("records")
            raise ValueError(
                "{} has non-finite values for {}: {}".format(
                    model_name, metric, bad_rows
                )
            )

    aligned = frame.loc[selected_mask].set_index("seed").loc[expected_seeds].reset_index()
    return aligned


def collect_baselines(args) -> List[Tuple[str, str]]:
    baselines = []
    for name, path in (
        ("PCE", args.pce),
        ("TCE", args.tce),
        ("BERT", args.bert),
        ("RoBERTa", args.roberta),
    ):
        if path:
            baselines.append((name, path))

    for name, path in args.baseline:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("--baseline NAME cannot be empty.")
        baselines.append((clean_name, path))

    if not baselines:
        raise ValueError(
            "No baseline CSVs were provided. Use --pce/--tce/--bert/--roberta "
            "or add extra comparisons with --baseline NAME CSV."
        )

    seen = set()
    for name, _ in baselines:
        key = name.lower()
        if key in seen:
            raise ValueError("Duplicate baseline name: {}".format(name))
        seen.add(key)
    return baselines


def mean_std(values: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(values))
    std = sample_std(values)
    return mean, std


def sample_std(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def format_float(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NaN"
    return "{:.{}f}".format(float(value), digits)


def format_mean_std(mean: float, std: float) -> str:
    return "{} ± {}".format(format_float(mean), format_float(std))


def paired_wilcoxon(full_values: np.ndarray, baseline_values: np.ndarray) -> Tuple[float, float, str]:
    differences = full_values - baseline_values
    if np.all(np.isclose(differences, 0.0, atol=ZERO_TOL)):
        return 0.0, 1.0, "all paired differences are zero; set W=0 and p=1"

    try:
        # zero_method="pratt" keeps zero differences in the ranking process and
        # is a conservative, standard choice when some paired differences are
        # exactly zero. All-zero differences are handled above because SciPy may
        # raise for that degenerate case.
        result = stats.wilcoxon(
            full_values,
            baseline_values,
            alternative="two-sided",
            zero_method="pratt",
        )
        return float(result.statistic), float(result.pvalue), ""
    except Exception as exc:
        return float("nan"), float("nan"), "Wilcoxon failed: {}".format(exc)


def paired_ttest(full_values: np.ndarray, baseline_values: np.ndarray) -> Tuple[float, float, str]:
    differences = full_values - baseline_values
    if len(differences) < 2:
        return (
            float("nan"),
            float("nan"),
            "paired t-test requires at least two paired seeds; only {} provided".format(
                len(differences)
            ),
        )
    if np.all(np.isclose(differences, 0.0, atol=ZERO_TOL)):
        return 0.0, 1.0, "all paired differences are zero; set t=0 and p=1"

    try:
        result = stats.ttest_rel(full_values, baseline_values)
        return float(result.statistic), float(result.pvalue), ""
    except Exception as exc:
        return float("nan"), float("nan"), "paired t-test failed: {}".format(exc)


def cohens_dz(differences: np.ndarray) -> Tuple[float, str]:
    diff_std = sample_std(differences)
    if not np.isfinite(diff_std) or math.isclose(diff_std, 0.0, abs_tol=ZERO_TOL):
        return float("nan"), "std(differences, ddof=1) is zero; Cohen's dz is undefined"
    return float(np.mean(differences) / diff_std), ""


def holm_bonferroni(pvals: Iterable[float], alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    pvals_array = np.asarray(list(pvals), dtype=float)
    adjusted = np.full_like(pvals_array, np.nan, dtype=float)
    rejected = np.zeros(pvals_array.shape, dtype=bool)

    finite_mask = np.isfinite(pvals_array)
    if not finite_mask.any():
        return rejected, adjusted

    finite_pvals = pvals_array[finite_mask]
    try:
        from statsmodels.stats.multitest import multipletests

        reject_finite, adjusted_finite, _, _ = multipletests(
            finite_pvals,
            alpha=alpha,
            method="holm",
        )
    except Exception:
        order = np.argsort(finite_pvals)
        sorted_p = finite_pvals[order]
        m = len(sorted_p)
        sorted_adjusted = np.empty(m, dtype=float)
        running_max = 0.0
        for rank, raw_p in enumerate(sorted_p):
            holm_p = min((m - rank) * raw_p, 1.0)
            running_max = max(running_max, holm_p)
            sorted_adjusted[rank] = min(running_max, 1.0)
        adjusted_finite = np.empty(m, dtype=float)
        adjusted_finite[order] = sorted_adjusted
        reject_finite = adjusted_finite <= alpha

    adjusted[finite_mask] = adjusted_finite
    rejected[finite_mask] = reject_finite
    return rejected, adjusted


def descriptive_stats(model_frames: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    stats_by_model_metric = {}
    for model_name, frame in model_frames.items():
        for metric in TEST_METRICS:
            stats_by_model_metric[(model_name, metric)] = mean_std(frame[metric].to_numpy(dtype=float))
    return stats_by_model_metric


def build_comparison_rows(
    full_frame: pd.DataFrame,
    baseline_frames: Dict[str, pd.DataFrame],
    model_stats: Dict[Tuple[str, str], Tuple[float, float]],
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    paired_tables: Dict[Tuple[str, str], pd.DataFrame] = {}

    for baseline_name, baseline_frame in baseline_frames.items():
        comparison = "{} vs {}".format(FULL_MODEL_NAME, baseline_name)
        for metric in TEST_METRICS:
            full_values = full_frame[metric].to_numpy(dtype=float)
            baseline_values = baseline_frame[metric].to_numpy(dtype=float)
            differences = full_values - baseline_values

            wins = int(np.sum(differences > ZERO_TOL))
            losses = int(np.sum(differences < -ZERO_TOL))
            ties = int(len(differences) - wins - losses)

            wilcoxon_stat, raw_p, wilcoxon_note = paired_wilcoxon(
                full_values, baseline_values
            )
            t_stat, t_p, ttest_note = paired_ttest(full_values, baseline_values)
            dz, dz_note = cohens_dz(differences)

            full_mean, full_std = model_stats[(FULL_MODEL_NAME, metric)]
            baseline_mean, baseline_std = model_stats[(baseline_name, metric)]

            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "full_model": FULL_MODEL_NAME,
                    "baseline_model": baseline_name,
                    "n_seeds": len(differences),
                    "full_mean": full_mean,
                    "full_std": full_std,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wilcoxon_statistic": wilcoxon_stat,
                    "raw_p": raw_p,
                    "paired_t_statistic": t_stat,
                    "paired_t_p": t_p,
                    "cohens_dz": dz,
                    "difference_std": sample_std(differences),
                    "wilcoxon_note": wilcoxon_note,
                    "paired_ttest_note": ttest_note,
                    "cohens_dz_note": dz_note,
                    "holm_adjusted_p": float("nan"),
                    "significant_raw_0.05": bool(np.isfinite(raw_p) and raw_p < ALPHA),
                    "significant_holm_0.05": False,
                }
            )

            paired_tables[(comparison, metric)] = pd.DataFrame(
                {
                    "seed": full_frame["seed"].tolist(),
                    FULL_MODEL_NAME: full_values,
                    baseline_name: baseline_values,
                    "difference": differences,
                }
            )

    return rows, paired_tables


def apply_holm_correction(rows: List[Dict[str, object]]) -> None:
    for metric in TEST_METRICS:
        metric_indices = [idx for idx, row in enumerate(rows) if row["metric"] == metric]
        raw_pvals = [float(rows[idx]["raw_p"]) for idx in metric_indices]
        rejected, adjusted = holm_bonferroni(raw_pvals, alpha=ALPHA)
        for local_idx, row_idx in enumerate(metric_indices):
            rows[row_idx]["holm_adjusted_p"] = float(adjusted[local_idx])
            rows[row_idx]["significant_holm_0.05"] = bool(rejected[local_idx])


def write_summary_txt(
    output_path: str,
    expected_seeds: List[int],
    model_stats: Dict[Tuple[str, str], Tuple[float, float]],
    model_order: List[str],
    rows: List[Dict[str, object]],
    paired_tables: Dict[Tuple[str, str], pd.DataFrame],
) -> None:
    lines: List[str] = []
    lines.append("Paired significance test for TTC-GNN temporal 8/1/1 experiments")
    lines.append("Split/seeds assumption: same per-bot chronological 8/1/1 split + same random seeds")
    lines.append("Expected paired seeds: {}".format(",".join(map(str, expected_seeds))))
    lines.append("Main test: Wilcoxon signed-rank test, two-sided, paired by seed")
    lines.append("Supplement: paired t-test; independent t-test is not used")
    if len(expected_seeds) == 1:
        lines.append(
            "Note: only one paired seed is included. Treat this as a paired-value "
            "sanity check before the full five-seed significance test."
        )
    lines.append("")

    lines.append("Model descriptive statistics")
    for metric in TEST_METRICS:
        lines.append("{}:".format(metric))
        for model_name in model_order:
            mean, std = model_stats[(model_name, metric)]
            lines.append("  {} = {}".format(model_name, format_mean_std(mean, std)))
        lines.append("")

    lines.append("Macro-F1 paper-style comparison summary")
    macro_rows = [row for row in rows if row["metric"] == "test_macro_f1"]
    for row in macro_rows:
        baseline_name = row["baseline_model"]
        full_mean, full_std = model_stats[(FULL_MODEL_NAME, "test_macro_f1")]
        baseline_mean, baseline_std = model_stats[(baseline_name, "test_macro_f1")]
        lines.append(row["comparison"])
        lines.append("Macro-F1:")
        lines.append("{} = {}".format(FULL_MODEL_NAME, format_mean_std(full_mean, full_std)))
        lines.append("{} = {}".format(baseline_name, format_mean_std(baseline_mean, baseline_std)))
        lines.append("Mean difference = {:+.4f}".format(float(row["mean_difference"])))
        lines.append(
            "Wins/Ties/Losses = {}/{}/{}".format(
                row["wins"], row["ties"], row["losses"]
            )
        )
        lines.append("Wilcoxon W = {}".format(format_float(row["wilcoxon_statistic"])))
        lines.append("Raw p = {}".format(format_float(row["raw_p"])))
        lines.append("Holm-adjusted p = {}".format(format_float(row["holm_adjusted_p"])))
        lines.append("Paired t statistic = {}".format(format_float(row["paired_t_statistic"])))
        lines.append("Paired t-test p = {}".format(format_float(row["paired_t_p"])))
        lines.append("Cohen's dz = {}".format(format_float(row["cohens_dz"])))
        lines.append(
            "Significant after Holm correction: {}".format(
                "YES" if row["significant_holm_0.05"] else "NO"
            )
        )
        notes = [
            str(row[key])
            for key in ("wilcoxon_note", "paired_ttest_note", "cohens_dz_note")
            if row.get(key)
        ]
        if notes:
            lines.append("Notes: {}".format("; ".join(notes)))
        lines.append("")

    lines.append("Original paired values used for each test")
    for row in rows:
        key = (row["comparison"], row["metric"])
        paired = paired_tables[key]
        lines.append("{} | {}".format(row["comparison"], row["metric"]))
        lines.append(paired.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def main():
    args = parse_args()
    expected_seeds = parse_seed_list(args.expected_seeds)
    baselines = collect_baselines(args)

    os.makedirs(args.output_dir, exist_ok=True)

    full_frame = read_results_csv(
        FULL_MODEL_NAME,
        args.full,
        expected_seeds,
        strict_seed_match=args.strict_seed_match,
    )
    baseline_frames = {
        name: read_results_csv(
            name,
            path,
            expected_seeds,
            strict_seed_match=args.strict_seed_match,
        )
        for name, path in baselines
    }

    model_frames = {FULL_MODEL_NAME: full_frame}
    model_frames.update(baseline_frames)
    model_order = [FULL_MODEL_NAME] + [name for name, _ in baselines]
    model_stats = descriptive_stats(model_frames)

    rows, paired_tables = build_comparison_rows(
        full_frame=full_frame,
        baseline_frames=baseline_frames,
        model_stats=model_stats,
    )
    apply_holm_correction(rows)

    all_results = pd.DataFrame(rows)
    all_results_path = os.path.join(args.output_dir, "all_significance_results.csv")
    all_results.to_csv(all_results_path, index=False, encoding="utf-8-sig")

    macro_columns = [
        "comparison",
        "full_mean",
        "baseline_mean",
        "mean_difference",
        "wins",
        "ties",
        "losses",
        "wilcoxon_statistic",
        "raw_p",
        "holm_adjusted_p",
        "cohens_dz",
        "significant_holm_0.05",
    ]
    macro_results = all_results.loc[
        all_results["metric"] == "test_macro_f1", macro_columns
    ].copy()
    macro_results_path = os.path.join(args.output_dir, "macro_f1_significance.csv")
    macro_results.to_csv(macro_results_path, index=False, encoding="utf-8-sig")

    summary_path = os.path.join(args.output_dir, "significance_summary.txt")
    write_summary_txt(
        output_path=summary_path,
        expected_seeds=expected_seeds,
        model_stats=model_stats,
        model_order=model_order,
        rows=rows,
        paired_tables=paired_tables,
    )

    print("[DONE] Significance results saved in {}".format(os.path.abspath(args.output_dir)))
    print("  - {}".format(all_results_path))
    print("  - {}".format(macro_results_path))
    print("  - {}".format(summary_path))


if __name__ == "__main__":
    main()
