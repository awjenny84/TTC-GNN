import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GRAPH_HIDDENS = [32,64, 128, 256]   # 你已有的档位；想画 50/100/150/200 就改这里
SEEDS = [0, 1, 2]
MAX_LEN = 128
LOG_DIR = Path("logs")

# 解析形如：
# [Epoch 2/10] ... val_F1=0.7517 (macro)
def build_pattern(metric_name):
    return re.compile(
        rf"\[Epoch\s+(\d+)/(\d+)\].*?{metric_name}=([0-9]*\.?[0-9]+)",
        re.IGNORECASE
    )


def parse_curve(log_path: Path, metric):
    PAT = build_pattern(metric)
    epochs, vals = [], []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                ep = int(m.group(1))
                val = float(m.group(3))
                epochs.append(ep)
                vals.append(val)

    if not epochs:
        raise ValueError(f"No {metric} found in {log_path}")

    return np.array(epochs), np.array(vals)


def align(curves):
    """curves: list of (epochs, f1s). Return epoch_grid, matrix [num_runs, max_ep] with NaN padding."""
    max_ep = max(int(eps.max()) for eps, _ in curves)
    grid = np.arange(1, max_ep + 1)
    mat = np.full((len(curves), max_ep), np.nan, dtype=float)
    for i, (eps, f1s) in enumerate(curves):
        for e, v in zip(eps, f1s):
            if 1 <= e <= max_ep:
                mat[i, e - 1] = v
    return grid, mat

def main():
    METRICS = {
        "val_acc": "Accuracy",
        "val_P": "Precision",
        "val_R": "Recall",
        "val_F1": "Macro-F1"
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (metric_key, metric_name) in enumerate(METRICS.items()):
        ax = axes[idx]

        for gh in GRAPH_HIDDENS:
            curves = []
            for s in SEEDS:
                name = f"maxlen_{MAX_LEN}_gh_{gh}_seed_{s}.log"
                p = LOG_DIR / name
                if not p.exists():
                    raise FileNotFoundError(f"Missing log: {p}")
                curves.append(parse_curve(p, metric_key))

            grid, mat = align(curves)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)

            ax.plot(grid, mean, label=f"gh={gh}")
            ax.fill_between(grid, mean - std, mean + std, alpha=0.15)

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Validation {metric_name}")
        ax.grid(True, linewidth=0.5)

    # 统一图例（只放一次）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(GRAPH_HIDDENS))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig_path = out_dir / f"val_metrics_2x2_maxlen{MAX_LEN}_gh_compare.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()
