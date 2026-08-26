import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ccdf_plot(y, out_png):
    y = np.asarray(y)
    y = y[y > 0]
    y = np.sort(y)
    ccdf = 1.0 - np.arange(1, len(y) + 1) / len(y)

    plt.figure(figsize=(6, 4))
    plt.loglog(y, ccdf)
    plt.xlabel("y (log)")
    plt.ylabel("CCDF P(Y>=y) (log)")
    plt.title("Empirical CCDF (log-log)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def hist_log_plot(y, out_png):
    y = np.asarray(y)
    y = y[y > 0]
    if len(y) == 0:
        return
    # log-spaced bins
    bins = np.logspace(np.log10(y.min()), np.log10(y.max()), 40)

    plt.figure(figsize=(6, 4))
    plt.hist(y, bins=bins)
    plt.xscale("log")
    plt.xlabel("y (log)")
    plt.ylabel("count")
    plt.title("Histogram (log x)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def fit_powerlaw_ks(y_tail, xmin):
    """
    Continuous power-law fit for y >= xmin
    Returns (alpha, ks)
    """
    y_tail = np.asarray(y_tail)
    y_tail = y_tail[y_tail >= xmin]
    n = len(y_tail)
    if n < 2:
        return None, np.inf

    # MLE for continuous power-law exponent
    alpha = 1.0 + n / np.sum(np.log(y_tail / xmin))

    # Empirical CDF on tail
    y_sorted = np.sort(y_tail)
    ecdf = np.arange(1, n + 1) / n

    # Model CDF: F(x) = 1 - (x/xmin)^(1-alpha)  for x>=xmin
    model_cdf = 1.0 - (y_sorted / xmin) ** (1.0 - alpha)

    ks = np.max(np.abs(ecdf - model_cdf))
    return alpha, ks

def find_xmin_by_ks(y, min_tail=100, candidate_points=60):
    """
    Search xmin among candidate quantiles; pick xmin that minimizes KS distance.
    """
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    if len(y) < min_tail:
        return None, None, None

    y_sorted = np.sort(y)
    # choose candidates from mid-high quantiles to avoid tiny xmin
    qs = np.linspace(0.50, 0.95, candidate_points)
    candidates = np.unique(np.quantile(y_sorted, qs))

    best = {"xmin": None, "alpha": None, "ks": np.inf, "n_tail": 0}
    for xmin in candidates:
        y_tail = y[y >= xmin]
        if len(y_tail) < min_tail:
            continue
        alpha, ks = fit_powerlaw_ks(y_tail, xmin)
        if ks < best["ks"]:
            best.update({"xmin": float(xmin), "alpha": float(alpha), "ks": float(ks), "n_tail": int(len(y_tail))})

    if best["xmin"] is None:
        return None, None, None
    return best["xmin"], best["alpha"], best

def head_coverage_cutoff(y, coverage=0.80):
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = y[y >= 0]
    y_sorted = np.sort(y)[::-1]
    s = y_sorted.sum()
    if s <= 0:
        return 0, 0.0
    cum = np.cumsum(y_sorted) / s
    k = int(np.searchsorted(cum, coverage) + 1)
    kth_value = float(y_sorted[k - 1])
    return k, kth_value

def make_labels(y, t_high, t_mid=None, separate_zero=True):
    """
    Default 3 classes (or 4 with zero):
      if separate_zero:
        0: y==0
        1: (0, t_mid]
        2: (t_mid, t_high]
        3: > t_high
      else:
        0: <= t_mid
        1: (t_mid, t_high]
        2: > t_high
    """
    y = np.asarray(y)
    labels = np.zeros(len(y), dtype=int)

    if separate_zero:
        # start from 0-class already
        nonzero = y > 0
        if t_mid is None:
            # split below high into two roughly balanced parts
            below_high = (y > 0) & (y <= t_high)
            t_mid = float(np.median(y[below_high])) if below_high.any() else float(t_high / 2)

        labels[(y > 0) & (y <= t_mid)] = 1
        labels[(y > t_mid) & (y <= t_high)] = 2
        labels[y > t_high] = 3
        return labels, t_mid
    else:
        if t_mid is None:
            t_mid = float(np.median(y[y <= t_high])) if (y <= t_high).any() else float(t_high / 2)
        labels[y <= t_mid] = 0
        labels[(y > t_mid) & (y <= t_high)] = 1
        labels[y > t_high] = 2
        return labels, t_mid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ycol", required=True, help="continuous engagement label column name")
    ap.add_argument("--out_csv", default="with_engagement_class.csv")
    ap.add_argument("--min_tail", type=int, default=100, help="min tail points for KS search")
    ap.add_argument("--coverage", type=float, default=0.80, help="head coverage for method A")
    ap.add_argument("--use_method", choices=["B", "A"], default="B", help="B=KS xmin, A=head-coverage")
    ap.add_argument("--separate_zero", action="store_true", help="add an explicit zero class")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.ycol not in df.columns:
        raise ValueError(f"Column {args.ycol} not found. Columns: {list(df.columns)[:30]} ...")

    y = pd.to_numeric(df[args.ycol], errors="coerce").fillna(0).values
    if (y < 0).any():
        raise ValueError("Found negative y values. Engagement should be >=0; please clean or shift first.")

    # plots
    hist_log_plot(y, "y_hist_log.png")
    ccdf_plot(y, "y_ccdf_loglog.png")

    # decide high threshold
    if args.use_method == "B":
        xmin, alpha, info = find_xmin_by_ks(y, min_tail=args.min_tail)
        if xmin is None:
            # fallback to A if B fails
            k, kth = head_coverage_cutoff(y, coverage=args.coverage)
            t_high = kth
            method_used = f"A(fallback): head {args.coverage:.0%} cutoff"
            extra = {"head_k": k}
        else:
            t_high = xmin
            method_used = f"B: KS-min xmin (power-law tail)"
            extra = {"alpha": alpha, **info}
    else:
        k, kth = head_coverage_cutoff(y, coverage=args.coverage)
        t_high = kth
        method_used = f"A: head {args.coverage:.0%} cutoff"
        extra = {"head_k": k}

    labels, t_mid = make_labels(y, t_high=t_high, t_mid=None, separate_zero=args.separate_zero)
    df["engagement_class"] = labels

    # report
    n = len(y)
    zeros = int((y == 0).sum())
    tail_n = int((y > t_high).sum())
    print("========== Long-tail threshold report ==========")
    print(f"n={n}, zeros={zeros} ({zeros/n:.2%})")
    print(f"method_used: {method_used}")
    print(f"t_high (tail starts) = {t_high}")
    print(f"t_mid  (low/mid split)= {t_mid}")
    print(f"tail_n (y>t_high)     = {tail_n} ({tail_n/n:.2%})")
    for k, v in extra.items():
        print(f"{k}: {v}")

    print("\nClass counts:")
    vc = pd.Series(labels).value_counts().sort_index()
    print(vc.to_string())

    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved labeled CSV: {args.out_csv}")
    print("Saved plots: y_hist_log.png, y_ccdf_loglog.png")

if __name__ == "__main__":
    main()
