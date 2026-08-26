# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 后端
import matplotlib.pyplot as plt

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

try:
    import shap
except Exception as e:
    raise SystemExit("请先安装 shap：pip install -U shap\n" + repr(e))

SEED = 42


def parse_feature_cols(arg: str, all_cols: List[str], target: str) -> List[str]:
    if not arg:
        return [c for c in all_cols if c != target]
    try:
        arr = json.loads(arg)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    return [c.strip() for c in arg.split(",") if c.strip()]


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 读数据与列检查
    df = pd.read_csv(args.csv_path)
    if args.target_col not in df.columns:
        raise ValueError(f"CSV 不存在目标列: {args.target_col}")
    features = parse_feature_cols(args.feature_cols, list(df.columns), args.target_col)
    for c in features:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少特征列: {c}")

    # X 数值化
    Xdf = df[features].copy()
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.values.astype(float)

    # y 与任务类型
    y_raw = df[args.target_col]
    task = args.task
    if task == "auto":
        # 数值且唯一值多 -> 回归；否则分类
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 20:
            task = "reg"
        else:
            task = "clf"

    if task == "clf":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str).fillna("__NA__"))
        model = HistGradientBoostingClassifier(random_state=SEED)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        y = y.fillna(np.nanmedian(y))
        model = HistGradientBoostingRegressor(random_state=SEED)

    # 切分&训练
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=SEED)
    model.fit(Xtr, ytr)

    # SHAP（树路径，快速）
    explainer = shap.TreeExplainer(model, Xtr)
    exp = explainer(Xte, check_additivity=False)

    # 兼容多分类：取“最后/正类”，得到 (n_samples, n_features)
    sv = exp.values
    if isinstance(sv, list):
        sv_use = sv[-1]
    elif sv.ndim == 3:
        sv_use = sv[:, :, -1]
    else:
        sv_use = sv

    # 特征重要性 = mean(|SHAP|)
    mean_abs = np.mean(np.abs(sv_use), axis=0)
    fi = pd.DataFrame({"feature": features, "importance": mean_abs}).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False)
    feature2cat = {
        # Structural
        "length": "Structural",
        "is_qa": "Structural",

        # Lexical (lexical diversity / vocabulary usage)
        "ttr": "Lexical",
        "rttr": "Lexical",
        "msttr": "Lexical",
        "mtld": "Lexical",
        "common_ratio": "Lexical",

        # Syntactic / function-word related
        "stop_ratio": "Syntactic",

        # Emotional / affective cues
        "sentiment": "Emotional",
        "emoji_count": "Emotional",
    }

    # If some features are not mapped, put them into "Other" (or raise error)
    cats = []
    for f in features:
        cats.append(feature2cat.get(f, "Other"))

    fi_with_cat = fi.copy()
    fi_with_cat["category"] = [feature2cat.get(f, "Other") for f in fi_with_cat["feature"]]

    # 2) Aggregate: mean and sum of mean(|SHAP|) within each category
    cat_agg = (
        fi_with_cat.groupby("category")["importance"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    cat_csv = os.path.join(args.outdir, "category_importance.csv")
    cat_agg.to_csv(cat_csv, index=False)

    # 3) Plot (barh): category-level mean(|SHAP|)
    cat_fig = os.path.join(args.outdir, "category_importance.png")
    plt.figure(figsize=(6.5, 3.2))
    y = np.arange(len(cat_agg))
    plt.barh(y, cat_agg["mean"].values)
    plt.yticks(y, cat_agg["category"].values)
    plt.gca().invert_yaxis()


    max_val = cat_agg["mean"].max()
    plt.xlim(0, max_val * 1.15)

    for i, v in enumerate(cat_agg["mean"].values):
        plt.text(v, i, f"  {v:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(cat_fig, dpi=300)
    plt.close()

    print("Saved:", cat_csv)
    print("Saved:", cat_fig)

    # 全局蚁群图（优先新API的 plots.beeswarm，失败回退 summary_plot）
    beeswarm_path = os.path.join(args.outdir, "beeswarm.png")
    plt.figure()
    try:
        # 构造一个二维 Explanation 以兼容新接口
        base = exp.base_values
        if isinstance(base, list):
            base = base[-1]
        if np.ndim(base) == 2:
            base = base[:, -1]
        exp2d = shap.Explanation(
            values=sv_use,
            base_values=base,
            data=Xte,
            feature_names=features
        )
        shap.plots.beeswarm(exp2d, max_display=args.max_display, show=False)
    except Exception:
        shap.summary_plot(sv_use, Xte, feature_names=features, show=False, max_display=args.max_display)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()

    print("Saved:", os.path.join(args.outdir, "feature_importance.csv"))
    print("Saved:", beeswarm_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, required=True)
    ap.add_argument("--target-col", type=str, required=True, help="目标列名（回归或分类标签）")
    ap.add_argument("--feature-cols", type=str, default="", help="特征列（逗号分隔或JSON数组）。缺省=除目标列外全部列")
    ap.add_argument("--task", type=str, default="auto", choices=["auto", "reg", "clf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-display", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="./shap_out")
    args = ap.parse_args()
    main(args)
