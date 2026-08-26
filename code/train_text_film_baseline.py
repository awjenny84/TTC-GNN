# -*- coding: utf-8 -*-
"""
Text-only multiclass engagement prediction with FiLM conditioning.

Compared with e2e graph-text version:
- Keep role/style-conditioned FiLM text encoder.
- Remove graph encoder and graph fusion.
- Keep multi-seed training/evaluation summary (mean ± std).

Examples:
  python train_text_film_multiclass_5seeds_with_train_metrics.py --run_multi_seed
  python train_text_film_multiclass_5seeds_with_train_metrics.py --run_multi_seed --seeds 42,43,44,45,46
  python train_text_film_multiclass_5seeds_with_train_metrics.py --seed 42
  python train_text_film_multiclass_5seeds_with_train_metrics.py --run_multi_seed --use_lora --lora_target_modules query,value
"""

import os
import json
import csv
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt


# -------------------------
# 0) Utils
# -------------------------
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_seeds(seed_str: str) -> List[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def maybe_apply_lora(
    model: nn.Module,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
    lora_bias: str,
) -> nn.Module:
    if not use_lora:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "use_lora=True 但当前环境无法导入 peft。请先安装 peft，例如: pip install peft"
        ) from e

    target_modules = lora_target_modules or ["query", "value"]
    if not target_modules:
        raise ValueError("LoRA target_modules 不能为空。")

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, lora_cfg)


def compute_mean_std(run_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = run_metrics[0].keys()
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in run_metrics], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"acc": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


# -------------------------
# 1) Dataset
# -------------------------
class WeiboTensorDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        role_ids: torch.Tensor,
        style_feats: torch.Tensor,
        labels: torch.Tensor,
        tokenizer,
        max_len: int = 128,
    ):
        self.texts = texts
        self.role_ids = role_ids
        self.style_feats = style_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tok = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "role_id": self.role_ids[idx],
            "style_vec": self.style_feats[idx],
            "label": self.labels[idx],
        }


# -------------------------
# 2) FiLM text model
# -------------------------
class RoleConditionedTextEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        cond_dim: int = 64,
        style_proj_dim: int = 64,
        film_hidden: int = 256,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_bias: str = "none",
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        self.bert = maybe_apply_lora(
            self.bert,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )
        hidden = self.bert.config.hidden_size

        self.role_emb = nn.Embedding(num_roles, cond_dim)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, style_proj_dim),
            nn.ReLU(),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim + style_proj_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, hidden * 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, role_id, style_vec):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state

        e_role = self.role_emb(role_id)
        z_style = self.style_mlp(style_vec)
        cond = torch.cat([e_role, z_style], dim=-1)

        gamma_beta = self.film_gen(cond)
        hdim = h.size(-1)
        gamma, beta = gamma_beta[:, :hdim], gamma_beta[:, hdim:]
        h_tilde = gamma.unsqueeze(1) * h + beta.unsqueeze(1)
        return self.dropout(h_tilde)


class FilmTextClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        num_classes: int,
        graph_hidden_unused: int = 64,
        dropout: float = 0.1,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_bias: str = "none",
    ):
        super().__init__()
        self.text_enc = RoleConditionedTextEncoder(
            backbone=backbone,
            num_roles=num_roles,
            style_in_dim=style_in_dim,
            dropout=dropout,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_bias=lora_bias,
        )
        hidden = self.text_enc.bert.config.hidden_size
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, role_id, style_vec):
        h_tilde = self.text_enc(input_ids, attention_mask, role_id, style_vec)
        text_cls = h_tilde[:, 0, :]
        logits = self.cls_head(text_cls)
        return logits


# -------------------------
# 3) Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(
    model: FilmTextClassifier,
    dl: DataLoader,
    device: str,
    criterion: nn.Module,
    average: str = "macro",
    return_preds: bool = False,
):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        role_id = batch["role_id"].to(device)
        style_vec = batch["style_vec"].to(device)
        y = batch["label"].to(device)

        logits = model(input_ids, attention_mask, role_id, style_vec)
        loss = criterion(logits, y)

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        n += bs

        y_pred = torch.argmax(logits, dim=-1)
        y_true_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(y_pred.detach().cpu().tolist())

    avg_loss = total_loss / max(1, n)
    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)
    metrics = compute_metrics(y_true_np, y_pred_np, average=average)

    if return_preds:
        return avg_loss, metrics, y_true_np, y_pred_np
    return avg_loss, metrics


# -------------------------
# 4) Data prep (fixed split)
# -------------------------
def prepare_data(args) -> Dict[str, Any]:
    df = pd.read_csv(args.csv_path)
    if args.text_col not in df.columns:
        raise ValueError(f"CSV missing text col: {args.text_col}")
    if args.role_col not in df.columns:
        raise ValueError(f"CSV missing role col: {args.role_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"CSV missing label col: {args.label_col}")

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    role_ids_all = torch.tensor(df[args.role_col].astype(int).values, dtype=torch.long)
    num_roles = int(role_ids_all.max().item() + 1)

    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for c in feat_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing style feature col: {c}")
    feats = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True))

    le = LabelEncoder()
    y_all_np = le.fit_transform(df[args.label_col].astype(str).fillna("NA").values)
    y_all = torch.tensor(y_all_np, dtype=torch.long)
    num_classes = int(len(le.classes_))
    print(f"[INFO] num_classes={num_classes}, classes={list(le.classes_)}")

    idx = np.arange(len(df))
    idx_train, idx_tmp, y_train_np, y_tmp_np = train_test_split(
        idx,
        y_all_np,
        test_size=0.2,
        random_state=args.split_seed,
        stratify=y_all_np if num_classes > 1 else None,
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp,
        y_tmp_np,
        test_size=0.5,
        random_state=args.split_seed,
        stratify=y_tmp_np if num_classes > 1 else None,
    )
    print(
        f"[INFO] Split fixed by split_seed={args.split_seed}: "
        f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
    )

    scaler = StandardScaler()
    scaler.fit(feats.iloc[idx_train].values)
    style_all = scaler.transform(feats.values)
    style_all_t = torch.tensor(style_all, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def make_ds(idxs: np.ndarray) -> WeiboTensorDataset:
        idxs_t = torch.as_tensor(idxs, dtype=torch.long)
        return WeiboTensorDataset(
            texts=df.iloc[idxs][args.text_col].tolist(),
            role_ids=role_ids_all[idxs_t],
            style_feats=style_all_t[idxs_t],
            labels=y_all[idxs_t],
            tokenizer=tokenizer,
            max_len=args.max_len,
        )

    return {
        "df": df,
        "feat_cols": feat_cols,
        "le": le,
        "scaler": scaler,
        "num_roles": num_roles,
        "num_classes": num_classes,
        "datasets": {"train": make_ds(idx_train), "val": make_ds(idx_val), "test": make_ds(idx_test)},
    }


def make_dataloaders(args, datasets, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    dl_train = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dl_val = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    return dl_train, dl_val, dl_test


def run_one_seed(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n{'='*20} Running seed={seed} {'='*20}")
    set_seed(seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_train, dl_val, dl_test = make_dataloaders(args, bundle["datasets"], seed)

    lora_target_modules = parse_csv_list(args.lora_target_modules)
    model = FilmTextClassifier(
        backbone=args.backbone,
        num_roles=bundle["num_roles"],
        style_in_dim=len(bundle["feat_cols"]),
        num_classes=bundle["num_classes"],
        dropout=0.1,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_bias=args.lora_bias,
    ).to(device)

    total_params, trainable_params_n = count_parameters(model)
    print(
        f"[INFO] model params: trainable={trainable_params_n:,} / total={total_params:,} "
        f"({100.0 * trainable_params_n / max(1, total_params):.2f}%)"
    )
    if args.use_lora and hasattr(model.text_enc.bert, "print_trainable_parameters"):
        model.text_enc.bert.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有可训练参数，请检查 LoRA 配置。")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    no_improve = 0
    best_epoch = -1
    ckpt_path = os.path.join("ckpt", f"text_film_best_seed{seed}_split{args.split_seed}.pt")

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in dl_train:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            role_id = batch["role_id"].to(device)
            style_vec = batch["style_vec"].to(device)
            y = batch["label"].to(device)

            logits = model(input_ids, attention_mask, role_id, style_vec)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n += bs

        train_loss = total_loss / max(1, n)
        train_eval_loss, train_metrics = evaluate(
            model, dl_train, device, criterion, average=args.metric_average, return_preds=False
        )
        val_loss, val_metrics = evaluate(
            model, dl_val, device, criterion, average=args.metric_average, return_preds=False
        )

        print(
            f"[Seed {seed} | Epoch {ep+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} | train_eval_loss={train_eval_loss:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_P={train_metrics['precision']:.4f} "
            f"train_R={train_metrics['recall']:.4f} "
            f"train_F1={train_metrics['f1']:.4f} ({args.metric_average}) | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_P={val_metrics['precision']:.4f} "
            f"val_R={val_metrics['recall']:.4f} "
            f"val_F1={val_metrics['f1']:.4f} ({args.metric_average})"
        )

        improved = val_metrics["f1"] > (best_val_f1 + args.early_min_delta)
        if improved:
            best_val_f1 = val_metrics["f1"]
            best_epoch = ep + 1
            no_improve = 0
            os.makedirs("ckpt", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": args.backbone,
                    "use_lora": args.use_lora,
                    "lora_config": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "target_modules": lora_target_modules,
                        "bias": args.lora_bias,
                    },
                    "num_roles": bundle["num_roles"],
                    "num_classes": bundle["num_classes"],
                    "label_classes": bundle["le"].classes_.tolist(),
                    "feature_cols": bundle["feat_cols"],
                    "role_col": args.role_col,
                    "seed": seed,
                    "split_seed": args.split_seed,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            print(
                f"[EarlyStop] no_improve={no_improve}/{args.early_patience} "
                f"(best_f1={best_val_f1:.4f} @ epoch {best_epoch})"
            )
            if no_improve >= args.early_patience:
                print(f"[EarlyStop] Stop at epoch {ep + 1}. Best epoch={best_epoch}, best_val_f1={best_val_f1:.4f}")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    train_eval_loss, train_metrics = evaluate(
        model, dl_train, device, criterion, average=args.metric_average, return_preds=False
    )
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, dl_test, device, criterion, average=args.metric_average, return_preds=True
    )

    print(
        f"[TRAIN seed={seed}] loss={train_eval_loss:.4f} | "
        f"acc={train_metrics['acc']:.4f} P={train_metrics['precision']:.4f} "
        f"R={train_metrics['recall']:.4f} F1={train_metrics['f1']:.4f}"
    )
    print(
        f"[TEST seed={seed}] loss={test_loss:.4f} | "
        f"acc={test_metrics['acc']:.4f} P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f}"
    )

    if args.save_each_seed_cm:
        labels = np.arange(bundle["num_classes"])
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        np.savetxt(f"cm_text_film_test_counts_seed{seed}.csv", cm, fmt="%d", delimiter=",")
        np.savetxt(f"cm_text_film_test_norm_true_seed{seed}.csv", cm_norm, fmt="%.6f", delimiter=",")

        class_names = [str(i) for i in labels]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(values_format="d", cmap=plt.cm.Blues)
        for text in disp.text_.ravel():
            text.set_color("black")
            text.set_fontsize(11)
        plt.tight_layout()
        plt.savefig(f"cm_text_film_test_seed{seed}.png", dpi=300)
        plt.close()

        print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4))

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "train_loss": float(train_eval_loss),
        "train_acc": float(train_metrics["acc"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_f1": float(train_metrics["f1"]),
        "test_loss": float(test_loss),
        "test_acc": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
    }


def save_summary(results: List[Dict[str, Any]], out_csv: str):
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="with_engagement_class.csv")
    ap.add_argument("--label_col", type=str, default="engagement_class")
    ap.add_argument("--text_col", type=str, default="text_raw")
    ap.add_argument("--role_col", type=str, default="role_id")
    ap.add_argument(
        "--feature_cols",
        type=str,
        default="length,emoji_count,is_qa,sentiment,ttr,rttr,mtld,msttr,common_ratio,stop_ratio",
    )

    ap.add_argument("--backbone", type=str, default="chinese-roberta-wwm-ext")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--use_lora", action="store_true", help="enable LoRA finetuning on text backbone")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="query,value",
        help="comma-separated target modules, e.g. query,value or q_proj,v_proj",
    )
    ap.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="42,52,62,72,82", help="comma-separated seeds for multi-seed runs")
    ap.add_argument("--run_multi_seed", action="store_true", help="whether to run all seeds in --seeds")
    ap.add_argument("--split_seed", type=int, default=42, help="random seed for fixed data split")
    ap.add_argument("--metric_average", type=str, default="macro")
    ap.add_argument("--early_patience", type=int, default=99)
    ap.add_argument("--early_min_delta", type=float, default=1e-4)
    ap.add_argument("--save_each_seed_cm", action="store_true")
    args = ap.parse_args()
    if args.use_lora and not parse_csv_list(args.lora_target_modules):
        raise ValueError("启用 LoRA 时，lora_target_modules 不能为空。")

    os.makedirs("ckpt", exist_ok=True)
    bundle = prepare_data(args)

    run_seeds = parse_seeds(args.seeds) if args.run_multi_seed else [args.seed]
    print(f"[INFO] run_seeds={run_seeds} | split_seed fixed at {args.split_seed}")

    all_results = []
    for seed in run_seeds:
        result = run_one_seed(args, seed, bundle)
        all_results.append(result)

    save_summary(all_results, "multi_seed_text_film_results.csv")
    summary = compute_mean_std([{k: v for k, v in r.items() if k not in ["seed", "best_epoch"]} for r in all_results])

    print("\n" + "=" * 60)
    print("Multi-seed summary")
    for metric in ["best_val_f1", "train_loss", "test_loss"]:
        mean, std = summary[metric]
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("Train metrics mean ± std")
    for metric in ["train_acc", "train_precision", "train_recall", "train_f1"]:
        mean, std = summary[metric]
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("Test metrics mean ± std")
    for metric in ["test_acc", "test_precision", "test_recall", "test_f1"]:
        mean, std = summary[metric]
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
    print("=" * 60)

    with open("multi_seed_text_film_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_seeds": run_seeds,
                "split_seed": args.split_seed,
                "summary": {k: {"mean": v[0], "std": v[1]} for k, v in summary.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open("ckpt/text_film_style_scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": bundle["scaler"].mean_.tolist(), "scale": bundle["scaler"].scale_.tolist()}, f, ensure_ascii=False)
    with open("ckpt/text_film_label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump({"label_col": args.label_col, "classes": bundle["le"].classes_.tolist()}, f, ensure_ascii=False)

    print(
        "[DONE] Saved: multi_seed_text_film_results.csv, multi_seed_text_film_summary.json, "
        "ckpt/text_film_style_scaler.json, ckpt/text_film_label_encoder_classes.json"
    )


if __name__ == "__main__":
    main()
