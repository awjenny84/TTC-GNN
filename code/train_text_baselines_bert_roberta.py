# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DEFAULT_SEEDS = "42,52,62,72,82"
DEFAULT_BACKBONE = "chinese-roberta-wwm-ext"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def resolve_local_path(path: str) -> str:
    if os.path.isdir(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    return path


def read_csv_with_fallback(path: str, encodings) -> pd.DataFrame:
    last_error = None
    for encoding in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"uid": str, "wid": str})
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        "Could not decode {} with encodings {}; last={}".format(
            path, list(dict.fromkeys(encodings)), last_error
        ),
    )


class WeiboTextClsDataset(Dataset):
    """Text-only post dataset for BERT engagement classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 128,
        label_encoder: Optional[LabelEncoder] = None,
        fit_label_encoder: bool = False,
        text_col: str = "text_raw",
        label_col: str = "engagement_class",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

        for column in (text_col, label_col):
            if column not in self.df.columns:
                raise ValueError("CSV is missing required column '{}'.".format(column))

        self.df[text_col] = self.df[text_col].fillna("").astype(str)

        if label_encoder is None:
            label_encoder = LabelEncoder()

        raw_y = self.df[label_col].astype(str).values
        if fit_label_encoder:
            label_encoder.fit(raw_y)

        self.le = label_encoder
        self.labels = torch.tensor(self.le.transform(raw_y), dtype=torch.long)
        self.texts = self.df[text_col].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
            "labels": self.labels[idx],
        }


class BertTextClassifier(nn.Module):
    """BERT [CLS] representation followed by a linear classifier."""

    def __init__(self, backbone: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def split_per_bot_chronological_8_1_1(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split posts within every bot by chronological order.

    This follows train_multiclass_engagement_e2e_5seeds_per_bot_811.py:
    earliest 80% train, next 10% validation, latest remainder test. Bot
    identities are shared across splits by design; individual posts are
    mutually disjoint.
    """
    required = {"uid", "wid", text_col, label_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(sorted(missing)))

    source = df.copy()
    source["uid"] = normalize_id(source["uid"])
    source["wid"] = normalize_id(source["wid"])
    source = source[(source["uid"] != "") & (source["wid"] != "")].copy()
    source[text_col] = source[text_col].fillna("").astype(str)

    source["__time"] = pd.to_datetime(source[time_col], errors="coerce")
    bad_time = source["__time"].isna()
    if bad_time.any():
        examples = source.loc[bad_time, ["uid", "wid", time_col]].head().to_dict("records")
        raise ValueError(
            "Invalid or missing timestamps in '{}'; examples: {}".format(
                time_col, examples
            )
        )

    duplicate_mask = source.duplicated(["uid", "wid", label_col], keep="first")
    if duplicate_mask.any():
        print(
            "[WARN] Dropping {} duplicate (uid, wid, label) rows before splitting.".format(
                int(duplicate_mask.sum())
            )
        )
        source = source.loc[~duplicate_mask].copy()

    conflicting = source.duplicated(["uid", "wid"], keep=False)
    if conflicting.any():
        examples = source.loc[conflicting, ["uid", "wid", label_col]].head().to_dict("records")
        raise ValueError("Duplicate (uid, wid) rows with conflicting labels: {}".format(examples))

    parts = []
    bot_stats = []
    for uid, group in source.groupby("uid", sort=True):
        group = (
            group.copy()
            .sort_values(["__time", "wid"], ascending=[True, True], kind="mergesort")
            .reset_index(drop=True)
        )
        n = len(group)
        if n < 3:
            raise ValueError(
                "Bot {} has only {} posts; at least 3 are required for an 8/1/1 split.".format(
                    uid, n
                )
            )

        n_train = int(np.floor(0.8 * n))
        n_val = int(np.floor(0.1 * n))
        n_test = n - n_train - n_val
        if n_val == 0:
            n_val = 1
            n_train -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1
        if n_train <= 0:
            raise ValueError(
                "Bot {} does not have enough posts for non-empty 8/1/1 splits.".format(
                    uid
                )
            )

        split_bounds = {
            "train": np.arange(0, n_train),
            "val": np.arange(n_train, n_train + n_val),
            "test": np.arange(n_train + n_val, n),
        }
        for split_name in ("train", "val", "test"):
            piece = group.iloc[split_bounds[split_name]].copy()
            piece["__split"] = split_name
            parts.append(piece)

        train_last = group.iloc[n_train - 1]["__time"]
        val_first = group.iloc[n_train]["__time"]
        val_last = group.iloc[n_train + n_val - 1]["__time"]
        test_first = group.iloc[n_train + n_val]["__time"]
        if not (train_last <= val_first <= val_last <= test_first):
            raise AssertionError("Chronological split order failed for bot {}.".format(uid))

        bot_stats.append(
            {
                "uid": uid,
                "total": n,
                "train": n_train,
                "val": n_val,
                "test": n_test,
                "earliest": group.iloc[0]["__time"],
                "latest": group.iloc[-1]["__time"],
            }
        )

    membership = pd.concat(parts, ignore_index=True)
    split_post_sets = {
        split: set(
            zip(
                membership.loc[membership["__split"] == split, "uid"],
                membership.loc[membership["__split"] == split, "wid"],
            )
        )
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_post_sets[left] & split_post_sets[right]
        if overlap:
            raise AssertionError(
                "Post leakage between {} and {}: {}".format(left, right, list(overlap)[:5])
            )

    print(
        "[INFO] Generated chronological per-bot post split using '{}': "
        "earliest 80% train / next 10% val / latest 10% test".format(time_col)
    )
    for split in ("train", "val", "test"):
        subset = membership[membership["__split"] == split]
        print("[INFO] {}: bots={} posts={}".format(split, subset["uid"].nunique(), len(subset)))
    print("[INFO] Same bot identities are shared across splits; posts are mutually disjoint.")
    print("[INFO] Per-bot split examples: {}".format(bot_stats[:5]))

    train_df = membership[membership["__split"] == "train"].copy()
    val_df = membership[membership["__split"] == "val"].copy()
    test_df = membership[membership["__split"] == "test"].copy()
    return train_df, val_df, test_df, membership


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    criterion: nn.Module,
    average: str = "macro",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    ys: List[int] = []
    preds: List[int] = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        pred = torch.argmax(logits, dim=-1)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        ys.extend(labels.detach().cpu().tolist())
        preds.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(ys, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ys, preds, average=average, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        ys, preds, average="macro", zero_division=0
    )
    return {
        "loss": float(total_loss / max(1, n)),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    label_col: str,
    text_col: str,
    max_len: int,
    batch_size: int,
    seed: int,
):
    le = LabelEncoder()
    train_ds = WeiboTextClsDataset(
        train_df,
        tokenizer,
        max_len=max_len,
        label_encoder=le,
        fit_label_encoder=True,
        text_col=text_col,
        label_col=label_col,
    )

    unseen = set(val_df[label_col].astype(str)) | set(test_df[label_col].astype(str))
    unseen -= set(train_ds.le.classes_)
    if unseen:
        raise ValueError("Validation/test contain labels absent from training: {}".format(sorted(unseen)))

    val_ds = WeiboTextClsDataset(
        val_df,
        tokenizer,
        max_len=max_len,
        label_encoder=train_ds.le,
        fit_label_encoder=False,
        text_col=text_col,
        label_col=label_col,
    )
    test_ds = WeiboTextClsDataset(
        test_df,
        tokenizer,
        max_len=max_len,
        label_encoder=train_ds.le,
        fit_label_encoder=False,
        text_col=text_col,
        label_col=label_col,
    )

    if len(train_ds.le.classes_) != 3:
        raise ValueError(
            "Expected three engagement classes in training, found {}.".format(
                train_ds.le.classes_.tolist()
            )
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl, train_ds.le


def run_one_seed(
    args,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    backbone: str,
    device: str,
) -> Dict[str, float]:
    seed_everything(seed)
    train_dl, val_dl, test_dl, le = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        label_col=args.label_col,
        text_col=args.text_col,
        max_len=args.max_len,
        batch_size=args.batch_size,
        seed=seed,
    )

    num_classes = len(le.classes_)
    model = BertTextClassifier(
        backbone=backbone,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.ckpt_dir,
        "text_only_bert_per_bot_811_seed{}.pt".format(seed),
    )

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics = None

    print("\n===== BERT text-only seed {} =====".format(seed))
    print("Classes={} -> {}".format(num_classes, le.classes_.tolist()))
    print("Split sizes: train={} val={} test={}".format(len(train_df), len(val_df), len(test_df)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs

        train_loss = total_loss / max(1, n)
        train_metrics = evaluate(model, train_dl, device=device, criterion=criterion, average=args.metric_average)
        val_metrics = evaluate(model, val_dl, device=device, criterion=criterion, average=args.metric_average)

        print(
            "[Seed {} | Epoch {}/{}] train_loss={:.4f} | "
            "train_acc={:.4f} train_P={:.4f} train_R={:.4f} train_F1={:.4f} | "
            "val_loss={:.4f} val_acc={:.4f} val_P={:.4f} val_R={:.4f} "
            "val_F1={:.4f} val_MacroF1={:.4f}".format(
                seed,
                epoch,
                args.epochs,
                train_loss,
                train_metrics["acc"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["macro_f1"],
            )
        )

        if val_metrics["macro_f1"] > best_val_macro_f1 + args.early_min_delta:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_variant": "text_only_bert",
                    "split_method": "per-bot chronological post-level 80/10/10",
                    "backbone": backbone,
                    "num_classes": num_classes,
                    "label_encoder_classes": le.classes_.tolist(),
                    "text_col": args.text_col,
                    "label_col": args.label_col,
                    "time_col": args.time_col,
                    "max_len": args.max_len,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                },
                ckpt_path,
            )

    if best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was saved.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    test_metrics = evaluate(model, test_dl, device=device, criterion=criterion, average=args.metric_average)
    train_metrics = evaluate(model, train_dl, device=device, criterion=criterion, average=args.metric_average)

    print(
        "[TEST seed={}] loss={:.4f} acc={:.4f} P={:.4f} R={:.4f} F1={:.4f} MacroF1={:.4f}".format(
            seed,
            test_metrics["loss"],
            test_metrics["acc"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["macro_f1"],
        )
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_metrics["loss"]),
        "best_val_acc": float(best_val_metrics["acc"]),
        "best_val_precision": float(best_val_metrics["precision"]),
        "best_val_recall": float(best_val_metrics["recall"]),
        "best_val_f1": float(best_val_metrics["f1"]),
        "best_val_macro_f1": float(best_val_macro_f1),
        "train_loss": float(train_metrics["loss"]),
        "train_acc": float(train_metrics["acc"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_f1": float(train_metrics["f1"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "ckpt_path": ckpt_path,
    }


def compute_mean_std(records: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = [key for key in records[0].keys() if key not in {"seed", "best_epoch", "ckpt_path"}]
    summary = {}
    for key in keys:
        values = np.array([record[key] for record in records], dtype=float)
        summary[key] = (
            float(values.mean()),
            float(values.std(ddof=1) if len(values) > 1 else 0.0),
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-only BERT engagement classification with per-bot chronological 8/1/1 splitting."
    )
    parser.add_argument("--csv_path", default="aigc_new_with_style_features_with_engagement_class.csv")
    parser.add_argument("--csv_encoding", default="utf-8-sig")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="BERT checkpoint path or model name.")
    parser.add_argument("--text_col", default="text_raw")
    parser.add_argument("--label_col", default="engagement_class")
    parser.add_argument("--time_col", default="create_time")
    parser.add_argument("--output_dir", default="text_only_bert_outputs")
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--run_multi_seed", action="store_true")
    parser.add_argument("--metric_average", default="macro")
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.csv_path = os.path.abspath(args.csv_path)
    args.output_dir = os.path.abspath(args.output_dir)
    if not args.ckpt_dir:
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")
    else:
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    seed_everything(args.seed)
    backbone = resolve_local_path(args.backbone)
    print("[INFO] MODEL VARIANT = TEXT ONLY BERT ([CLS] + Linear)")
    print("[INFO] backbone={}".format(backbone))
    print("[INFO] csv_path={}".format(args.csv_path))
    print("[INFO] split=per-bot chronological 8/1/1; time_col={}".format(args.time_col))

    df = read_csv_with_fallback(
        args.csv_path,
        [args.csv_encoding, "utf-8-sig", "utf-8", "gbk"],
    )
    train_df, val_df, test_df, membership = split_per_bot_chronological_8_1_1(
        df,
        text_col=args.text_col,
        label_col=args.label_col,
        time_col=args.time_col,
    )

    manifest_cols = ["uid", "wid", args.time_col, args.label_col, "__split"]
    membership[manifest_cols].to_csv(
        os.path.join(args.output_dir, "split_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    run_seeds = parse_seeds(args.seeds) if args.run_multi_seed else [args.seed]
    results = []
    for seed in run_seeds:
        results.append(
            run_one_seed(
                args=args,
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tokenizer=tokenizer,
                backbone=backbone,
                device=args.device,
            )
        )

    results_df = pd.DataFrame(results).sort_values("seed")
    results_csv = os.path.join(args.output_dir, "multi_seed_results.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    summary = compute_mean_std(results)
    with open(os.path.join(args.output_dir, "multi_seed_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "model_variant": "text_only_bert",
                "uses_text": True,
                "uses_roberta": False,
                "uses_topology": False,
                "uses_style": False,
                "split_method": "per-bot chronological post-level 80/10/10",
                "time_col": args.time_col,
                "backbone": backbone,
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

    print("[DONE] Outputs saved in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
