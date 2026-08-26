# -*- coding: utf-8 -*-
"""
Role-conditioned (role + style FiLM) text classifier for engagement_class (multiclass).

Pipeline:
  text_raw -> BERT -> FiLM(role_id, style_vec) -> CLS_tilde -> Linear -> engagement_class

Required CSV columns (defaults; configurable in main()):
  - text_raw
  - engagement_class
  - uid                 (role id key; will be mapped to 0..num_roles-1)
  - style feature cols  (auto-detected by prefix; or specify style_cols in main())

Run:
  conda run python role_conditioned_text_multiclass.py

Notes:
  - Uses 8:1:1 split (stratified when possible).
  - Metrics: Accuracy / Precision / Recall / F1 (macro by default).
  - No graph features are used. Only role+style-conditioned text.
"""
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ===== LoRA (optional) =====
USE_LORA = False
if USE_LORA:
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "USE_LORA=True 但环境里无法导入 peft。请先安装 peft，或把 USE_LORA=False。"
        ) from e


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _guess_style_cols(df: pd.DataFrame, text_col: str, label_col: str, role_col: str) -> List[str]:
    """Heuristics: prefer style_* then feat_*; else numeric columns excluding key cols."""
    key_exclude = {text_col, label_col, role_col}

    style_cols = [c for c in df.columns if c.startswith("style_")]
    if style_cols:
        return style_cols

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if feat_cols:
        return feat_cols

    # last resort: any numeric cols except keys
    num_cols = []
    for c in df.columns:
        if c in key_exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


# ========= Dataset: text + role_id + style_vec + engagement_class =========
class WeiboRoleStyleTextClsDataset(Dataset):
    """
    Required columns:
      - text_col (default: text_raw)
      - label_col (default: engagement_class)
      - role_col (default: uid)
      - style_cols: list[str] (continuous style vector columns; will be StandardScaler'ed)

    Returns keys:
      input_ids, attention_mask, role_id, style_vec, labels
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        style_cols: List[str],
        max_len: int = 128,
        label_encoder: Optional[LabelEncoder] = None,
        fit_label_encoder: bool = False,
        role_vocab: Optional[Dict[str, int]] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        text_col: str = "text_raw",
        label_col: str = "engagement_class",
        role_col: str = "uid",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col
        self.role_col = role_col
        self.style_cols = style_cols

        for col in [text_col, label_col, role_col]:
            if col not in self.df.columns:
                raise ValueError(f"CSV 必须包含列 {col}")

        if not style_cols:
            raise ValueError("style_cols 为空：你需要提供风格特征列（或让脚本自动检测到 style_* / feat_* 等列）。")
        for c in style_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV 缺少风格变量列：{c}")

        self.df[text_col] = self.df[text_col].fillna("").astype(str)

        # ---- role vocab ----
        role_keys = self.df[role_col].astype(str).tolist()
        if role_vocab is None:
            role_vocab = {k: i for i, k in enumerate(sorted(set(role_keys)))}
        self.role_vocab = role_vocab
        self.role_ids = torch.tensor([self.role_vocab.get(k, 0) for k in role_keys], dtype=torch.long)

        # ---- style vector (scaled) ----
        feats = self.df[style_cols].apply(pd.to_numeric, errors="coerce")
        feats = feats.replace([np.inf, -np.inf], np.nan)
        feats = feats.fillna(feats.median(numeric_only=True))

        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            scaler.fit(feats.values)
        self.scaler = scaler
        feats = scaler.transform(feats.values)
        self.style_feats = torch.tensor(feats, dtype=torch.float32)

        # ---- label encoder ----
        if label_encoder is None:
            label_encoder = LabelEncoder()
        raw_y = self.df[label_col].astype(str).values
        if fit_label_encoder:
            label_encoder.fit(raw_y)
        self.le = label_encoder
        y = self.le.transform(raw_y)
        self.labels = torch.tensor(y, dtype=torch.long)

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
            "role_id": self.role_ids[idx],
            "style_vec": self.style_feats[idx],
            "labels": self.labels[idx],
        }


# ========= Role-conditioned text encoder (FiLM) =========
class RoleConditionedTextEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        cond_dim: int = 64,
        style_proj_dim: int = 64,
        film_hidden: int = 256,
        proj_c_dim: int = 256,
        proj_s_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size

        if USE_LORA:
            lora_cfg = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS",
            )
            self.bert = get_peft_model(self.bert, lora_cfg)

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

        self.proj_c = nn.Sequential(nn.Linear(hidden, proj_c_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_s = nn.Sequential(nn.Linear(hidden, proj_s_dim), nn.Tanh(), nn.Dropout(dropout))

    def forward(self, input_ids, attention_mask, role_id, style_vec):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state

        e_role = self.role_emb(role_id)
        z_style = self.style_mlp(style_vec)
        cond = torch.cat([e_role, z_style], dim=-1)

        gamma_beta = self.film_gen(cond)
        Hdim = H.size(-1)
        gamma, beta = gamma_beta[:, :Hdim], gamma_beta[:, Hdim:]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        H_tilde = gamma * H + beta
        cls_tilde = H_tilde[:, 0, :]
        c = self.proj_c(cls_tilde)
        s = self.proj_s(cls_tilde)
        return {"c": c, "s": s, "H": H_tilde, "cls": cls_tilde}


# ========= Classifier uses modulated CLS (cls_tilde) =========
class RoleCondBertClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = RoleConditionedTextEncoder(
            backbone=backbone,
            num_roles=num_roles,
            style_in_dim=style_in_dim,
            cond_dim=64,
            style_proj_dim=64,
            film_hidden=256,
            proj_c_dim=256,
            proj_s_dim=128,
            dropout=dropout,
        )
        hidden = self.encoder.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, role_id, style_vec):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_id=role_id,
            style_vec=style_vec,
        )
        cls_tilde = enc_out["cls"]
        logits = self.classifier(self.dropout(cls_tilde))
        return logits


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: str, average: str = "macro") -> Dict[str, float]:
    model.eval()
    ys: List[int] = []
    preds: List[int] = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        role_id = batch["role_id"].to(device)
        style_vec = batch["style_vec"].to(device)
        y = batch["labels"].cpu().numpy().tolist()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_id=role_id,
            style_vec=style_vec,
        )
        p = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()

        ys.extend(y)
        preds.extend(p)

    acc = accuracy_score(ys, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average=average, zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def split_8_1_1(df: pd.DataFrame, label_col: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[label_col].astype(str).values
    try:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=y)
        y_temp = temp_df[label_col].astype(str).values
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=y_temp)
    except ValueError:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)
    return train_df, val_df, test_df


def main(
    csv_path: str = "with_engagement_class.csv",
    backbone: str = "bert-based-chinese",
    text_col: str = "text_raw",
    label_col: str = "engagement_class",
    role_col: str = "uid",
    style_cols=["length",
        "emoji_count",
        "is_qa",
        "sentiment",
        "ttr",
        "rttr",
        "mtld",
        "msttr",
        "common_ratio",
        "stop_ratio",],
    max_len: int = 128,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    average: str = "macro",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(seed)

    df = pd.read_csv(csv_path)
    for col in [text_col, label_col, role_col]:
        if col not in df.columns:
            raise ValueError(f"CSV 必须包含列 {col}")

    if style_cols is None:
        style_cols = _guess_style_cols(df, text_col=text_col, label_col=label_col, role_col=role_col)
        print(f"[Auto] style_cols={style_cols}")
    if not style_cols:
        raise ValueError("未能检测到风格特征列。请在 main(style_cols=[...]) 里显式指定。")

    train_df, val_df, test_df = split_8_1_1(df, label_col=label_col, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # label encoder fit on train only
    le = LabelEncoder()

    # role vocab on full df (avoid unseen role in val/test)
    role_vocab = {k: i for i, k in enumerate(sorted(set(df[role_col].astype(str).tolist())))}

    # scaler fit on train only
    scaler = StandardScaler()

    train_ds = WeiboRoleStyleTextClsDataset(
        train_df, tokenizer=tokenizer, style_cols=style_cols, max_len=max_len,
        label_encoder=le, fit_label_encoder=True,
        role_vocab=role_vocab,
        scaler=scaler, fit_scaler=True,
        text_col=text_col, label_col=label_col, role_col=role_col
    )
    val_ds = WeiboRoleStyleTextClsDataset(
        val_df, tokenizer=tokenizer, style_cols=style_cols, max_len=max_len,
        label_encoder=train_ds.le, fit_label_encoder=False,
        role_vocab=role_vocab,
        scaler=train_ds.scaler, fit_scaler=False,
        text_col=text_col, label_col=label_col, role_col=role_col
    )
    test_ds = WeiboRoleStyleTextClsDataset(
        test_df, tokenizer=tokenizer, style_cols=style_cols, max_len=max_len,
        label_encoder=train_ds.le, fit_label_encoder=False,
        role_vocab=role_vocab,
        scaler=train_ds.scaler, fit_scaler=False,
        text_col=text_col, label_col=label_col, role_col=role_col
    )

    num_classes = len(train_ds.le.classes_)
    num_roles = len(role_vocab)
    print(f"Classes={num_classes} -> {list(train_ds.le.classes_)}")
    print(f"Roles={num_roles} -> sample role_vocab={dict(list(role_vocab.items())[:8])}")
    print(f"Style dim={len(style_cols)}")
    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = RoleCondBertClassifier(
        backbone=backbone,
        num_roles=num_roles,
        style_in_dim=len(style_cols),
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    os.makedirs("ckpt", exist_ok=True)
    ckpt_path = "ckpt/rolecond_text_engagement_class.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            role_id = batch["role_id"].to(device)
            style_vec = batch["style_vec"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                role_id=role_id,
                style_vec=style_vec,
            )
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs

        val_metrics = evaluate(model, val_dl, device=device, average=average)
        train_loss = total_loss / max(1, n)

        print(
            f"[Epoch {ep}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_P={val_metrics['precision']:.4f} "
            f"val_R={val_metrics['recall']:.4f} "
            f"val_F1={val_metrics['f1']:.4f} ({average})"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": backbone,
                    "num_classes": num_classes,
                    "label_encoder_classes": train_ds.le.classes_.tolist(),
                    "role_vocab": role_vocab,
                    "style_cols": style_cols,
                    "scaler_mean": train_ds.scaler.mean_.tolist(),
                    "scaler_scale": train_ds.scaler.scale_.tolist(),
                    "text_col": text_col,
                    "label_col": label_col,
                    "role_col": role_col,
                    "max_len": max_len,
                },
                ckpt_path,
            )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_dl, device=device, average=average)

    print("\n========== Test metrics ==========")
    print(
        f"test_acc={test_metrics['acc']:.4f} "
        f"test_P={test_metrics['precision']:.4f} "
        f"test_R={test_metrics['recall']:.4f} "
        f"test_F1={test_metrics['f1']:.4f} ({average})"
    )
    print(f"✅ best ckpt saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
