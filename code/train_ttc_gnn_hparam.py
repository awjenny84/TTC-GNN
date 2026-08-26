# -*- coding: utf-8 -*-
"""
End-to-end 参与度多类别预测：Text Encoder + Graph Encoder(可选 GT / GraphSAGE) + 融合（Dual-Co-Attention）
并可选加入“图自监督损失”（随机游走 + 负采样）来复现你截图里的 GraphSAGE 无监督目标。

用法示例：
  # 纯端到端（只用参与度监督，不加图自监督）
  conda run python train_multiclass_engagement_e2e.py --lambda_unsup 0

  # 端到端 + 图自监督（复现截图公式思想：random-walk positives + negative sampling）
  conda run python train_multiclass_engagement_e2e.py --lambda_unsup 0.1 --unsup_every 1 --graph_encoder sage

注意：
- 端到端不是“分开训练两个模型”，而是一个整体模型里有两个可学习子模块（text encoder / graph encoder），
  用同一个 loss（或 loss 的加权和）一次反传同时更新参数。
- 结构向量取法：用 role_id 映射到图中“智能体节点”的 node_index，再去拿该节点的 embedding。
  你需要保证 CSV 的 role_id 和 nodes.csv 的 role_id 能对齐（最省事：两边都是 0..7）。
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from torch_geometric.data import Data  # noqa: F401 (kept for clarity)
from torch_geometric.nn import GATConv, SAGEConv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_EXTRA_NODE_CSV = "fan_bfs_nodes.csv"

# -------------------------
# 0) Utils
# -------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"acc": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """edge_index: [2, E] (torch.long, CPU)
    返回每个节点的邻居列表（当作无向图：双向加入）
    """
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip("'").str.strip('"')


def is_disabled_path_list(value: str) -> bool:
    return bool(value) and str(value).strip().lower() in {"none", "null", "-"}


def split_path_list(value: str) -> List[str]:
    if not value or is_disabled_path_list(value):
        return []
    paths = []
    for path in str(value).replace(";", ",").split(","):
        path = path.strip().strip("'").strip('"')
        if path:
            paths.append(path)
    return paths


def resolve_relative_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def read_csv_with_fallback(path: str, encodings) -> pd.DataFrame:
    tried = []
    for encoding in dict.fromkeys(encodings):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                dtype={"uid": str, "uid_1": str, "seed_uid": str},
            )
        except UnicodeDecodeError:
            tried.append(encoding)
    raise UnicodeDecodeError(
        encodings[0],
        b"",
        0,
        1,
        "Could not decode CSV {} with encodings: {}".format(path, tried),
    )


def find_uid_column(frame: pd.DataFrame, context: str) -> str:
    for column in ("uid", "uid_1", "seed_uid"):
        if column in frame.columns:
            return column
    raise ValueError("{} must contain a uid-like column: uid, uid_1, or seed_uid.".format(context))


def resolve_extra_nodes_csvs(args) -> List[str]:
    if is_disabled_path_list(getattr(args, "extra_nodes_csv", "")):
        return []

    base_dir = os.path.dirname(os.path.abspath(args.nodes_csv)) or os.getcwd()
    primary = os.path.abspath(args.nodes_csv)
    paths = [
        resolve_relative_path(path, base_dir)
        for path in split_path_list(getattr(args, "extra_nodes_csv", ""))
    ]
    paths = [path for path in dict.fromkeys(paths) if path != primary]

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Cannot find extra node CSV(s): {}. Use --extra_nodes_csv none to disable.".format(missing)
        )
    return paths


def prepare_node_frame(path: str, frame: pd.DataFrame) -> pd.DataFrame:
    uid_col = find_uid_column(frame, "nodes CSV {}".format(path))
    frame = frame.copy()
    if uid_col != "uid":
        if "uid" in frame.columns:
            frame["uid"] = frame["uid"].where(frame["uid"].notna(), frame[uid_col])
        else:
            frame = frame.rename(columns={uid_col: "uid"})
    frame["uid"] = normalize_id(frame["uid"])
    frame = frame[frame["uid"] != ""].copy()
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame["__node_source"] = os.path.basename(path)
    return frame


def read_nodes_csvs(args) -> Tuple[pd.DataFrame, List[str]]:
    primary = os.path.abspath(args.nodes_csv)
    if not os.path.isfile(primary):
        raise FileNotFoundError("Cannot find nodes CSV: {}".format(primary))

    node_paths = [primary] + resolve_extra_nodes_csvs(args)
    encodings = [args.nodes_encoding, "utf-8-sig", "utf-8", "gbk"]
    frames = [
        prepare_node_frame(path, read_csv_with_fallback(path, encodings))
        for path in node_paths
    ]
    nodes = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_uids = int(nodes["uid"].duplicated().sum()) if "uid" in nodes.columns else 0
    print(
        "[INFO] Loaded node CSVs in graph order: {} -> total_nodes={} duplicate_uids={}".format(
            ", ".join("{}({})".format(os.path.basename(path), len(frame)) for path, frame in zip(node_paths, frames)),
            len(nodes),
            duplicate_uids,
        )
    )
    return nodes, node_paths


def make_chronological_per_bot_split(
    df: pd.DataFrame,
    label_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic per-bot chronological 80/10/10 post split.

    This mirrors train_multiclass_engagement_e2e_5seeds_per_bot_811.py:
    sort each bot's posts by timestamp, assign the earliest 80% to train,
    the next 10% to validation, and the latest remainder to test. Bots are
    shared across splits by design, but (uid, wid) posts are disjoint.
    """
    required = {"uid", "wid", label_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing columns required for per-bot chronological split: {}".format(
                sorted(missing)
            )
        )

    df = df.copy()
    df["uid"] = normalize_id(df["uid"])
    df["wid"] = normalize_id(df["wid"])
    df = df[(df["uid"] != "") & (df["wid"] != "")].copy()
    if df.empty:
        raise ValueError("No rows remain after removing empty uid/wid values.")

    df["__time"] = pd.to_datetime(df[time_col], errors="coerce")
    bad_time = df["__time"].isna()
    if bad_time.any():
        examples = df.loc[bad_time, ["uid", "wid", time_col]].head().to_dict("records")
        raise ValueError(
            "Invalid or missing timestamps in '{}'; examples: {}".format(
                time_col, examples
            )
        )

    duplicate_mask = df.duplicated(["uid", "wid", label_col], keep="first")
    if duplicate_mask.any():
        print("[WARN] dropped {} exact duplicate post rows before splitting.".format(
            int(duplicate_mask.sum())
        ))
        df = df.loc[~duplicate_mask].copy()

    conflicting = df.duplicated(["uid", "wid"], keep=False)
    if conflicting.any():
        examples = df.loc[conflicting, ["uid", "wid", label_col]].head().to_dict("records")
        raise ValueError("Duplicate (uid, wid) rows with conflicting labels: {}".format(examples))

    df = df.reset_index(drop=True)
    parts = []
    bot_stats = []
    for uid, group in df.groupby("uid", sort=True):
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
            raise ValueError("Bot {} does not have enough posts for non-empty 8/1/1 splits.".format(uid))

        boundaries = {
            "train": np.arange(0, n_train),
            "val": np.arange(n_train, n_train + n_val),
            "test": np.arange(n_train + n_val, n),
        }
        for split in ("train", "val", "test"):
            piece = group.iloc[boundaries[split]].copy()
            piece["__split"] = split
            parts.append(piece)

        train_last = group.iloc[n_train - 1]["__time"]
        val_first = group.iloc[n_train]["__time"]
        val_last = group.iloc[n_train + n_val - 1]["__time"]
        test_first = group.iloc[n_train + n_val]["__time"]
        if not (train_last <= val_first <= val_last <= test_first):
            raise AssertionError("Chronological split order failed for bot {}.".format(uid))

        bot_stats.append((
            uid, n, n_train, n_val, n_test,
            group.iloc[0]["__time"], group.iloc[-1]["__time"],
        ))

    split_df = pd.concat(parts, ignore_index=True)
    split_post_sets = {
        split: set(zip(
            split_df.loc[split_df["__split"] == split, "uid"],
            split_df.loc[split_df["__split"] == split, "wid"],
        ))
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_post_sets[left] & split_post_sets[right]
        if overlap:
            raise AssertionError("Post leakage between {} and {}: {}".format(left, right, list(overlap)[:5]))

    print(
        "[INFO] Generated chronological per-bot post split using '{}': "
        "earliest 80% train / next 10% val / latest 10% test".format(time_col)
    )
    for split in ("train", "val", "test"):
        subset = split_df[split_df["__split"] == split]
        print("[INFO] {}: bots={} posts={}".format(split, subset["uid"].nunique(), len(subset)))
    print("[INFO] Same bot identities are intentionally shared across train/val/test; posts are mutually disjoint.")
    print(
        "[INFO] Per-bot split examples "
        "(uid,total,train,val,test,earliest,latest): {}".format(bot_stats[:5])
    )

    split_df = split_df.drop(columns=["__time"], errors="ignore").reset_index(drop=True)
    idx_train = np.flatnonzero(split_df["__split"].to_numpy() == "train")
    idx_val = np.flatnonzero(split_df["__split"].to_numpy() == "val")
    idx_test = np.flatnonzero(split_df["__split"].to_numpy() == "test")
    return split_df, idx_train, idx_val, idx_test


def random_walk_one(start: int, adj: List[List[int]], walk_len: int, rng: np.random.Generator) -> List[int]:
    """简单随机游走：每步在邻居里均匀采样一个节点，返回包含 start 在内的节点序列"""
    walk = [start]
    cur = start
    for _ in range(walk_len):
        neigh = adj[cur]
        if len(neigh) == 0:
            break
        cur = int(rng.choice(neigh))
        walk.append(cur)
    return walk


def sample_rw_positives(
    roots: torch.Tensor,
    adj: List[List[int]],
    walk_len: int,
    num_pos: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """对每个 root 做一次 random walk，从访问到的节点里采样 num_pos 个 positive context
    返回 pos_nodes: [B, num_pos]
    """
    roots_np = roots.detach().cpu().numpy().tolist()
    pos = []
    for r in roots_np:
        walk = random_walk_one(int(r), adj, walk_len, rng)
        ctx = [v for v in walk[1:]] if len(walk) > 1 else []
        if len(ctx) == 0:
            ctx = [int(r)]
        chosen = rng.choice(ctx, size=num_pos, replace=True).tolist()
        pos.append(chosen)
    return torch.tensor(pos, dtype=torch.long, device=roots.device)


def unsup_rw_neg_sampling_loss(
    z: torch.Tensor,
    roots: torch.Tensor,
    pos_nodes: torch.Tensor,
    num_neg: int,
) -> torch.Tensor:
    """复现截图里的无监督目标（random-walk positives + negative sampling）：
    L = - E_r [ sum_{vp in P_r} log σ(z_r^T z_vp) + sum_{vn in N_r} log σ(- z_r^T z_vn) ]
    这里用 mean 近似。
    """
    device = z.device
    B, P = pos_nodes.shape
    N = z.size(0)

    z_r = z[roots]  # [B, D]
    z_p = z[pos_nodes]  # [B, P, D]
    pos_score = (z_r.unsqueeze(1) * z_p).sum(dim=-1)  # [B, P]
    pos_loss = -F.logsigmoid(pos_score).mean()

    neg_nodes = torch.randint(low=0, high=N, size=(B, num_neg), device=device)
    z_n = z[neg_nodes]  # [B, K, D]
    neg_score = (z_r.unsqueeze(1) * z_n).sum(dim=-1)  # [B, K]
    neg_loss = -F.logsigmoid(-neg_score).mean()

    return pos_loss + neg_loss


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
# 2) Text Encoder
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
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
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
        H = out.last_hidden_state  # [B, L, hidden]

        e_role = self.role_emb(role_id)
        z_style = self.style_mlp(style_vec)
        cond = torch.cat([e_role, z_style], dim=-1)

        gamma_beta = self.film_gen(cond)
        Hdim = H.size(-1)
        gamma, beta = gamma_beta[:, :Hdim], gamma_beta[:, Hdim:]
        H_tilde = gamma.unsqueeze(1) * H + beta.unsqueeze(1)
        return self.dropout(H_tilde)


# -------------------------
# 3) Graph Encoders
# -------------------------


class GraphTransformerEncoder(nn.Module):
    """GATConv 堆叠 + FFN + LN，作为结构 encoder，输出所有节点 embedding"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.dropout = dropout


        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))
            self.ln1.append(nn.LayerNorm(hidden_dim))
            self.ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                )
            )
            self.ln2.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for conv, ln1, ffn, ln2 in zip(self.convs, self.ln1, self.ffn, self.ln2):
            out = conv(x, edge_index)
            h = F.gelu(out)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = ln1(x + h)
            x = ln2(x + ffn(x))
        return x


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE encoder（更贴近你截图的设置）"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -------------------------
# 4) Fusion + classifier
# -------------------------


class DualCoAttentionClassifier(nn.Module):
    def __init__(self, dim_text: int, dim_graph: int, num_classes: int, d_att: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_t = nn.Linear(dim_text, d_att)
        self.kg = nn.Linear(dim_graph, d_att)
        self.vg = nn.Linear(dim_graph, d_att)

        self.q_g = nn.Linear(dim_graph, d_att)
        self.kt = nn.Linear(dim_text, d_att)
        self.vt = nn.Linear(dim_text, d_att)

        self.att_t2g = nn.MultiheadAttention(d_att, nhead, dropout=dropout, batch_first=True)
        self.att_g2t = nn.MultiheadAttention(d_att, nhead, dropout=dropout, batch_first=True)

        self.ln_t = nn.LayerNorm(d_att)
        self.ln_g = nn.LayerNorm(d_att)

        self.pred = nn.Sequential(
            nn.Linear(2 * d_att, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, H_text: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
        if g_vec.dim() == 2:
            G = g_vec.unsqueeze(1)
        else:
            G = g_vec

        # Text -> Graph
        H_t2g, _ = self.att_t2g(self.q_t(H_text), self.kg(G), self.vg(G), need_weights=False)
        H_t2g = self.ln_t(H_t2g)

        # Graph -> Text
        G_g2t, _ = self.att_g2t(self.q_g(G), self.kt(H_text), self.vt(H_text), need_weights=False)
        G_g2t = self.ln_g(G_g2t)

        fused = torch.cat([H_t2g[:, 0, :], G_g2t.mean(dim=1)], dim=-1)
        return self.pred(fused)


# -------------------------
# 5) E2E wrapper
# -------------------------


class EngagementE2EModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        graph_in_dim: int,
        num_classes: int,
        graph_encoder: str = "gt",
        graph_hidden: int = 64,
    ):
        super().__init__()
        self.text_enc = RoleConditionedTextEncoder(backbone, num_roles, style_in_dim)
        dt = self.text_enc.bert.config.hidden_size

        if graph_encoder == "sage":
            self.graph_enc = GraphSAGEEncoder(graph_in_dim, hidden_dim=graph_hidden, num_layers=2)
        else:
            self.graph_enc = GraphTransformerEncoder(graph_in_dim, hidden_dim=graph_hidden, num_layers=2)

        self.fuser = DualCoAttentionClassifier(dim_text=dt, dim_graph=graph_hidden, num_classes=num_classes)
        self.g_null = nn.Parameter(torch.zeros(graph_hidden))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_id: torch.Tensor,
        style_vec: torch.Tensor,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        role_to_node: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_text = self.text_enc(input_ids, attention_mask, role_id, style_vec)
        z_nodes = self.graph_enc(graph_x, graph_edge_index)
        node_idx = role_to_node[role_id]
        mask = (node_idx >= 0)  # [B] bool

        safe_idx = node_idx.clamp(min=0)  # 防止索引报错
        g_vec = z_nodes[safe_idx]  # 先取一个“占位”的
        g_vec = torch.where(mask.unsqueeze(1), g_vec, self.g_null.expand_as(g_vec))
        logits = self.fuser(H_text, g_vec)
        return logits, z_nodes


# -------------------------
# 6) Train/Eval
# -------------------------


@torch.no_grad()
def evaluate(
    model,
    dl: DataLoader,
    graph_x: torch.Tensor,
    graph_edge_index: torch.Tensor,
    role_to_node: torch.Tensor,
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

    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            role_id = batch["role_id"].to(device)
            style_vec = batch["style_vec"].to(device)
            y = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask, role_id, style_vec,
                              graph_x, graph_edge_index, role_to_node)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="aigc_new_with_style_features_with_engagement_class.csv")
    ap.add_argument("--label_col", type=str, default="engagement_class")
    ap.add_argument("--text_col", type=str, default="text_raw")
    ap.add_argument("--role_col", type=str, default="role_id")
    ap.add_argument("--time_col", type=str, default="create_time")
    ap.add_argument("--nodes_csv", type=str, default="nodes1.csv")
    ap.add_argument("--extra_nodes_csv", type=str, default=DEFAULT_EXTRA_NODE_CSV)
    ap.add_argument("--nodes_encoding", type=str, default="gbk")
    ap.add_argument("--graph_x", type=str, default="node_features_raw.npy")
    ap.add_argument("--edge_index", type=str, default="edge_index.npy")
    ap.add_argument(
        "--feature_cols",
        type=str,
        default="length,emoji_count,is_qa,sentiment_score,TTR,RTTR,MTLD,MSTTR,common_ratio,stop_ratio",
    )

    ap.add_argument("--backbone", type=str, default="chinese-roberta-wwm-ext")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42, help="Deprecated; per-bot chronological splitting is deterministic.")
    ap.add_argument("--metric_average", type=str, default="macro")

    ap.add_argument("--graph_encoder", type=str, default="sage", choices=["gt", "sage"])
    ap.add_argument("--graph_hidden", type=int, default=64)
    ap.add_argument("--early_patience", type=int, default=99, help="val_f1 连续多少轮不提升就停止")
    ap.add_argument("--early_min_delta", type=float, default=1e-4, help="认为提升的最小幅度")

    # 图自监督（复现截图思想）
    ap.add_argument("--lambda_unsup", type=float, default=0.0)
    ap.add_argument("--unsup_every", type=int, default=1)
    ap.add_argument("--rw_len", type=int, default=5)
    ap.add_argument("--rw_pos", type=int, default=5)
    ap.add_argument("--rw_neg", type=int, default=10)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- CSV --------
    df = pd.read_csv(args.csv_path, dtype={"uid": str, "wid": str})
    if args.text_col not in df.columns:
        raise ValueError(f"CSV 必须包含列 {args.text_col}")
    if args.role_col not in df.columns:
        raise ValueError(f"CSV 必须包含列 {args.role_col}（建议用 0..7 这种 role_id 与 nodes.csv 对齐）")
    if args.label_col not in df.columns:
        raise ValueError(f"CSV 必须包含标签列 {args.label_col}")

    df, idx_train, idx_val, idx_test = make_chronological_per_bot_split(
        df,
        label_col=args.label_col,
        time_col=args.time_col,
    )

    df[args.text_col] = df[args.text_col].fillna("").astype(str)

    role_ids_all = torch.tensor(df[args.role_col].astype(int).values, dtype=torch.long)
    num_roles = int(role_ids_all.max().item() + 1)

    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for c in feat_cols:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少风格变量列：{c}")
    feats = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    le = LabelEncoder()
    labels_all = df[args.label_col].astype(str).fillna("NA")
    le.fit(labels_all.iloc[idx_train].values)
    unseen = set(labels_all.values) - set(le.classes_)
    if unseen:
        raise ValueError("Non-training split(s) contain labels absent from training: {}".format(sorted(unseen)))
    y_all_np = le.transform(labels_all.values)
    y_all = torch.tensor(y_all_np, dtype=torch.long)
    num_classes = int(len(le.classes_))
    print(f"[INFO] num_classes={num_classes}, classes={list(le.classes_)}")

    scaler = StandardScaler()
    train_medians = feats.iloc[idx_train].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    if not np.isfinite(feats.to_numpy(dtype=np.float64)).all():
        raise ValueError("Style features still contain NaN/Inf after train-split median imputation.")
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

    ds_train = make_ds(idx_train)
    ds_val = make_ds(idx_val)
    ds_test = make_ds(idx_test)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -------- Graph --------
    nodes, node_paths = read_nodes_csvs(args)
    X = np.load(args.graph_x)
    edge_index = np.load(args.edge_index)
    if len(nodes) != X.shape[0]:
        raise ValueError(
            "Node CSV rows ({}) do not match graph_x rows ({}). "
            "Rebuild {} and {} from the same node sources, or pass matching "
            "--graph_x/--edge_index files. node_sources={}".format(
                len(nodes),
                X.shape[0],
                args.graph_x,
                args.edge_index,
                node_paths,
            )
        )
    graph_x = torch.tensor(X, dtype=torch.float32, device=device)
    graph_edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    N, graph_in_dim = X.shape

    if not os.path.exists(args.nodes_csv):
        raise FileNotFoundError(f"找不到 {args.nodes_csv}，用于 role_id -> node_index 对齐。")
    if "role_id" not in nodes.columns:
        raise ValueError("nodes.csv 必须包含 role_id 列（ai 节点 0..7，用户节点 -1）")

    role_to_node = torch.full((num_roles,), -1, dtype=torch.long)
    role_values = pd.to_numeric(nodes["role_id"], errors="coerce")
    for node_idx, rid in enumerate(role_values.values.tolist()):
        if pd.isna(rid):
            continue
        rid = int(rid)
        if 0 <= rid < num_roles and role_to_node[rid] < 0:
            role_to_node[rid] = int(node_idx)

    missing = torch.where(role_to_node < 0)[0].tolist()
    if missing:
        print(f"[WARN] role_id {missing} 在 nodes.csv 里没找到对应节点，将为这些样本使用 g_null（仅文本/风格通道）进行预测。")
    role_to_node = role_to_node.to(device)

    # 无监督损失用邻接表（CPU）
    adj = build_adj_list(torch.tensor(edge_index, dtype=torch.long), num_nodes=N)
    rng = np.random.default_rng(args.seed)

    # -------- Model --------
    model = EngagementE2EModel(
        backbone=args.backbone,
        num_roles=num_roles,
        style_in_dim=len(feat_cols),
        graph_in_dim=graph_in_dim,
        num_classes=num_classes,
        graph_encoder=args.graph_encoder,
        graph_hidden=args.graph_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    global_step = 0
    no_improve = 0
    best_epoch = -1

    # unique checkpoint path per (hyperparams, seed)
    ckpt_path = os.path.join(
        "ckpt",
        f"best_maxlen{args.max_len}_gh{args.graph_hidden}_seed{args.seed}_perbot811.pt",
    )

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
                # roots：用智能体节点做 root（更贴近“网络影响力”学习）
                roots = role_to_node[torch.arange(num_roles, device=device)]
                roots = roots[roots >= 0]
                if roots.numel() > 0:
                    pos_nodes = sample_rw_positives(roots, adj, args.rw_len, args.rw_pos, rng)
                    loss_unsup = unsup_rw_neg_sampling_loss(z_nodes, roots, pos_nodes, args.rw_neg)
                    loss = loss + args.lambda_unsup * loss_unsup
                else:
                    # 所有智能体都缺拓扑（极端情况），跳过无监督项
                    pass

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n += bs
            global_step += 1

        train_loss = total_loss / max(1, n)
        val_loss, val_metrics = evaluate(
            model,
            dl_val,
            graph_x,
            graph_edge_index,
            role_to_node,
            device,
            criterion,
            average=args.metric_average,
        )

        print(
            f"[Epoch {ep+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
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
                    "num_roles": num_roles,
                    "num_classes": num_classes,
                    "label_classes": le.classes_.tolist(),
                    "feature_cols": feat_cols,
                    "graph_encoder": args.graph_encoder,
                    "graph_hidden": args.graph_hidden,
                    "role_col": args.role_col,
                    "seed": args.seed,
                    "split_method": "per-bot chronological post-level 80/10/10",
                    "time_col": args.time_col,
                    "node_sources": node_paths,
                    "graph_x": args.graph_x,
                    "edge_index": args.edge_index,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            print(
                f"[EarlyStop] no_improve={no_improve}/{args.early_patience} (best_f1={best_val_f1:.4f} @ epoch {best_epoch})")
            if no_improve >= args.early_patience:
                print(f"[EarlyStop] Stop at epoch {ep + 1}. Best epoch={best_epoch}, best_val_f1={best_val_f1:.4f}")
                break
    # 先确保你加载的是 best ckpt（很关键）
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 跑 test（val 同理）
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, dl_test, graph_x, graph_edge_index, role_to_node,
        device, criterion, average="macro", return_preds=True
    )

    print("[TEST_DETAIL]", test_loss, test_metrics)

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    labels = np.arange(num_classes)

    # 1) 原始计数混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (counts):\n", cm)

    # 2) 按真实类归一化（每行和为1，更直观）
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    print("Confusion matrix (normalize=true):\n", cm_norm)

    # 3) 保存 CSV
    np.savetxt("cm_test_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt("cm_test_norm_true.csv", cm_norm, fmt="%.6f", delimiter=",")

    # 4) 画图保存（不指定颜色，默认就行）
    class_names = [str(i) for i in labels]  # 或换成你自己的类别名列表
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format="d", cmap=plt.cm.Blues)
    for text in disp.text_.ravel():
        text.set_color("black")
        text.set_fontsize(11)

    plt.tight_layout()
    plt.savefig("cm_test.png", dpi=300)
    plt.close()

    # 5) 可选：更详细的分类报告（每类precision/recall/f1）
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4))

    test_loss, test_metrics = evaluate(
        model,
        dl_test,
        graph_x,
        graph_edge_index,
        role_to_node,
        device,
        criterion,
        average=args.metric_average,
    )
    print(
        f"[TEST] loss={test_loss:.4f} | "
        f"acc={test_metrics['acc']:.4f} "
        f"P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f} ({args.metric_average})"
    )

    os.makedirs("ckpt", exist_ok=True)
    with open("ckpt/style_scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, ensure_ascii=False)
    with open("ckpt/label_encoder_classes.json", "w", encoding="utf-8") as f:
        json.dump({"label_col": args.label_col, "classes": le.classes_.tolist()}, f, ensure_ascii=False)

    print(f"[DONE] Saved: {ckpt_path}, ckpt/style_scaler.json, ckpt/label_encoder_classes.json")


if __name__ == "__main__":
    main()
