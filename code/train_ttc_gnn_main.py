import os
import json
import csv
import random
import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from torch_geometric.nn import GATConv, SAGEConv
from topology_feature_encoder import TopologyFeatureEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


# -------------------------
# 0) Utils
# -------------------------

def set_seed(seed: int, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_seeds(seed_str: str) -> List[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def compute_mean_std(run_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = run_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in run_metrics], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }


def build_adj_list(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


def random_walk_one(start: int, adj: List[List[int]], walk_len: int, rng: np.random.Generator) -> List[int]:
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
    device = z.device
    B, P = pos_nodes.shape
    N = z.size(0)

    z_r = z[roots]
    z_p = z[pos_nodes]
    pos_score = (z_r.unsqueeze(1) * z_p).sum(dim=-1)
    pos_loss = -F.logsigmoid(pos_score).mean()

    neg_nodes = torch.randint(low=0, high=N, size=(B, num_neg), device=device)
    z_n = z[neg_nodes]
    neg_score = (z_r.unsqueeze(1) * z_n).sum(dim=-1)
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
        H = out.last_hidden_state

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
    def __init__(
        self,
        dim_text: int,
        dim_graph: int,
        num_classes: int,
        d_att: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
        pred_hidden: int = 256,
    ):
        super().__init__()
        if d_att % nhead != 0:
            raise ValueError(f"cross-attention hidden dimension {d_att} must be divisible by heads {nhead}.")
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
            nn.Linear(2 * d_att, pred_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, num_classes),
        )

    def forward(self, H_text: torch.Tensor, g_vec: torch.Tensor) -> torch.Tensor:
        G = g_vec.unsqueeze(1) if g_vec.dim() == 2 else g_vec

        H_t2g, _ = self.att_t2g(self.q_t(H_text), self.kg(G), self.vg(G), need_weights=False)
        H_t2g = self.ln_t(H_t2g)

        G_g2t, _ = self.att_g2t(self.q_g(G), self.kt(H_text), self.vt(H_text), need_weights=False)
        G_g2t = self.ln_g(G_g2t)

        fused = torch.cat([H_t2g[:, 0, :], G_g2t.mean(dim=1)], dim=-1)
        return self.pred(fused)


# -------------------------
# 5) E2E wrapper
# -------------------------

class EngagementE2EModel(nn.Module):
    """E2E model using the paper-aligned learnable topology feature encoder.

    Topology pipeline:
        [in/out degree centrality] -> centrality embedding c_v
        [log posting activity]     -> activity embedding a_v (users)
        [bot/user category]        -> type embedding t_v
        x_v = c_v + a_v + t_v
        h_v^(0) = W0 x_v + b0
        h_v^(0) -> GraphSAGE / GAT -> node representation
    """

    def __init__(
        self,
        backbone: str,
        num_roles: int,
        style_in_dim: int,
        graph_in_dim: int,  # retained for call compatibility; no longer used
        num_classes: int,
        graph_encoder: str = "gt",
        graph_hidden: int = 64,
        graph_layers: int = 2,
        att_heads: int = 4,
        dropout: float = 0.1,
        text_cond_dim: int = 64,
        style_proj_dim: int = 64,
        film_hidden: int = 256,
        att_hidden: int = 256,
        pred_hidden: int = 256,
    ):
        super().__init__()
        self.text_enc = RoleConditionedTextEncoder(
            backbone,
            num_roles,
            style_in_dim,
            cond_dim=text_cond_dim,
            style_proj_dim=style_proj_dim,
            film_hidden=film_hidden,
            dropout=dropout,
        )
        dt = self.text_enc.bert.config.hidden_size

        # Implements c_v + a_v + t_v and Eq. (5) before message passing.
        self.topology_feature_enc = TopologyFeatureEncoder(
            graph_dim=graph_hidden,
            dropout=dropout,
        )

        # After the feature encoder, every node already has graph_hidden dims.
        if graph_encoder == "sage":
            self.graph_enc = GraphSAGEEncoder(
                graph_hidden,
                hidden_dim=graph_hidden,
                num_layers=graph_layers,
                dropout=dropout,
            )
        else:
            self.graph_enc = GraphTransformerEncoder(
                graph_hidden,
                hidden_dim=graph_hidden,
                num_layers=graph_layers,
                dropout=dropout,
            )

        self.fuser = DualCoAttentionClassifier(
            dim_text=dt,
            dim_graph=graph_hidden,
            num_classes=num_classes,
            d_att=att_hidden,
            nhead=att_heads,
            dropout=dropout,
            pred_hidden=pred_hidden,
        )
        self.g_null = nn.Parameter(torch.zeros(graph_hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_id: torch.Tensor,
        style_vec: torch.Tensor,
        centrality_features: torch.Tensor,
        activity_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        graph_edge_index: torch.Tensor,
        role_to_node: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_text = self.text_enc(input_ids, attention_mask, role_id, style_vec)

        # Paper-aligned learnable node feature construction.
        graph_x = self.topology_feature_enc(
            centrality_features,
            activity_features,
            node_type_ids,
        )
        z_nodes = self.graph_enc(graph_x, graph_edge_index)

        node_idx = role_to_node[role_id]
        mask = node_idx >= 0
        safe_idx = node_idx.clamp(min=0)
        g_vec = z_nodes[safe_idx]
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
    centrality_features: torch.Tensor,
    activity_features: torch.Tensor,
    node_type_ids: torch.Tensor,
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

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        role_id = batch["role_id"].to(device)
        style_vec = batch["style_vec"].to(device)
        y = batch["label"].to(device)

        logits, _ = model(
            input_ids,
            attention_mask,
            role_id,
            style_vec,
            centrality_features,
            activity_features,
            node_type_ids,
            graph_edge_index,
            role_to_node,
        )
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


def prepare_data(args):
    """Legacy standalone data-preparation path.

    NOTE: the per-bot chronological 8/1/1 script does NOT call this function.
    That script builds its train/validation/test datasets itself and only reuses
    model/training utilities from this module.
    """
    df = pd.read_csv(args.csv_path)
    if args.text_col not in df.columns:
        raise ValueError(f"CSV 必须包含列 {args.text_col}")
    if args.role_col not in df.columns:
        raise ValueError(f"CSV 必须包含列 {args.role_col}（建议用 0..7 这种 role_id 与 nodes.csv 对齐）")
    if args.label_col not in df.columns:
        raise ValueError(f"CSV 必须包含标签列 {args.label_col}")

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    role_ids_all = torch.tensor(df[args.role_col].astype(int).values, dtype=torch.long)
    num_roles = int(role_ids_all.max().item() + 1)

    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for c in feat_cols:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少风格变量列：{c}")
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
    print(f"[INFO] Split fixed by split_seed={args.split_seed}: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

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

    ds_train = make_ds(idx_train)
    ds_val = make_ds(idx_val)
    ds_test = make_ds(idx_test)

    X = np.load("node_features_raw.npy")
    edge_index = np.load("edge_index.npy")
    N, graph_in_dim = X.shape

    if not os.path.exists(args.nodes_csv):
        raise FileNotFoundError(f"找不到 {args.nodes_csv}，用于 role_id -> node_index 对齐。")
    nodes = pd.read_csv(args.nodes_csv, encoding="gbk")
    if "role_id" not in nodes.columns:
        raise ValueError("nodes.csv 必须包含 role_id 列（ai 节点 0..7，用户节点 -1）")

    role_to_node = torch.full((num_roles,), -1, dtype=torch.long)
    for node_idx, rid in enumerate(nodes["role_id"].astype(int).values.tolist()):
        if 0 <= rid < num_roles:
            role_to_node[rid] = int(node_idx)

    return {
        "df": df,
        "feat_cols": feat_cols,
        "le": le,
        "scaler": scaler,
        "num_roles": num_roles,
        "num_classes": num_classes,
        "graph_in_dim": graph_in_dim,
        "role_to_node_cpu": role_to_node,
        "graph_x_np": X,
        "edge_index_np": edge_index,
        "adj": build_adj_list(torch.tensor(edge_index, dtype=torch.long), num_nodes=N),
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
    }


def get_arg(args, name: str, default):
    return getattr(args, name, default)


def make_model(args, bundle: Dict[str, Any]) -> EngagementE2EModel:
    return EngagementE2EModel(
        backbone=args.backbone,
        num_roles=bundle["num_roles"],
        style_in_dim=len(bundle["feat_cols"]),
        graph_in_dim=bundle["graph_in_dim"],
        num_classes=bundle["num_classes"],
        graph_encoder=args.graph_encoder,
        graph_hidden=args.graph_hidden,
        graph_layers=get_arg(args, "graph_layers", 2),
        att_heads=get_arg(args, "att_heads", 4),
        dropout=get_arg(args, "dropout", 0.1),
        text_cond_dim=get_arg(args, "text_cond_dim", 64),
        style_proj_dim=get_arg(args, "style_proj_dim", 64),
        film_hidden=get_arg(args, "film_hidden", 256),
        att_hidden=get_arg(args, "att_hidden", 256),
        pred_hidden=get_arg(args, "pred_hidden", 256),
    )


def default_checkpoint_path(args, seed: int) -> str:
    """Return a checkpoint path without assuming that a random split seed exists.

    When this module is imported by the per-bot chronological 8/1/1 trainer,
    train/val/test membership is deterministic and ``args.split_seed`` is absent.
    In that case the checkpoint is tagged ``chronological``.
    """
    explicit_path = get_arg(args, "checkpoint_path", "")
    if explicit_path:
        return explicit_path
    ckpt_dir = get_arg(args, "ckpt_dir", "ckpt")

    split_seed = get_arg(args, "split_seed", None)
    if split_seed is None:
        split_tag = get_arg(args, "split_tag", "chronological")
        suffix = str(split_tag)
    else:
        suffix = f"split{split_seed}"

    return os.path.join(
        ckpt_dir,
        f"best_maxlen{args.max_len}_gh{args.graph_hidden}_seed{seed}_{suffix}.pt",
    )


def checkpoint_payload(args, bundle: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "backbone": args.backbone,
        "num_roles": bundle["num_roles"],
        "num_classes": bundle["num_classes"],
        "label_classes": bundle["le"].classes_.tolist(),
        "feature_cols": bundle["feat_cols"],
        "graph_encoder": args.graph_encoder,
        "graph_hidden": args.graph_hidden,
        "graph_layers": get_arg(args, "graph_layers", 2),
        "att_heads": get_arg(args, "att_heads", 4),
        "dropout": get_arg(args, "dropout", 0.1),
        "text_cond_dim": get_arg(args, "text_cond_dim", 64),
        "style_proj_dim": get_arg(args, "style_proj_dim", 64),
        "film_hidden": get_arg(args, "film_hidden", 256),
        "att_hidden": get_arg(args, "att_hidden", 256),
        "pred_hidden": get_arg(args, "pred_hidden", 256),
        "role_col": args.role_col,
        "seed": seed,
        "split_seed": get_arg(args, "split_seed", None),
        "split_method": get_arg(args, "split_method", "unspecified"),
    }


def make_dataloaders(args, datasets, seed: int, include_test: bool = True):
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
    if not include_test:
        return dl_train, dl_val
    dl_test = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    return dl_train, dl_val, dl_test


def run_one_seed(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n{'='*20} Running seed={seed} {'='*20}")
    started_at = time.perf_counter()
    set_seed(seed, deterministic=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl_train, dl_val, dl_test = make_dataloaders(args, bundle["datasets"], seed)

    centrality_features = torch.tensor(
        bundle["centrality_features_np"], dtype=torch.float32, device=device
    )
    activity_features = torch.tensor(
        bundle["activity_features_np"], dtype=torch.float32, device=device
    )
    node_type_ids = torch.tensor(
        bundle["node_type_ids_np"], dtype=torch.long, device=device
    )
    graph_edge_index = torch.tensor(
        bundle["edge_index_np"], dtype=torch.long, device=device
    )
    role_to_node = bundle["role_to_node_cpu"].to(device)
    adj = bundle["adj"]
    rng = np.random.default_rng(seed)

    if not torch.isfinite(centrality_features).all():
        raise ValueError("centrality_features contain NaN/Inf before training.")
    if not torch.isfinite(activity_features).all():
        raise ValueError("activity_features contain NaN/Inf before training.")

    missing = torch.where(bundle["role_to_node_cpu"] < 0)[0].tolist()
    if missing:
        print(f"[WARN] role_id {missing} missing from graph; using g_null for those roles.")

    print(
        f"[INFO] topology inputs: centrality={tuple(centrality_features.shape)} "
        f"activity={tuple(activity_features.shape)} "
        f"node_types={tuple(node_type_ids.shape)} "
        f"edges={graph_edge_index.shape[1]}"
    )

    model = make_model(args, bundle).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(dl_train))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    global_step = 0
    no_improve = 0
    best_epoch = -1
    best_train_eval_loss = None
    best_train_metrics = None
    best_val_loss = None
    best_val_metrics = None
    ckpt_path = default_checkpoint_path(args, seed)

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0

        for batch_idx, batch in enumerate(dl_train):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            role_id = batch["role_id"].to(device)
            style_vec = batch["style_vec"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, z_nodes = model(
                input_ids,
                attention_mask,
                role_id,
                style_vec,
                centrality_features,
                activity_features,
                node_type_ids,
                graph_edge_index,
                role_to_node,
            )
            loss = criterion(logits, y)

            if args.lambda_unsup > 0 and (global_step % max(1, args.unsup_every) == 0):
                roots = role_to_node[torch.arange(bundle["num_roles"], device=device)]
                roots = roots[roots >= 0]
                if roots.numel() > 0:
                    pos_nodes = sample_rw_positives(
                        roots, adj, args.rw_len, args.rw_pos, rng
                    )
                    loss_unsup = unsup_rw_neg_sampling_loss(
                        z_nodes, roots, pos_nodes, args.rw_neg
                    )
                    loss = loss + args.lambda_unsup * loss_unsup

            # Do not allow one bad batch to corrupt all model parameters.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at seed={seed}, epoch={ep+1}, batch={batch_idx}. "
                    "Check topology/text intermediate tensors."
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at seed={seed}, epoch={ep+1}, "
                    f"batch={batch_idx}."
                )

            optimizer.step()
            scheduler.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n += bs
            global_step += 1

        train_loss = total_loss / max(1, n)
        train_eval_loss, train_metrics = evaluate(
            model, dl_train,
            centrality_features, activity_features, node_type_ids,
            graph_edge_index, role_to_node,
            device, criterion, average=args.metric_average,
        )
        val_loss, val_metrics = evaluate(
            model, dl_val,
            centrality_features, activity_features, node_type_ids,
            graph_edge_index, role_to_node,
            device, criterion, average=args.metric_average,
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
            f"val_F1={val_metrics['f1']:.4f} ({args.metric_average}) "
            f"val_MacroF1={val_metrics['macro_f1']:.4f}"
        )

        improved = val_metrics["macro_f1"] > (
            best_val_macro_f1 + args.early_min_delta
        )
        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = ep + 1
            best_train_eval_loss = float(train_eval_loss)
            best_train_metrics = dict(train_metrics)
            best_val_loss = float(val_loss)
            best_val_metrics = dict(val_metrics)
            no_improve = 0

            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            payload = checkpoint_payload(args, bundle, seed)
            payload.update({
                "model": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_metrics": best_val_metrics,
                "topology_feature_encoder": "c_v + a_v + t_v",
            })
            torch.save(payload, ckpt_path)
        else:
            no_improve += 1
            print(
                f"[EarlyStop] no_improve={no_improve}/{args.early_patience} "
                f"(best_macro_f1={best_val_macro_f1:.4f} @ epoch {best_epoch})"
            )
            if no_improve >= args.early_patience:
                print(
                    f"[EarlyStop] Stop at epoch {ep + 1}. "
                    f"Best epoch={best_epoch}, "
                    f"best_val_macro_f1={best_val_macro_f1:.4f}"
                )
                break

    if best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was saved.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    train_eval_loss, train_metrics = evaluate(
        model, dl_train,
        centrality_features, activity_features, node_type_ids,
        graph_edge_index, role_to_node,
        device, criterion, average=args.metric_average,
    )
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, dl_test,
        centrality_features, activity_features, node_type_ids,
        graph_edge_index, role_to_node,
        device, criterion, average=args.metric_average,
        return_preds=True,
    )

    print(
        f"[TRAIN seed={seed}] loss={train_eval_loss:.4f} | "
        f"acc={train_metrics['acc']:.4f} P={train_metrics['precision']:.4f} "
        f"R={train_metrics['recall']:.4f} F1={train_metrics['f1']:.4f}"
    )
    print(
        f"[TEST seed={seed}] loss={test_loss:.4f} | "
        f"acc={test_metrics['acc']:.4f} P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} "
        f"MacroF1={test_metrics['macro_f1']:.4f}"
    )

    if args.save_each_seed_cm:
        labels = np.arange(bundle["num_classes"])
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(
            y_true, y_pred, labels=labels, normalize="true"
        )
        np.savetxt(
            f"cm_test_counts_seed{seed}.csv", cm, fmt="%d", delimiter=","
        )
        np.savetxt(
            f"cm_test_norm_true_seed{seed}.csv",
            cm_norm, fmt="%.6f", delimiter=","
        )

        class_names = [str(i) for i in labels]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=class_names
        )
        disp.plot(values_format="d", cmap=plt.cm.Blues)
        for text in disp.text_.ravel():
            text.set_color("black")
            text.set_fontsize(11)
        plt.tight_layout()
        plt.savefig(f"cm_test_seed{seed}.png", dpi=300)
        plt.close()

        print(
            classification_report(
                y_true, y_pred, labels=labels,
                target_names=class_names, digits=4
            )
        )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_metrics["acc"]),
        "best_val_precision": float(best_val_metrics["precision"]),
        "best_val_recall": float(best_val_metrics["recall"]),
        "best_val_f1": float(best_val_metrics["f1"]),
        "best_val_macro_f1": float(best_val_macro_f1),
        "train_loss": float(train_eval_loss),
        "train_acc": float(train_metrics["acc"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_f1": float(train_metrics["f1"]),
        "train_macro_f1": float(train_metrics["macro_f1"]),
        "test_loss": float(test_loss),
        "test_acc": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "training_time": float(time.perf_counter() - started_at),
    }


def run_one_seed_validation_only(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    raise NotImplementedError(
        "This topology-aware companion trainer currently supports run_one_seed(). "
        "Adapt validation-only grid search to pass centrality/activity/node_type "
        "inputs before using it."
    )


def save_summary(results: List[Dict[str, Any]], out_csv: str):
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main():
    raise RuntimeError(
        "Run train_multiclass_engagement_e2e_5seeds_bot_disjoint_topology.py "
        "instead. This file is the topology-aware training backend."
    )


if __name__ == "__main__":
    main()