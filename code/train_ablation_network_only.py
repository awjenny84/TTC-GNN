import os
import json
import csv
import random
import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import get_linear_schedule_with_warmup
from torch_geometric.nn import GATConv, GraphConv, SAGEConv
from topology_feature_encoder import TopologyFeatureEncoder
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
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
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
    B, _ = pos_nodes.shape
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
# 1) Dataset: graph-only
# -------------------------

class GraphOnlyPostDataset(Dataset):
    """Each post contributes only (bot identity -> graph node, label).

    No text tokenization, persona description, or style vector is constructed.
    The post's bot identity is used solely as a lookup key for the corresponding
    graph node embedding.
    """

    def __init__(self, role_ids: torch.Tensor, labels: torch.Tensor):
        if len(role_ids) != len(labels):
            raise ValueError("role_ids and labels must have the same length.")
        self.role_ids = role_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "role_id": self.role_ids[idx],
            "label": self.labels[idx],
        }


# Backward-compatible alias for any outer script that still imports this name.
WeiboTensorDataset = GraphOnlyPostDataset


# -------------------------
# 2) Graph encoder
# -------------------------

def normalize_graph_encoder_name(name: str) -> str:
    value = str(name or "gn").strip().lower()
    aliases = {
        "graph": "gn",
        "graphconv": "gn",
        "graph_network": "gn",
        "graph-network": "gn",
        "graphsage": "sage",
    }
    value = aliases.get(value, value)
    if value not in {"gn", "gat", "sage"}:
        raise ValueError("--graph_encoder must be one of: gn, gat, sage.")
    return value


def graph_encoder_display_name(name: str) -> str:
    value = normalize_graph_encoder_name(name)
    if value == "gn":
        return "GN(GraphConv)"
    if value == "gat":
        return "GAT"
    return "GraphSAGE"


class GraphNetworkEncoder(nn.Module):
    """Stacked GraphConv encoder used for the GN option."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("graph_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Stacked GAT encoder with concat=False to keep graph_hidden fixed."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("graph_layers must be >= 1")
        if heads < 1:
            raise ValueError("gat_heads must be >= 1")
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("graph_layers must be >= 1")
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


def make_graph_encoder(
    graph_encoder: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    gat_heads: int,
) -> nn.Module:
    encoder = normalize_graph_encoder_name(graph_encoder)
    if encoder == "gn":
        return GraphNetworkEncoder(
            in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
        )
    if encoder == "gat":
        return GATEncoder(
            in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=gat_heads,
            dropout=dropout,
        )
    return GraphSAGEEncoder(
        in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
    )


# -------------------------
# 3) NETWORK-ONLY model
# -------------------------

class GraphOnlyDualCoAttentionClassifier(nn.Module):
    """Dual co-attention classifier for graph-only inputs.

    The anchor sequence is the target bot node embedding. The context sequence
    is a fixed-size graph neighborhood for that bot, prepared by the outer
    script. This keeps the DCA-style bidirectional attention head without
    introducing a text branch into the graph-only experiment.
    """

    def __init__(
        self,
        dim_graph: int,
        num_classes: int,
        d_att: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
        pred_hidden: int = 256,
    ):
        super().__init__()
        if d_att % nhead != 0:
            raise ValueError("dca_dim must be divisible by dca_heads.")

        self.q_anchor = nn.Linear(dim_graph, d_att)
        self.k_context = nn.Linear(dim_graph, d_att)
        self.v_context = nn.Linear(dim_graph, d_att)

        self.q_context = nn.Linear(dim_graph, d_att)
        self.k_anchor = nn.Linear(dim_graph, d_att)
        self.v_anchor = nn.Linear(dim_graph, d_att)

        self.att_anchor_to_context = nn.MultiheadAttention(
            d_att, nhead, dropout=dropout, batch_first=True
        )
        self.att_context_to_anchor = nn.MultiheadAttention(
            d_att, nhead, dropout=dropout, batch_first=True
        )
        self.ln_anchor = nn.LayerNorm(d_att)
        self.ln_context = nn.LayerNorm(d_att)

        self.pred = nn.Sequential(
            nn.Linear(2 * d_att, pred_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, num_classes),
        )

    def forward(
        self,
        g_vec: torch.Tensor,
        context_vec: torch.Tensor,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        anchor = g_vec.unsqueeze(1)
        if context_vec.dim() == 2:
            context_vec = context_vec.unsqueeze(1)

        key_padding_mask = None
        if context_mask is not None:
            context_mask = context_mask.bool()
            key_padding_mask = ~context_mask
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        anchor_context, _ = self.att_anchor_to_context(
            self.q_anchor(anchor),
            self.k_context(context_vec),
            self.v_context(context_vec),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        anchor_context = self.ln_anchor(anchor_context)

        context_anchor, _ = self.att_context_to_anchor(
            self.q_context(context_vec),
            self.k_anchor(anchor),
            self.v_anchor(anchor),
            need_weights=False,
        )
        context_anchor = self.ln_context(context_anchor)

        if context_mask is None:
            context_summary = context_anchor.mean(dim=1)
        else:
            weights = context_mask.to(context_anchor.dtype).unsqueeze(-1)
            context_summary = (context_anchor * weights).sum(dim=1)
            context_summary = context_summary / weights.sum(dim=1).clamp_min(1.0)

        fused = torch.cat([anchor_context[:, 0, :], context_summary], dim=-1)
        return self.pred(fused)


class EngagementE2EModel(nn.Module):
    """Topology-only ablation.

    Pipeline:
        topology_feature_encoder (c_v + a_v + t_v)
        -> selectable GN/GAT graph encoder
        -> graph-context DCA classifier using the corresponding bot node
           embedding and its local graph context
    """

    def __init__(
        self,
        num_classes: int,
        graph_hidden: int = 64,
        graph_layers: int = 2,
        graph_encoder: str = "gn",
        gat_heads: int = 4,
        dropout: float = 0.1,
        pred_hidden: int = 256,
        dca_dim: int = 256,
        dca_heads: int = 4,
    ):
        super().__init__()
        self.graph_encoder_name = normalize_graph_encoder_name(graph_encoder)

        self.topology_feature_enc = TopologyFeatureEncoder(
            graph_dim=graph_hidden,
            dropout=dropout,
        )
        self.graph_enc = make_graph_encoder(
            self.graph_encoder_name,
            in_dim=graph_hidden,
            hidden_dim=graph_hidden,
            num_layers=graph_layers,
            dropout=dropout,
            gat_heads=gat_heads,
        )
        self.dca_classifier = GraphOnlyDualCoAttentionClassifier(
            dim_graph=graph_hidden,
            num_classes=num_classes,
            d_att=dca_dim,
            nhead=dca_heads,
            dropout=dropout,
            pred_hidden=pred_hidden,
        )
        self.g_null = nn.Parameter(torch.zeros(graph_hidden))

    def encode_graph(
        self,
        centrality_features: torch.Tensor,
        activity_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        graph_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        graph_x = self.topology_feature_enc(
            centrality_features,
            activity_features,
            node_type_ids,
        )
        return self.graph_enc(graph_x, graph_edge_index)

    def forward(
        self,
        role_id: torch.Tensor,
        centrality_features: torch.Tensor,
        activity_features: torch.Tensor,
        node_type_ids: torch.Tensor,
        graph_edge_index: torch.Tensor,
        role_to_node: torch.Tensor,
        role_to_dca_context: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_nodes = self.encode_graph(
            centrality_features,
            activity_features,
            node_type_ids,
            graph_edge_index,
        )

        node_idx = role_to_node[role_id]
        mask = node_idx >= 0
        safe_idx = node_idx.clamp(min=0)
        g_vec = z_nodes[safe_idx]
        g_vec = torch.where(
            mask.unsqueeze(1),
            g_vec,
            self.g_null.unsqueeze(0).expand_as(g_vec),
        )

        if role_to_dca_context is None:
            context_vec = g_vec.unsqueeze(1)
            context_mask = torch.ones(
                (g_vec.size(0), 1), dtype=torch.bool, device=g_vec.device
            )
        else:
            context_idx = role_to_dca_context[role_id]
            context_mask = context_idx >= 0
            safe_context_idx = context_idx.clamp(min=0)
            context_vec = z_nodes[safe_context_idx]
            context_vec = torch.where(
                context_mask.unsqueeze(-1),
                context_vec,
                torch.zeros_like(context_vec),
            )

        logits = self.dca_classifier(g_vec, context_vec, context_mask)
        return logits, z_nodes


# -------------------------
# 4) Train / Eval
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
    role_to_dca_context: torch.Tensor,
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
        role_id = batch["role_id"].to(device)
        y = batch["label"].to(device)

        logits, _ = model(
            role_id,
            centrality_features,
            activity_features,
            node_type_ids,
            graph_edge_index,
            role_to_node,
            role_to_dca_context,
        )
        loss = criterion(logits, y)

        bs = y.size(0)
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
    raise RuntimeError(
        "This NETWORK-ONLY module is a training backend. "
        "Use train_graph_only_engagement.py to construct the per-bot "
        "chronological 8/1/1 data bundle."
    )


def get_arg(args, name: str, default):
    return getattr(args, name, default)


def make_model(args, bundle: Dict[str, Any]) -> EngagementE2EModel:
    return EngagementE2EModel(
        num_classes=bundle["num_classes"],
        graph_hidden=args.graph_hidden,
        graph_layers=get_arg(args, "graph_layers", 2),
        graph_encoder=get_arg(args, "graph_encoder", "gn"),
        gat_heads=get_arg(args, "gat_heads", 4),
        dropout=get_arg(args, "dropout", 0.1),
        pred_hidden=get_arg(args, "pred_hidden", 256),
        dca_dim=get_arg(args, "dca_dim", 256),
        dca_heads=get_arg(args, "dca_heads", 4),
    )


def default_checkpoint_path(args, seed: int) -> str:
    explicit_path = get_arg(args, "checkpoint_path", "")
    if explicit_path:
        return explicit_path
    ckpt_dir = get_arg(args, "ckpt_dir", "ckpt")
    split_tag = get_arg(args, "split_tag", "chronological_811_dca")
    graph_encoder = normalize_graph_encoder_name(get_arg(args, "graph_encoder", "gn")).upper()
    return os.path.join(
        ckpt_dir,
        f"best_NETWORK_ONLY_{graph_encoder}_DCA_gh{args.graph_hidden}_seed{seed}_{split_tag}.pt",
    )


def checkpoint_payload(args, bundle: Dict[str, Any], seed: int) -> Dict[str, Any]:
    graph_encoder = normalize_graph_encoder_name(get_arg(args, "graph_encoder", "gn"))
    return {
        "model_variant": "network_only_{}_dca".format(graph_encoder),
        "uses_text": False,
        "uses_persona_text_conditioning": False,
        "uses_style": False,
        "uses_topology": True,
        "uses_dca": True,
        "num_roles": bundle["num_roles"],
        "num_classes": bundle["num_classes"],
        "label_classes": bundle["le"].classes_.tolist(),
        "graph_encoder": graph_encoder_display_name(graph_encoder),
        "graph_hidden": args.graph_hidden,
        "graph_layers": get_arg(args, "graph_layers", 2),
        "gat_heads": get_arg(args, "gat_heads", 4),
        "dropout": get_arg(args, "dropout", 0.1),
        "pred_hidden": get_arg(args, "pred_hidden", 256),
        "dca_dim": get_arg(args, "dca_dim", 256),
        "dca_heads": get_arg(args, "dca_heads", 4),
        "dca_context_size": get_arg(args, "dca_context_size", None),
        "seed": seed,
        "split_method": get_arg(
            args,
            "split_method",
            "per-bot chronological post-level 80/10/10",
        ),
        "time_col": get_arg(args, "time_col", None),
        "topology_feature_encoder": "c_v + a_v + t_v",
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
    dl_val = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    if not include_test:
        return dl_train, dl_val
    dl_test = DataLoader(
        datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    return dl_train, dl_val, dl_test


def run_one_seed(args, seed: int, bundle: Dict[str, Any]) -> Dict[str, Any]:
    graph_encoder = normalize_graph_encoder_name(get_arg(args, "graph_encoder", "gn"))
    graph_encoder_label = graph_encoder_display_name(graph_encoder)
    print(f"\n{'='*20} Running NETWORK-ONLY {graph_encoder_label}+DCA seed={seed} {'='*20}")
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
    role_to_dca_context_cpu = bundle.get("role_to_dca_context_cpu")
    role_to_dca_context = (
        role_to_dca_context_cpu.to(device)
        if role_to_dca_context_cpu is not None
        else None
    )
    adj = bundle["adj"]
    rng = np.random.default_rng(seed)

    if not torch.isfinite(centrality_features).all():
        raise ValueError("centrality_features contain NaN/Inf before training.")
    if not torch.isfinite(activity_features).all():
        raise ValueError("activity_features contain NaN/Inf before training.")

    missing = torch.where(bundle["role_to_node_cpu"] < 0)[0].tolist()
    if missing:
        raise ValueError(
            f"Some bot role IDs cannot be aligned to graph nodes: {missing}. "
            "Graph-only training requires every target bot to exist in the graph."
        )

    print(
        f"[INFO] MODEL = TopologyFeatureEncoder + {graph_encoder_label} + graph-context DCA | "
        f"centrality={tuple(centrality_features.shape)} "
        f"activity={tuple(activity_features.shape)} "
        f"node_types={tuple(node_type_ids.shape)} "
        f"edges={graph_edge_index.shape[1]} | "
        f"gat_heads={get_arg(args, 'gat_heads', 4) if graph_encoder == 'gat' else 'n/a'} | "
        f"dca_context={tuple(role_to_dca_context.shape) if role_to_dca_context is not None else None} | "
        "Text/Persona/Style disabled."
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
    best_val_loss = None
    best_val_metrics = None
    ckpt_path = default_checkpoint_path(args, seed)

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0

        for batch_idx, batch in enumerate(dl_train):
            role_id = batch["role_id"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, z_nodes = model(
                role_id,
                centrality_features,
                activity_features,
                node_type_ids,
                graph_edge_index,
                role_to_node,
                role_to_dca_context,
            )
            loss = criterion(logits, y)

            if get_arg(args, "lambda_unsup", 0.0) > 0 and (
                global_step % max(1, get_arg(args, "unsup_every", 1)) == 0
            ):
                roots = role_to_node[torch.arange(bundle["num_roles"], device=device)]
                roots = roots[roots >= 0]
                if roots.numel() > 0:
                    pos_nodes = sample_rw_positives(
                        roots,
                        adj,
                        get_arg(args, "rw_len", 5),
                        get_arg(args, "rw_pos", 5),
                        rng,
                    )
                    loss_unsup = unsup_rw_neg_sampling_loss(
                        z_nodes, roots, pos_nodes, get_arg(args, "rw_neg", 10)
                    )
                    loss = loss + get_arg(args, "lambda_unsup", 0.0) * loss_unsup

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at seed={seed}, epoch={ep+1}, batch={batch_idx}."
                )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at seed={seed}, epoch={ep+1}, batch={batch_idx}."
                )

            optimizer.step()
            scheduler.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs
            global_step += 1

        train_loss = total_loss / max(1, n)
        train_eval_loss, train_metrics = evaluate(
            model,
            dl_train,
            centrality_features,
            activity_features,
            node_type_ids,
            graph_edge_index,
            role_to_node,
            role_to_dca_context,
            device,
            criterion,
            average=args.metric_average,
        )
        val_loss, val_metrics = evaluate(
            model,
            dl_val,
            centrality_features,
            activity_features,
            node_type_ids,
            graph_edge_index,
            role_to_node,
            role_to_dca_context,
            device,
            criterion,
            average=args.metric_average,
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
            best_val_loss = float(val_loss)
            best_val_metrics = dict(val_metrics)
            no_improve = 0

            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            payload = checkpoint_payload(args, bundle, seed)
            payload.update(
                {
                    "model": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_metrics": best_val_metrics,
                    "ablation": "Network only: TopologyFeatureEncoder + {} + graph-context DCA".format(
                        graph_encoder_label
                    ),
                }
            )
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
                    f"Best epoch={best_epoch}, best_val_macro_f1={best_val_macro_f1:.4f}"
                )
                break

    if best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was saved.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    train_eval_loss, train_metrics = evaluate(
        model,
        dl_train,
        centrality_features,
        activity_features,
        node_type_ids,
        graph_edge_index,
        role_to_node,
        role_to_dca_context,
        device,
        criterion,
        average=args.metric_average,
    )
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model,
        dl_test,
        centrality_features,
        activity_features,
        node_type_ids,
        graph_edge_index,
        role_to_node,
        role_to_dca_context,
        device,
        criterion,
        average=args.metric_average,
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
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        np.savetxt(f"cm_test_counts_seed{seed}.csv", cm, fmt="%d", delimiter=",")
        np.savetxt(
            f"cm_test_norm_true_seed{seed}.csv", cm_norm, fmt="%.6f", delimiter=","
        )

        class_names = bundle["le"].classes_.astype(str).tolist()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(values_format="d", cmap=plt.cm.Blues)
        for text in disp.text_.ravel():
            text.set_color("black")
            text.set_fontsize(11)
        plt.tight_layout()
        plt.savefig(f"cm_test_seed{seed}.png", dpi=300)
        plt.close()

        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=class_names,
                digits=4,
                zero_division=0,
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


def save_summary(results: List[Dict[str, Any]], out_csv: str):
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main():
    raise RuntimeError(
        "This is the NETWORK-ONLY graph backend. "
        "Run/import it through train_graph_only_engagement.py."
    )


if __name__ == "__main__":
    main()
