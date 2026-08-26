from collections import defaultdict
import torch
import numpy as np

@torch.no_grad()
def collect_dca_attention(
    model, dataloader,
    graph_x, graph_edge_index, role_to_node,
    device
):
    model.eval()

    bucket = defaultdict(list)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        role_id = batch["role_id"].to(device)
        style_vec = batch["style_vec"].to(device)
        labels = batch["label"].cpu().numpy()

        logits, _, attn = model(
            input_ids, attention_mask, role_id, style_vec,
            graph_x, graph_edge_index, role_to_node,
            return_attn=True
        )

        # 取 Text→Graph 注意力
        attn_t2g = attn["attn_t2g"]  # [B, H, L_text, 1]

        # 聚合方式（建议先这样，后面可以换）
        attn_score = attn_t2g.mean(dim=(1, 2, 3))  # [B]

        for a, y in zip(attn_score.cpu().numpy(), labels):
            bucket[int(y)].append(a)

    return bucket
def summarize_by_label(attn_bucket):
    stats = {}
    for label, vals in attn_bucket.items():
        v = np.array(vals)
        stats[label] = {
            "mean": v.mean(),
            "std": v.std(),
            "median": np.median(v),
            "n": len(v)
        }
    return stats


