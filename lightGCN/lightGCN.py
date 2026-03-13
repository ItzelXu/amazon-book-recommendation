"""
- k-core (item_min=5)
- warm test split (per-user stratified)
- heavy-user handling (optional cap or per-row norm style through sampling)
- NDCG@20 evaluation
- simple grid search over LightGCN hyperparameters
"""

import os
import sys
import math
import random
import logging
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("LightGCN_pipeline")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# =========================
# Utilities (same style as EASE)
# =========================

def load_train_txt_as_dataframe(
    path: str,
    user_col: str = "user",
    item_col: str = "item",
) -> pd.DataFrame:
    """
    Each line: user_id item_1 item_2 ... item_n
    -> one row per (user,item) pair, deduped within user.
    """
    users: List[int] = []
    items: List[int] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            seen = set()
            for it in parts[1:]:
                it_int = int(it)
                if it_int in seen:
                    continue
                seen.add(it_int)
                users.append(u)
                items.append(it_int)
    return pd.DataFrame({user_col: users, item_col: items})


def kcore_filter(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    user_min: int = 1,
    item_min: int = 3,
) -> pd.DataFrame:
    """
    Iterative k-core filter. Ensure every kept item appears >= item_min times.
    """
    cur = df.copy()
    changed = True
    while changed:
        changed = False
        if user_min > 1:
            uc = cur[user_col].value_counts()
            good_users = set(uc[uc >= user_min].index)
            new = cur[cur[user_col].isin(good_users)]
            if len(new) != len(cur):
                cur = new
                changed = True
        ic = cur[item_col].value_counts()
        good_items = set(ic[ic >= item_min].index) if item_min > 1 else set(ic.index)
        new = cur[cur[item_col].isin(good_items)]
        if len(new) != len(cur):
            cur = new
            changed = True
    return cur


def stratified_user_split(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    eval_ratio: float = 0.2,
    seed: int = 42,
    warm_eval: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user stratified split into TRAIN and EVAL.

    - shuffle each user's items
    - take ~eval_ratio as eval, at least 1 if user has >=2 items
    - warm_eval: eval items must appear >=2 times globally if possible
    """
    rng = random.Random(seed)
    global_counts = df[item_col].value_counts().to_dict()

    train_rows: List[Tuple[int, int]] = []
    eval_rows: List[Tuple[int, int]] = []

    grouped = df.groupby(user_col)[item_col].apply(list)
    for u, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n == 0:
            continue

        if n == 1:
            eval_size = 0
        else:
            eval_size = int(round(n * eval_ratio))
            eval_size = max(1, eval_size)
            eval_size = min(eval_size, n - 1)

        if eval_size == 0:
            train_rows.extend((u, i) for i in items)
            continue

        if warm_eval:
            warm_items = [i for i in items if global_counts.get(i, 0) > 1]
        else:
            warm_items = items

        if len(warm_items) < eval_size:
            candidates = items
        else:
            candidates = warm_items

        chosen_eval = set(candidates[:eval_size])
        train_items = [i for i in items if i not in chosen_eval]
        eval_items = list(chosen_eval)

        if not train_items and eval_items:
            train_items.append(eval_items.pop())

        train_rows.extend((u, i) for i in train_items)
        eval_rows.extend((u, i) for i in eval_items)

    train_df = pd.DataFrame(train_rows, columns=[user_col, item_col])
    eval_df = pd.DataFrame(eval_rows, columns=[user_col, item_col])
    return train_df, eval_df


def cap_heavy_users(df, user_col="user", item_col="item", cap=300, seed=42):
    rng = np.random.RandomState(seed)
    parts = []
    for u, grp in df.groupby(user_col):
        if len(grp) <= cap:
            parts.append(grp)
        else:
            parts.append(grp.sample(n=cap, random_state=rng.randint(1e9)))
    return pd.concat(parts, axis=0, ignore_index=True)


def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int = 20) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in gt:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gt))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# =========================
# Simple implicit CF data model for LightGCN
# =========================

class ImplicitCFData:
    """
    Minimal data wrapper:
    - maps ext user/item IDs to 0..U-1 / 0..I-1
    - stores positive items per user
    - builds normalized adjacency A (user+item graph) for LightGCN
    - provides BPR sampling
    """

    def __init__(self, df: pd.DataFrame, user_col: str = "user", item_col: str = "item"):
        self.user_col = user_col
        self.item_col = item_col

        users = df[user_col].values
        items = df[item_col].values

        self.user2id: Dict[int, int] = {}
        self.id2user: List[int] = []
        self.item2id: Dict[int, int] = {}
        self.id2item: List[int] = []

        for u in users:
            if u not in self.user2id:
                self.user2id[u] = len(self.id2user)
                self.id2user.append(u)
        for i in items:
            if i not in self.item2id:
                self.item2id[i] = len(self.id2item)
                self.id2item.append(i)

        self.n_users = len(self.id2user)
        self.n_items = len(self.id2item)
        logger.info(f"LightGCN data: {self.n_users} users, {self.n_items} items")

        # positive items per user (internal ids)
        self.user_pos_items: List[List[int]] = [[] for _ in range(self.n_users)]
        for u_ext, i_ext in zip(users, items):
            u = self.user2id[u_ext]
            i = self.item2id[i_ext]
            self.user_pos_items[u].append(i)

        # dedupe
        for u in range(self.n_users):
            self.user_pos_items[u] = list(sorted(set(self.user_pos_items[u])))

        # build normalized adjacency
        self.A_norm = self._build_normalized_adj()

    def _build_normalized_adj(self) -> torch.sparse.FloatTensor:
        """
        Build LightGCN normalized adjacency matrix A_hat (symmetric, no self-loop).
        Size: (n_users + n_items) x (n_users + n_items)
        """
        n_nodes = self.n_users + self.n_items
        rows: List[int] = []
        cols: List[int] = []

        for u in range(self.n_users):
            for i in self.user_pos_items[u]:
                ui = self.n_users + i
                # undirected bipartite edge
                rows.append(u)
                cols.append(ui)
                rows.append(ui)
                cols.append(u)

        if not rows:
            raise ValueError("No interactions to build adjacency matrix.")

        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        data = np.ones_like(rows, dtype=np.float32)

        # degree
        deg = np.bincount(rows, minlength=n_nodes).astype(np.float32)
        # because graph is symmetric, deg from rows=deg from cols

        # compute normalized edge weights: 1 / sqrt(d_i * d_j)
        norm = 1.0 / np.sqrt(deg[rows] * deg[cols])
        values = data * norm

        indices = torch.from_numpy(np.vstack([rows, cols]))
        values_t = torch.from_numpy(values)
        A_norm = torch.sparse_coo_tensor(
            indices, values_t, size=(n_nodes, n_nodes), dtype=torch.float32
        ).coalesce()
        return A_norm

    def sample_bpr_batch(
        self,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample BPR (user, pos_item, neg_item) triples with replacement.
        """
        users = rng.randint(0, self.n_users, size=batch_size)
        pos_items = np.zeros(batch_size, dtype=np.int64)
        neg_items = np.zeros(batch_size, dtype=np.int64)

        for idx, u in enumerate(users):
            pos_list = self.user_pos_items[u]
            pos_items[idx] = pos_list[rng.randint(len(pos_list))]

            while True:
                j = rng.randint(0, self.n_items)
                if j not in pos_list:
                    neg_items[idx] = j
                    break

        return users, pos_items, neg_items


# =========================
# LightGCN model (PyTorch)
# =========================

class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, A_norm: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate user+item embeddings through A_norm n_layers times,
        then average layer-wise embeddings.
        """
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(A_norm, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1).mean(dim=1)
        user_final, item_final = torch.split(embs, [self.n_users, self.n_items], dim=0)
        return user_final, item_final


def bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    l2_reg: float,
) -> torch.Tensor:
    """
    BPR loss with L2 regularization on embeddings.
    """
    u_e = user_emb[users]          # (B, D)
    pos_e = item_emb[pos_items]    # (B, D)
    neg_e = item_emb[neg_items]    # (B, D)

    pos_scores = torch.sum(u_e * pos_e, dim=1)
    neg_scores = torch.sum(u_e * neg_e, dim=1)
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

    reg = (u_e.norm(2).pow(2) +
           pos_e.norm(2).pow(2) +
           neg_e.norm(2).pow(2)) / users.shape[0]
    return loss + l2_reg * reg


# =========================
# Evaluation helpers
# =========================

def build_eval_ground_truth(
    eval_df: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> Dict[int, List[int]]:
    """
    Returns dict: external_user_id -> list of external item ids in eval.
    """
    return eval_df.groupby(user_col)[item_col].apply(list).to_dict()


def recommend_for_all_users(
    model: LightGCN,
    data: ImplicitCFData,
    A_norm: torch.sparse.FloatTensor,
    k: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, List[int]]:
    """
    Return top-k recommendations for all users (external IDs).
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(A_norm.to(device))
        scores = torch.matmul(user_emb, item_emb.t())  # (U, I)

        # mask seen items
        for u in range(data.n_users):
            pos_items = data.user_pos_items[u]
            if pos_items:
                scores[u, pos_items] = -1e9

        # top-k per user
        topk_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        id2item = np.array(data.id2item)
        recs: Dict[int, List[int]] = {}
        for u_int in range(data.n_users):
            u_ext = data.id2user[u_int]
            ext_items = id2item[topk_items[u_int]].tolist()
            recs[u_ext] = ext_items
    return recs


def evaluate_ndcg_at20(
    recs: Dict[int, List[int]],
    eval_gt: Dict[int, List[int]],
) -> float:
    scores = []
    for u, gt_items in eval_gt.items():
        if u not in recs:
            continue
        scores.append(ndcg_at_k(recs[u], gt_items, k=20))
    return float(np.mean(scores)) if scores else 0.0


# =========================
# Training + grid search
# =========================

def train_and_eval_lightgcn(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    embed_dim: int,
    n_layers: int,
    lr: float,
    l2_reg: float,
    epochs: int,
    batch_size: int,
    seed: int = 42,
    device: Optional[torch.device] = None,
    eval_every: int = 1,   # how often to log eval, 1 = every epoch
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = ImplicitCFData(train_df, "user", "item")
    A_norm = data.A_norm.to(device)

    model = LightGCN(data.n_users, data.n_items, embed_dim=embed_dim, n_layers=n_layers)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    rng = np.random.RandomState(seed)
    n_interactions = sum(len(v) for v in data.user_pos_items)
    steps_per_epoch = max(1, n_interactions // batch_size)

    # ground truth for eval
    eval_gt = build_eval_ground_truth(eval_df, "user", "item")

    epoch_losses = []
    val_ndcgs = []   # <-- new: validation NDCG per eval step

    for epoch in range(1, epochs + 1):
        # --------- TRAIN ---------
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            users, pos_items, neg_items = data.sample_bpr_batch(batch_size, rng)
            users_t = torch.from_numpy(users).long().to(device)
            pos_t = torch.from_numpy(pos_items).long().to(device)
            neg_t = torch.from_numpy(neg_items).long().to(device)

            user_emb, item_emb = model(A_norm)
            loss = bpr_loss(user_emb, item_emb, users_t, pos_t, neg_t, l2_reg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        epoch_losses.append(epoch_loss)

        # --------- EVAL ---------
        val_ndcg_str = ""
        if (epoch % eval_every == 0) or (epoch == epochs):
            model.eval()
            with torch.no_grad():
                recs = recommend_for_all_users(model, data, A_norm, k=20, device=device)
                val_ndcg = evaluate_ndcg_at20(recs, eval_gt)
            val_ndcgs.append((epoch, val_ndcg))
            val_ndcg_str = f", val NDCG@20 = {val_ndcg:.5f}"

        # log both train and val
        logger.info(
            f"LightGCN epoch {epoch}/{epochs}, embed={embed_dim}, layers={n_layers}, "
            f"lr={lr}, l2={l2_reg}, train loss = {epoch_loss:.4f}{val_ndcg_str}"
        )

    # final NDCG (last eval value if we logged every epoch)
    if val_ndcgs:
        final_ndcg = val_ndcgs[-1][1]
    else:
        # fallback: compute once if eval_every was very large
        model.eval()
        with torch.no_grad():
            recs = recommend_for_all_users(model, data, A_norm, k=20, device=device)
            final_ndcg = evaluate_ndcg_at20(recs, eval_gt)

    logger.info(
        f"Final NDCG@20 for embed={embed_dim}, layers={n_layers}, "
        f"lr={lr}, l2={l2_reg} -> {final_ndcg:.5f}"
    )

    return final_ndcg, epoch_losses, val_ndcgs


def run_pipeline():
    # ----------------- CONFIG (fixed in code, like EASE.py) -----------------
    train_path = "../train-1.txt"

    user_min = 1
    item_min = 5
    eval_ratio = 0.2
    warm_eval = True
    cap_per_user = 300

    seed = 42
    epochs = 60
    batch_size = 2048

    # simple hyperparam grid (analogous to λ grid for EASE)
    # embed_dims = [32, 64]
    # n_layers_list = [2, 3]
    # lrs = [1e-3, 5e-4]
    # l2_regs = [1e-4, 5e-4, 1e-3]

    embed_dims = [64]
    n_layers_list = [2]
    lrs = [1e-3]
    l2_regs = [5e-4]

    # -----------------------------------------------------------------------
    logger.info(f"Loading data from {train_path}")
    raw_df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    logger.info(f"Applying k-core filter: user_min={user_min}, item_min={item_min}")
    df = kcore_filter(raw_df, "user", "item", user_min=user_min, item_min=item_min)

    logger.info(
        f"Stratified per-user split (eval_ratio={eval_ratio}, warm_eval={warm_eval})"
    )
    train_df, eval_df = stratified_user_split(
        df, "user", "item", eval_ratio=eval_ratio, seed=seed, warm_eval=warm_eval
    )

    if cap_per_user > 0:
        logger.info(f"Capping heavy users in TRAIN to ≤{cap_per_user} interactions")
        train_df = cap_heavy_users(train_df, "user", "item", cap=cap_per_user, seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    best_cfg = None
    best_ndcg = -1.0
    history: List[Tuple[Tuple[int, int, float, float], float]] = []
    loss_records = []

    for ed in embed_dims:
        for nl in n_layers_list:
            for lr in lrs:
                for l2 in l2_regs:
                    logger.info(
                        f"=== Hyperparams: embed_dim={ed}, n_layers={nl}, lr={lr}, l2={l2} ==="
                    )
                    ndcg, epoch_losses, val_ndcgs = train_and_eval_lightgcn(
                        train_df, eval_df,
                        embed_dim=ed,
                        n_layers=nl,
                        lr=lr,
                        l2_reg=l2,
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed,
                        device=device,
                        eval_every=5,  # val every 5 epoch
                    )
                    history.append(((ed, nl, lr, l2), ndcg))

                    # record per-epoch loss for plotting
                    for epoch_idx, loss in enumerate(epoch_losses, start=1):
                        loss_records.append({
                            "embed_dim": ed,
                            "n_layers": nl,
                            "lr": lr,
                            "l2": l2,
                            "epoch": epoch_idx,
                            "loss": loss,
                        })

                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_cfg = (ed, nl, lr, l2)

    logger.info("=== Grid search finished ===")
    logger.info(f"Best config: embed_dim={best_cfg[0]}, n_layers={best_cfg[1]}, "
                f"lr={best_cfg[2]}, l2={best_cfg[3]} with NDCG@20={best_ndcg:.5f}")


    # (Optional) save history to CSV for plotting
    out_hist = "lightgcn_grid_history.csv"
    rows = []
    for (ed, nl, lr, l2), ndcg in history:
        rows.append(
            {"embed_dim": ed, "n_layers": nl, "lr": lr, "l2": l2, "ndcg20": ndcg}
        )
    pd.DataFrame(rows).to_csv(out_hist, index=False)
    logger.info(f"Saved grid search history to {out_hist}")
    out_loss = "../lightgcn_loss_curves.csv"
    pd.DataFrame(loss_records).to_csv(out_loss, index=False)
    logger.info(f"Saved per-epoch loss curves to {out_loss}")


if __name__ == "__main__":
    run_pipeline()
