"""
- Trains BPR-MF on FULL train-1.txt
- Uses best hyperparameters (fill in from BPRMF_pipeline tuning)
- Writes top-20 item recommendations per user (no user IDs)
"""

import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import math

logger = logging.getLogger("BPRMF_submit")
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# =========================
# Utilities
# =========================

def load_train_txt_as_dataframe(
    path: str,
    user_col: str = "user",
    item_col: str = "item",
) -> pd.DataFrame:
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


def save_recommendations(recs: np.ndarray, path: str) -> None:
    """
    Same format as EASE_submission: no user IDs, one line per user.
    """
    num_users, k = recs.shape
    with open(path, "w") as f:
        for u in range(num_users):
            items_str = " ".join(str(int(it)) for it in recs[u])
            f.write(items_str + "\n")


# =========================
# BPR-MF model (simplified)
# =========================

class BPRMF:
    def __init__(
        self,
        n_factors: int = 256,
        lr: float = 1e-3,
        reg: float = 1e-4,
        n_epochs: int = 40,
        n_neg: int = 1,
        use_idf_weight: bool = True,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_neg = n_neg
        self.use_idf_weight = use_idf_weight
        self.seed = seed

        self.user2idx: Dict[int, int] = {}
        self.idx2user: np.ndarray = np.array([], dtype=np.int64)
        self.item2idx: Dict[int, int] = {}
        self.idx2item: np.ndarray = np.array([], dtype=np.int64)

        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.user_pos: Optional[List[np.ndarray]] = None
        self.user_pos_sets: Optional[List[set]] = None
        self.user_pos_probs: Optional[List[Optional[np.ndarray]]] = None
        self.popular_items: Optional[np.ndarray] = None

    def _build_mappings(self, df: pd.DataFrame, user_col: str, item_col: str):
        users = df[user_col].values
        items = df[item_col].values
        user2idx: Dict[int, int] = {}
        item2idx: Dict[int, int] = {}
        user_list: List[int] = []
        item_list: List[int] = []
        for u in users:
            if u not in user2idx:
                user2idx[u] = len(user_list)
                user_list.append(u)
        for it in items:
            if it not in item2idx:
                item2idx[it] = len(item_list)
                item_list.append(it)

        self.user2idx = user2idx
        self.idx2user = np.array(user_list, dtype=np.int64)
        self.item2idx = item2idx
        self.idx2item = np.array(item_list, dtype=np.int64)

        u_idx = df[user_col].map(self.user2idx).values.astype(np.int64)
        i_idx = df[item_col].map(self.item2idx).values.astype(np.int64)
        return u_idx, i_idx

    def fit(self, train_df: pd.DataFrame, user_col: str = "user", item_col: str = "item"):
        rng = np.random.RandomState(self.seed)

        u_idx, i_idx = self._build_mappings(train_df, user_col, item_col)
        n_users = len(self.idx2user)
        n_items = len(self.idx2item)
        F = self.n_factors
        logger.info(f"[BPRMF_submit] Users={n_users}, Items={n_items}, Factors={F}")

        scale = 1.0 / math.sqrt(F)
        self.P = rng.normal(0.0, scale, size=(n_users, F)).astype(np.float32)
        self.Q = rng.normal(0.0, scale, size=(n_items, F)).astype(np.float32)

        # positives
        user_pos: List[List[int]] = [[] for _ in range(n_users)]
        for uu, ii in zip(u_idx, i_idx):
            user_pos[uu].append(int(ii))
        self.user_pos = [np.array(lst, dtype=np.int64) for lst in user_pos]
        self.user_pos_sets = [set(lst) for lst in user_pos]

        item_counts = np.bincount(i_idx, minlength=n_items)
        self.popular_items = np.argsort(-item_counts)

        # optional IDF weights
        if self.use_idf_weight:
            df_item = train_df.groupby(item_col)[user_col].nunique().to_dict()
            n_users_ext = train_df[user_col].nunique()
            item_idf = {
                int(i_ext): math.log((1.0 + n_users_ext) / (1.0 + df_item[i_ext]))
                for i_ext in df_item
            }
            idf_internal = np.zeros(n_items, dtype=np.float32)
            for ext_id, w in item_idf.items():
                if ext_id in self.item2idx:
                    idf_internal[self.item2idx[ext_id]] = float(w)
            user_pos_probs: List[Optional[np.ndarray]] = []
            for pos_items in self.user_pos:
                if pos_items.size == 0:
                    user_pos_probs.append(None)
                    continue
                weights = idf_internal[pos_items]
                s = weights.sum()
                if s <= 0:
                    user_pos_probs.append(None)
                else:
                    user_pos_probs.append(weights / s)
            self.user_pos_probs = user_pos_probs
            logger.info("[BPRMF_submit] Using IDF-weighted positives.")
        else:
            self.user_pos_probs = [None for _ in range(n_users)]
            logger.info("[BPRMF_submit] Using uniform positive sampling.")

        lr = self.lr
        reg = self.reg
        logger.info(f"[BPRMF_submit] Training for {self.n_epochs} epochs, reg={reg}")

        for epoch in range(self.n_epochs):
            user_order = rng.permutation(n_users)
            for u in user_order:
                pos_items = self.user_pos[u]
                if pos_items.size == 0:
                    continue
                for _ in pos_items:
                    i = self._sample_positive(u, rng)
                    for _ in range(self.n_neg):
                        j = self._sample_negative(u, rng, n_items)
                        self._update_triplet(u, i, j, lr, reg)
            logger.info(f"[BPRMF_submit] Epoch {epoch+1}/{self.n_epochs} done")

    def _sample_positive(self, u: int, rng: np.random.RandomState) -> int:
        pos_items = self.user_pos[u]
        probs = self.user_pos_probs[u]
        if probs is not None:
            return int(rng.choice(pos_items, p=probs))
        else:
            return int(rng.choice(pos_items))

    def _sample_negative(self, u: int, rng: np.random.RandomState, n_items: int) -> int:
        pos_set = self.user_pos_sets[u]
        while True:
            j = int(rng.randint(n_items))
            if j not in pos_set:
                return j

    def _update_triplet(self, u: int, i: int, j: int, lr: float, reg: float):
        pu = self.P[u].copy()
        qi = self.Q[i].copy()
        qj = self.Q[j].copy()

        x_ui = float(np.dot(pu, qi))
        x_uj = float(np.dot(pu, qj))
        x_uij = x_ui - x_uj

        sigm = 1.0 / (1.0 + math.exp(-x_uij))
        grad = 1.0 - sigm

        self.P[u] += lr * (grad * (qi - qj) - reg * pu)
        self.Q[i] += lr * (grad * pu - reg * qi)
        self.Q[j] += lr * (-grad * pu - reg * qj)

    def _scores_for_user_idx(self, u_idx: int) -> np.ndarray:
        scores = self.P[u_idx] @ self.Q.T
        seen = self.user_pos[u_idx]
        if seen.size > 0:
            scores[seen] = -1e9
        return scores

    def recommend_all_users(self, k: int = 20) -> np.ndarray:
        if self.P is None or self.Q is None:
            raise RuntimeError("Model not fitted.")
        n_users = self.P.shape[0]
        recs = np.zeros((n_users, k), dtype=np.int64)
        for u_idx in range(n_users):
            scores = self._scores_for_user_idx(u_idx)
            if k >= scores.size:
                top_idx = np.argsort(-scores)[:k]
            else:
                top_idx = np.argpartition(-scores, k)[:k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
            recs[u_idx] = self.idx2item[top_idx]
        return recs


# =========================
# Main
# =========================

def main() -> None:
    train_path = "../train-1.txt"
    out_path = "bprmf_submission.txt"

    # ---- FILL THESE WITH THE BEST VALUES FROM BPRMF_pipeline ----
    n_factors = 128
    lr = 1e-3
    reg = 1e-4     # <- replace with best lambda
    n_epochs = 40
    n_neg = 1
    use_idf_weight = True
    # -------------------------------------------------------------

    logger.info(f"Loading training data from: {train_path}")
    train_df = load_train_txt_as_dataframe(train_path, user_col="user", item_col="item")

    model = BPRMF(
        n_factors=n_factors,
        lr=lr,
        reg=reg,
        n_epochs=n_epochs,
        n_neg=n_neg,
        use_idf_weight=use_idf_weight,
        seed=42,
    )
    model.fit(train_df, user_col="user", item_col="item")

    logger.info("Generating top-20 recommendations for all users")
    recs = model.recommend_all_users(k=20)

    logger.info(f"Saving submission file (no user ids) to: {out_path}")
    save_recommendations(recs, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
