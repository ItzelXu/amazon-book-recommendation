# cited from https://www.stepbystepdatascience.com/ease

import scipy.sparse as sp
import numpy as np

def load_interactions_as_csr(path):
    """
    Reads a file where each line is:
        user_id item_1 item_2 ... item_k
    and returns a CSR matrix X (num_users x num_items) with binary entries.
    """
    user_ids = []
    item_ids = []

    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            u = int(tokens[0])
            for t in tokens[1:]:
                i = int(t)
                user_ids.append(u)
                item_ids.append(i)

    if not user_ids:
        raise ValueError("No interactions found in file:", path)

    num_users = max(user_ids) + 1
    num_items = max(item_ids) + 1

    data = np.ones(len(user_ids), dtype=np.float64)

    X = sp.csr_matrix(
        (data, (user_ids, item_ids)),
        shape=(num_users, num_items),
        dtype=np.float64,
    )
    return X

class EASE:
    def __init__(self, lambda_: float = 0.5):
        """
        lambda_ : L2 regularization strength used on the diagonal of the
                  item–item Gram matrix.
        """
        self.lambda_ = lambda_
        self.B = None           # item–item weight matrix
        self.num_items = None

    def fit(self, X: sp.csr_matrix):
        """
        Fit EASE on a user–item interaction matrix X (CSR, shape [U, I]).
        X is assumed binary (implicit feedback).
        """
        # Ensure float64 for numerical stability
        X = X.astype(np.float64)

        # Item–item Gram matrix G = X^T X (co-occurrence counts)
        # This is sparse @ sparse -> sparse, then densified.
        G = (X.T @ X).toarray()

        n_items = G.shape[0]
        self.num_items = n_items

        # Add L2 regularization to the diagonal
        diag_idx = np.arange(n_items)
        G[diag_idx, diag_idx] += self.lambda_

        # Invert the regularized Gram matrix
        P = np.linalg.inv(G)

        # Compute B with safety guards for NaN/Inf
        diagP = np.diag(P).copy()
        # Clamp very small |diagP| to avoid division by ~0
        eps = 1e-8
        small = np.abs(diagP) < eps
        if np.any(small):
            print(f"[EASE] Warning: {small.sum()} diagonal entries of P are very small; clamping.")
            diagP[small] = eps

        # Compute EASE weight matrix B
        # B = -P / np.diag(P)
        B = -P / diagP
        B[diag_idx, diag_idx] = 0.0  # no self-loops

        self.B = B
        return self

    def recommend(self, X: sp.csr_matrix, top_k: int = 20, remove_seen: bool = True):
        """
        Generate top_k recommendations for *all* users in X.

        Parameters
        ----------
        X : csr_matrix, shape (num_users, num_items)
            Interaction matrix for the users we want recommendations for.
            Typically the same as the training matrix (X_train).
        top_k : int
            Number of items to recommend per user.
        remove_seen : bool
            If True, filter out items the user has already interacted with.

        Returns
        -------
        recs : np.ndarray, shape (num_users, top_k)
            Each row u contains the item indices of the top_k recommended items
            for user u, sorted by descending score.
        """
        if self.B is None:
            raise RuntimeError("You must call fit(X) before recommend().")

        X = X.tocsr()
        num_users, num_items = X.shape
        assert num_items == self.num_items, "X must have same #items as training."

        top_k = min(top_k, num_items)
        recs = np.zeros((num_users, top_k), dtype=np.int64)

        for u in range(num_users):
            row = X[u]  # 1 x num_items CSR
            # If user has no interactions, give zero scores
            if row.nnz == 0:
                scores = np.zeros(num_items, dtype=np.float64)
            else:
                scores = row @ self.B      # (1 x I) dense
                scores = np.asarray(scores).ravel()

            if remove_seen and row.nnz > 0:
                # Set scores of seen items to a very low value
                scores[row.indices] = -1e9

            # Top-k indices for this user
            idx = np.argpartition(scores, -top_k)[-top_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]  # sort by score desc

            recs[u, :] = idx

        return recs

# 1. Load data
X_train = load_interactions_as_csr("../train-1.txt")

# 2. Train EASE (tune lambda_ if you want)
ease = EASE(lambda_=500.0)   # example value; you can tune
ease.fit(X_train)

# 3. Get top-20 recs for each user
top_k = 20
recs = ease.recommend(X_train, top_k=top_k, remove_seen=True)

# 4. Save to submission file: "item1 ... item20"
def save_recommendations(recs: np.ndarray, path: str):
    """
    Each line: 20 item IDs (space separated), in the same user order
    as the rows in `recs` / X_train.
    """
    num_users, k = recs.shape
    with open(path, "w") as f:
        for u in range(num_users):
            items_str = " ".join(str(it) for it in recs[u])
            f.write(items_str + "\n")

save_recommendations(recs, "ease_submission.txt")
