"""Label-free scoring and deterministic ranking on the shared training catalog."""
from pathlib import Path
import numpy as np
import pandas as pd


def minmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite model scores')
    spread = np.ptp(scores)
    return np.zeros_like(scores) if spread == 0 else (scores - scores.min()) / spread


def ranks_from_scores(scores):
    """1-based ranks; ties broken by lower index for stability."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def top_indices(scores, item_ids, k):
    if k < 1 or not np.isfinite(scores).all():
        raise ValueError('Positive K and finite scores are required')
    return np.lexsort((item_ids, -np.asarray(scores)))[:k]


def recommendation_frame(records):
    return pd.DataFrame(records, columns=['visitorid', 'itemid', 'score', 'rank'])


def append_ranking(records, uid, item_ids, scores, k):
    records.extend((int(uid), int(item_ids[i]), float(scores[i]), rank)
                   for rank, i in enumerate(top_indices(scores, item_ids, k), 1))


def hybrid_scores(als_scores, gnn_scores, available, als_weight, inventory_gamma,
                  fusion='minmax', rrf_k=60):
    """Fuse ALS and GNN candidate scores, then apply inventory gamma.

    fusion:
      - minmax: weighted sum of min-max scores (revision default)
      - rrf: reciprocal rank fusion
      - gate: ALS base + (1-als_weight) * (GNN - ALS) on min-max scale
    """
    if not 0 <= als_weight <= 1 or not np.isfinite(inventory_gamma) or inventory_gamma <= 0:
        raise ValueError('Hybrid weight must be in [0, 1]; inventory gamma must be positive')
    als_scores = np.asarray(als_scores, dtype=np.float64)
    gnn_scores = np.asarray(gnn_scores, dtype=np.float64)
    if als_scores.shape != gnn_scores.shape:
        raise ValueError('ALS and GNN score shapes disagree')
    if fusion == 'minmax':
        fused = als_weight * minmax(als_scores) + (1 - als_weight) * minmax(gnn_scores)
    elif fusion == 'rrf':
        if rrf_k < 1:
            raise ValueError('rrf_k must be positive')
        fused = (als_weight / (rrf_k + ranks_from_scores(als_scores))
                 + (1 - als_weight) / (rrf_k + ranks_from_scores(gnn_scores)))
    elif fusion == 'gate':
        als_n = minmax(als_scores)
        gnn_n = minmax(gnn_scores)
        fused = als_n + (1 - als_weight) * (gnn_n - als_n)
    else:
        raise ValueError(f'Unknown fusion mode: {fusion}')
    return fused * np.where(available, inventory_gamma, 1.)


def hybrid_rerank_scores(als_scores, gnn_scores, available, als_weight, inventory_gamma,
                         fusion='minmax', rrf_k=60, rerank_m=None):
    """Optionally fuse only the ALS top-M candidates; keep the rest in ALS order below them."""
    als_scores = np.asarray(als_scores, dtype=np.float64)
    gnn_scores = np.asarray(gnn_scores, dtype=np.float64)
    available = np.asarray(available)
    if rerank_m is None or rerank_m >= len(als_scores):
        return hybrid_scores(als_scores, gnn_scores, available, als_weight, inventory_gamma, fusion, rrf_k)
    if rerank_m < 1:
        raise ValueError('rerank_m must be positive or None')
    order = np.argsort(-als_scores, kind='stable')
    top = order[:rerank_m]
    rest = order[rerank_m:]
    fused_top = hybrid_scores(als_scores[top], gnn_scores[top], available[top],
                              als_weight, inventory_gamma, fusion, rrf_k)
    result = np.empty(len(als_scores), dtype=np.float64)
    result[top] = fused_top
    floor = float(fused_top.min()) - 1.0 if len(fused_top) else 0.0
    result[rest] = floor - np.arange(len(rest), dtype=np.float64)
    return result


class FactorScorer:
    def __init__(self, directory, ensemble_alpha):
        directory = Path(directory)
        self.alpha = ensemble_alpha
        users = pd.read_csv(directory / 'als_user_mapping.csv').visitorid.to_numpy(dtype=np.int64)
        self.item_ids = pd.read_csv(directory / 'als_item_mapping.csv').itemid.to_numpy(dtype=np.int64)
        self.users = {int(uid): idx for idx, uid in enumerate(users)}
        expected_users = set(pd.read_csv(directory / 'events_train_clean.csv', usecols=['visitorid']).visitorid)
        if set(self.users) != expected_users:
            raise ValueError('ALS/BPR user mapping is missing retained training users')
        self.als_u = np.load(directory / 'als_user_factors.npy')
        self.als_i = np.load(directory / 'als_item_factors.npy')
        self.bpr_u = np.load(directory / 'bpr_user_factors.npy')
        self.bpr_i = np.load(directory / 'bpr_item_factors.npy')
        catalog = pd.read_csv(directory / 'candidate_items.csv').itemid.to_numpy(dtype=np.int64)
        if len(set(self.item_ids)) != len(self.item_ids) or set(self.item_ids) != set(catalog):
            raise ValueError('ALS/BPR item factors do not match the shared training catalog')
        if (len(self.als_u) != len(users) or len(self.bpr_u) != len(users)
                or len(self.als_i) != len(catalog) or len(self.bpr_i) != len(catalog)):
            raise ValueError('Factor dimensions and ID mappings disagree')

    def scores(self, uid):
        if uid not in self.users:
            raise ValueError(f'ALS/BPR embedding missing for evaluation user {uid}')
        row = self.users[uid]
        als = self.als_i @ self.als_u[row]
        bpr = self.bpr_i @ self.bpr_u[row]
        combined = self.alpha * minmax(als) + (1 - self.alpha) * minmax(bpr)
        return als, bpr, combined


class GNNScorer:
    def __init__(self, directory, item_order=None):
        directory = Path(directory)
        users = pd.read_csv(directory / 'gnn_user_embeddings.csv').set_index('visitorid')
        items = pd.read_csv(directory / 'gnn_item_embeddings.csv').set_index('itemid')
        catalog = pd.read_csv(directory / 'candidate_items.csv').itemid.to_numpy(dtype=np.int64)
        if not users.index.is_unique or not items.index.is_unique or set(items.index) != set(catalog):
            raise ValueError('GNN embeddings do not match the shared training vocabulary')
        if item_order is not None:
            items = items.loc[item_order]
        self.item_ids = items.index.to_numpy(dtype=np.int64)
        self.users = {int(uid): idx for idx, uid in enumerate(users.index)}
        expected_users = set(pd.read_csv(directory / 'events_train_clean.csv', usecols=['visitorid']).visitorid)
        if set(self.users) != expected_users:
            raise ValueError('GNN embeddings are missing retained training users')
        self.user_vectors = users.to_numpy(dtype=np.float32)
        self.item_vectors = items.to_numpy(dtype=np.float32)

    def scores(self, uid, indices=None):
        if uid not in self.users:
            raise ValueError(f'GNN embedding missing for evaluation user {uid}')
        vectors = self.item_vectors if indices is None else self.item_vectors[indices]
        return vectors @ self.user_vectors[self.users[uid]]
