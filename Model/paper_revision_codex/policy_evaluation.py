"""One evaluation context for every model under an explicit repeat policy.

Policies (never mixed in one comparison):
  repeat: the original protocol. Truth = all outer-period positive items of the
          fixed evaluation users. Items seen in training may be recommended.
  novel:  truth = original truth minus every item the user touched in the raw
          (pre-filter) training period. Users whose novel truth is empty leave
          the population. Each user's training-catalog items are never ranked.
In both policies the denominator is the policy truth, including items outside
the training catalog, and users without training vectors score zero.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data_protocol import POSITIVE_EVENTS, positive_sets, require_protocol
from recommendation_scoring import hybrid_scores, minmax

POLICIES = ('repeat', 'novel')


def top_k(scores, k, exclude=None, tie=None):
    """Positions of the k highest scores; ties go to the smaller ``tie`` key.

    ``tie`` defaults to the position itself. With catalog-aligned vectors the
    position is the catalog index, and the catalog is sorted by item ID, so this
    matches ``recommendation_scoring.top_indices``. For a candidate subset pass
    the candidates' catalog indices as ``tie``. Excluded positions are never
    returned, so the list may be shorter than k.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if k < 0 or not np.isfinite(scores).all():
        raise ValueError('Non-negative K and finite scores are required')
    if exclude is not None and len(exclude):
        scores = scores.copy()
        scores[np.asarray(exclude, dtype=np.int64)] = -np.inf
    allowed = int(np.isfinite(scores).sum())
    k = min(k, allowed)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k < len(scores):
        boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
        pool = np.flatnonzero(scores >= boundary)
    else:
        pool = np.flatnonzero(np.isfinite(scores))
    key = pool if tie is None else np.asarray(tie)[pool]
    order = np.lexsort((key, -scores[pool]))[:k]
    return pool[order].astype(np.int64)


class EvaluationContext:
    def __init__(self, directory, policy='repeat'):
        if policy not in POLICIES:
            raise ValueError(f'Unknown policy {policy}')
        self.directory = Path(directory)
        self.policy = policy
        self.manifest = require_protocol(self.directory)
        catalog = pd.read_csv(self.directory / 'candidate_items.csv').itemid.to_numpy(dtype=np.int64)
        if not np.all(np.diff(catalog) > 0):
            raise ValueError('Catalog must be strictly sorted by item ID')
        self.ids = catalog
        self.index = pd.Index(catalog)
        train = pd.read_csv(self.directory / 'events_train_clean.csv')
        self.train = train
        original = positive_sets(pd.read_csv(self.directory / 'events_test_original.csv'))
        users = pd.read_csv(self.directory / 'evaluation_users.csv').visitorid.astype(int).tolist()
        if set(users) != set(original):
            raise ValueError('Evaluation users and original truth disagree')
        raw = pd.read_csv(self.directory / 'train_event_audit.csv', usecols=['visitorid', 'itemid'])
        raw = raw.loc[raw.visitorid.isin(users)]
        self.raw_seen = {int(u): set(map(int, g.itemid)) for u, g in raw.groupby('visitorid')}
        if policy == 'repeat':
            truth = original
        else:
            truth = {u: original[u] - self.raw_seen.get(u, set()) for u in users}
            truth = {u: items for u, items in truth.items() if items}
        self.truth = truth
        self.users = sorted(truth)
        warm = set(map(int, train.visitorid.unique()))
        self.active = [u for u in self.users if u in warm]
        seen = train.loc[train.visitorid.isin(self.active), ['visitorid', 'itemid']].drop_duplicates()
        seen = seen.assign(idx=self.index.get_indexer(seen.itemid))
        if (seen.idx < 0).any():
            raise ValueError('Training item outside catalog')
        self.seen_idx = {int(u): np.sort(g.idx.to_numpy(dtype=np.int64)) for u, g in seen.groupby('visitorid')}
        if policy == 'novel':
            for u in self.active:
                if set(self.ids[self.seen_idx.get(u, [])]) & truth[u]:
                    raise AssertionError('Novel truth overlaps a training-catalog item')
        self.truth_idx = {u: set(self.index.get_indexer([i for i in truth[u] if i in self.index]).tolist())
                          for u in self.users}
        features = pd.read_csv(self.directory / 'feature_train_inliers.csv').set_index('itemid')
        self.available = features.reindex(self.ids).has_available.eq(1).to_numpy()
        self.discounts = 1 / np.log2(np.arange(1, 1001) + 1)

    # ---- population descriptors -------------------------------------------------
    def population(self):
        reachable = sum(1 for u in self.active if self.truth_idx[u])
        return {'policy': self.policy, 'evaluation_users': len(self.users), 'warm_users': len(self.active),
                'users_with_reachable_truth': reachable,
                'truth_items': int(sum(len(t) for t in self.truth.values())),
                'reachable_truth_items': int(sum(len(self.truth_idx[u]) for u in self.active)),
                'catalog_items': len(self.ids)}

    def exclusion(self, uid):
        return self.seen_idx.get(uid) if self.policy == 'novel' else None

    # ---- metrics ------------------------------------------------------------------
    def details(self, rankings, k=50):
        """Per-user metrics. Users absent from ``rankings`` contribute zeros."""
        rows = []
        for uid in self.users:
            ranked = np.asarray(rankings.get(uid, []), dtype=np.int64)[:k]
            relevant = self.truth_idx[uid]
            hit = np.fromiter((int(i) in relevant for i in ranked), dtype=bool, count=len(ranked))
            hits = int(hit.sum())
            idcg = float(self.discounts[:min(k, len(self.truth[uid]))].sum())
            if self.policy == 'novel' and len(ranked) and np.isin(ranked, self.seen_idx.get(uid, [])).any():
                raise AssertionError('Novel-only list contains a seen item')
            if len(set(ranked.tolist())) != len(ranked):
                raise AssertionError('Duplicate item in a recommendation list')
            rows.append({'visitorid': uid, 'ndcg': float((hit * self.discounts[:len(hit)]).sum() / idcg),
                         'recall': hits / len(self.truth[uid]), 'precision': hits / k,
                         'hit_rate': float(hits > 0), 'hits': hits, 'recommendations': len(ranked)})
        return pd.DataFrame(rows, columns=['visitorid', 'ndcg', 'recall', 'precision', 'hit_rate', 'hits',
                                           'recommendations'])

    def metrics(self, rankings, k=50):
        frame = self.details(rankings, k)
        return {'ndcg': float(frame.ndcg.mean()), 'recall': float(frame.recall.mean()),
                'precision': float(frame.precision.mean()), 'hit_rate': float(frame.hit_rate.mean()),
                'hit_users': int(frame.hit_rate.sum()), 'users': len(frame)}

    # ---- model rankings ---------------------------------------------------------------
    def rank_scores(self, scores, k=50):
        """``scores``: uid -> full-catalog score vector (aligned with ``self.ids``)."""
        return {u: top_k(v, k, self.exclusion(u)) for u, v in scores.items() if u in self.truth}

    def popularity_scores(self):
        counts = self.train.groupby('itemid').size().reindex(self.ids).fillna(0).to_numpy(dtype=np.float64)
        return counts

    def popularity(self, k=50):
        counts = self.popularity_scores()
        return {u: top_k(counts, k, self.exclusion(u)) for u in self.active}

    def recent(self, k=50, fill=False):
        """Most recent distinct training items, newest first (ties: lower item ID).

        Users with fewer than k distinct training items receive a shorter list
        unless ``fill`` appends popularity items not already listed. Under the
        novel policy every recent item is seen, so only the popularity fill remains.
        """
        frame = self.train.loc[self.train.visitorid.isin(self.active), ['visitorid', 'itemid', 'timestamp']]
        last = frame.groupby(['visitorid', 'itemid'], as_index=False).timestamp.max()
        last = last.sort_values(['visitorid', 'timestamp', 'itemid'], ascending=[True, False, True], kind='stable')
        last['idx'] = self.index.get_indexer(last.itemid)
        ordered = {int(u): g.idx.to_numpy(dtype=np.int64) for u, g in last.groupby('visitorid', sort=False)}
        popularity = self.popularity_scores() if fill else None
        result = {}
        for u in self.active:
            items = ordered.get(u, np.empty(0, dtype=np.int64))
            if self.policy == 'novel':
                items = items[:0]
            items = items[:k]
            if fill and len(items) < k:
                exclude = np.union1d(items, self.exclusion(u) if self.exclusion(u) is not None else [])
                items = np.r_[items, top_k(popularity, k - len(items), exclude.astype(np.int64))]
            result[u] = items.astype(np.int64)
        return result


class FactorScores:
    """Full-catalog ALS/BPR/GNN score vectors for the context's warm users."""

    def __init__(self, context, als=None, bpr=None, gnn=None):
        self.context = context
        self.als = {u: minmax(v).astype(np.float32) for u, v in (als or {}).items()}
        self.bpr = {u: minmax(v).astype(np.float32) for u, v in (bpr or {}).items()}
        self.gnn = {u: np.asarray(v, dtype=np.float32) for u, v in (gnn or {}).items()}

    def mf(self, uid, alpha):
        return alpha * self.als[uid] + (1 - alpha) * self.bpr[uid]


def factor_vectors(context, user_ids, user_factors, item_ids, item_factors):
    """uid -> score vector aligned with the context catalog."""
    order = pd.Index(np.asarray(item_ids, dtype=np.int64)).get_indexer(context.ids)
    if (order < 0).any():
        raise ValueError('Factor items do not cover the shared catalog')
    items = np.asarray(item_factors)[order]
    rows = pd.Index(np.asarray(user_ids, dtype=np.int64))
    result = {}
    for uid in context.active:
        position = rows.get_indexer([uid])[0]
        if position < 0:
            raise ValueError(f'Training user {uid} has no factor vector')
        result[uid] = items @ np.asarray(user_factors)[position]
    return result


def candidate_indices(scores, uid, alpha, count, strategy):
    """Same rule as improve_hybrid.candidate_indices, plus policy exclusion."""
    context = scores.context
    exclude = context.exclusion(uid)
    blend = scores.mf(uid, alpha)
    n = len(context.ids) if count == 'all' else int(count)
    if strategy == 'blend' or count == 'all':
        return top_k(blend, n, exclude)
    if strategy != 'balanced_union':
        raise ValueError(f'Unknown candidate strategy {strategy}')
    a = top_k(scores.als[uid], (n + 1) // 2, exclude)
    b = top_k(scores.bpr[uid], n // 2, exclude)
    union = np.union1d(a, b)
    if len(union) < n:
        blocked = union if exclude is None else np.union1d(union, exclude)
        union = np.r_[union, top_k(blend, n - len(union), blocked.astype(np.int64))]
    return union[top_k(blend[union], min(n, len(union)), tie=union)]


def hybrid_ranking(scores, config, k=50, prepared=None, mf_source='ensemble'):
    context = scores.context
    rankings = {}
    for uid in context.active:
        if uid not in scores.gnn:
            continue
        idx = prepared[uid] if prepared is not None else candidate_indices(
            scores, uid, config['alpha'], config['candidate_k'], config['strategy'])
        if not len(idx):
            rankings[uid] = idx
            continue
        mf = scores.als[uid] if mf_source == 'als' else scores.mf(uid, config['alpha'])
        fused = hybrid_scores(mf[idx], scores.gnn[uid][idx], context.available[idx],
                              config['omega'], config['gamma'], config['fusion'])
        rankings[uid] = idx[top_k(fused, k, tie=idx)]
    return rankings


def selection_score(metrics):
    """Single preregistered validation criterion for every model family."""
    return (metrics['ndcg'] + metrics['recall']) / 2
