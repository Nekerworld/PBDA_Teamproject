"""v9 Hybrid: MF module, graph module (LightGCN + item co-visit graph), behaviour/context module, learned re-ranker.

Modules and what each one adds to the candidate pool and to the re-ranker features:
  mf       ALS and BPR: top items of each, scores, catalog ranks, similarity to the user's last items in ALS space.
  graph    LightGCN (GNN) and an item co-visit graph built from the training events (two items are linked when one
           user touches both within ``window_ms``; links weighted by behaviour and recency, cosine-normalised):
           top items of each, scores, ranks, similarity to the last items, co-visit weight from the last item.
  context  the user's own history (repeat policy only), recently popular items and popular items of the user's recent
           categories; history counts and recency, item popularity, trend and conversion rate, the user's affinity to
           the item's category, user activity.
The ladder of audit/v9_dev_plan.md drops whole modules: a ranker without a module neither pools its candidates nor
reads its features.

Leakage rules (same as hybrid_ltr): everything here reads the training events of the context only
(``check_training_only``); labels come from the window the ranker is trained on.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

import hybrid_ltr as ltr
from learning import BEHAVIOR_WEIGHTS
from policy_evaluation import top_k

DAY_MS = ltr.DAY_MS
HOUR_MS = 3_600_000
MODULES = ('mf', 'graph', 'context')
FEATURE_GROUPS = {
    'mf': ['als', 'bpr', 'als_rank', 'bpr_rank', 'sim_als_last'],
    'graph': ['gnn', 'gnn_rank', 'sim_gnn_last', 'cv', 'cv_rank', 'cv_last'],
    'context': ['h_views', 'h_carts', 'h_purchases', 'h_weight', 'h_days', 'h_recent_pos', 'h_seen',
                'i_pop', 'i_pop7', 'i_pos7', 'i_days_last', 'i_pop1', 'i_cvr', 'c_pop7',
                'u_events', 'u_items', 'u_days_last', 'uc_share', 'uc_recent_pos', 'uc_last'],
}
FEATURES = [f for m in MODULES for f in FEATURE_GROUPS[m]]
POOL = dict(per_model=100, history=100, popular=50, category=50)


class CoVisitGraph:
    """Item-item co-visit graph of the training events, in catalog positions.

    Each pair of events of one user at most ``window_ms`` apart (and at most ``max_lag`` events apart) links the two
    items with weight mean(behaviour weights) * 0.5 ** (age of the later event / half-life). The matrix is symmetric,
    has a zero diagonal and is normalised by sqrt(degree_i * degree_j)."""

    def __init__(self, context, window_ms=HOUR_MS, max_lag=20, half_life_days=7.):
        end = ltr.check_training_only(context)
        train = context.train[['visitorid', 'itemid', 'event', 'timestamp']].sort_values(
            ['visitorid', 'timestamp'], kind='stable')
        idx = context.index.get_indexer(train.itemid)
        if (idx < 0).any():
            raise ValueError('Training item outside the catalog')
        user = train.visitorid.to_numpy()
        ts = train.timestamp.to_numpy(dtype=np.int64)
        weight = train.event.map(BEHAVIOR_WEIGHTS).to_numpy(dtype=np.float64)
        rows, cols, values = [], [], []
        for lag in range(1, max_lag + 1):
            a, b = slice(0, len(idx) - lag), slice(lag, len(idx))
            keep = (user[a] == user[b]) & (ts[b] - ts[a] <= window_ms) & (idx[a] != idx[b])
            if not keep.any():
                break
            decay = 0.5 ** ((end - ts[b][keep]) / DAY_MS / half_life_days)
            rows.append(idx[a][keep])
            cols.append(idx[b][keep])
            values.append((weight[a][keep] + weight[b][keep]) / 2 * decay)
        n = len(context.ids)
        r, c, v = (np.concatenate(x) if x else np.empty(0) for x in (rows, cols, values))
        matrix = sp.coo_matrix((np.r_[v, v], (np.r_[r, c].astype(np.int64), np.r_[c, r].astype(np.int64))),
                               shape=(n, n)).tocsr()
        matrix.sum_duplicates()
        degree = np.asarray(matrix.sum(axis=1)).ravel()
        scale = sp.diags(1 / np.sqrt(np.where(degree > 0, degree, 1)))
        self.matrix = (scale @ matrix @ scale).tocsr()
        self.end = end

    def scores(self, seeds, weights):
        """Sparse (positions, scores) of sum_s weight_s * row_s."""
        if not len(seeds):
            return np.empty(0, dtype=np.int64), np.empty(0)
        row = sp.csr_matrix((np.asarray(weights, dtype=np.float64), (np.zeros(len(seeds), dtype=np.int64),
                             np.asarray(seeds, dtype=np.int64))), shape=(1, self.matrix.shape[0]))
        out = (row @ self.matrix).tocoo()
        return out.col.astype(np.int64), out.data


class Signals:
    """Training-window signals of one context: History plus the co-visit graph, categories and conversion rates."""

    def __init__(self, context, history=None, graph=None, seeds=10, prior=20.):
        self.history = history or ltr.History(context)
        self.graph = graph or CoVisitGraph(context)
        end = self.history.end
        train = context.train[['visitorid', 'itemid', 'event', 'timestamp']].copy()
        train['idx'] = context.index.get_indexer(train.itemid)
        n = len(context.ids)
        flags = pd.read_csv(context.directory / 'feature_train_inliers.csv', usecols=['itemid', 'categoryid'])
        self.category = flags.set_index('itemid').categoryid.reindex(context.ids).fillna(-1).to_numpy(np.int64)
        views = np.bincount(train.idx[train.event == 'view'], minlength=n).astype(np.float64)
        positive = np.bincount(train.idx[train.event != 'view'], minlength=n).astype(np.float64)
        rate = positive.sum() / max(views.sum(), 1)
        day = train.timestamp >= end - DAY_MS
        week = train.timestamp >= end - 7 * DAY_MS
        cat_codes, cat_index = np.unique(self.category, return_inverse=True)
        cat_week = np.bincount(cat_index[train.idx[week]], minlength=len(cat_codes)).astype(np.float64)
        self.cat_index = cat_index
        self.item_extra = pd.DataFrame({'i_pop1': np.log1p(np.bincount(train.idx[day], minlength=n)),
                                        'i_cvr': (positive + prior * rate) / (views + prior),
                                        'c_pop7': np.log1p(cat_week[cat_index])})
        # popular items of each category over the last training week, most popular first (ties: lower position)
        counts = np.bincount(train.idx[week], minlength=n).astype(np.float64)
        order = np.lexsort((np.arange(n), -counts))
        order = order[counts[order] > 0]
        self.category_top = pd.Series(order).groupby(cat_index[order], sort=False).apply(np.asarray).to_dict()
        pairs = self.history.pairs.reset_index()
        pairs['cat'] = cat_index[pairs.idx.to_numpy()]
        share = pairs.groupby(['visitorid', 'cat']).weight.sum()
        self.user_cat_share = (share / share.groupby(level=0).transform('sum')).to_dict()
        self.seeds = seeds

    def user_categories(self, uid):
        """The user's categories (cat_index codes), most recent first."""
        recent = self.history.recent.get(uid, np.empty(0, dtype=np.int64))
        return list(dict.fromkeys(self.cat_index[recent].tolist()))

    def covisit(self, uid):
        """Co-visit scores from the user's ``seeds`` most recent items, each weighted by its history weight."""
        recent = self.history.recent.get(uid, np.empty(0, dtype=np.int64))[:self.seeds]
        if not len(recent):
            return np.empty(0, dtype=np.int64), np.empty(0)
        weights = self.history.pairs.loc[uid].reindex(recent).weight.to_numpy()
        return self.graph.scores(recent, weights)


def _top_sparse(positions, values, n, exclude=None):
    if exclude is not None and len(exclude) and len(positions):
        keep = ~np.isin(positions, exclude)
        positions, values = positions[keep], values[keep]
    order = np.lexsort((positions, -values))[:n]
    return positions[order]


def candidates(context, scores, signals, pool=POOL):
    """uid -> {module: catalog positions} (policy exclusions applied). Reads no labels."""
    history = signals.history
    result = {}
    for uid in context.active:
        exclude = context.exclusion(uid)
        cv_pos, cv_val = signals.covisit(uid)
        parts = {'mf': [top_k(scores.als[uid], pool['per_model'], exclude),
                        top_k(scores.bpr[uid], pool['per_model'], exclude)],
                 'graph': [top_k(scores.gnn[uid], pool['per_model'], exclude),
                           _top_sparse(cv_pos, cv_val, pool['per_model'], exclude)],
                 'context': [top_k(history.recent_popular, pool['popular'], exclude)]}
        if context.policy == 'repeat':
            parts['context'].append(history.recent.get(uid, np.empty(0, dtype=np.int64))[:pool['history']])
        cats = signals.user_categories(uid)[:3]
        for cat in cats:
            top = signals.category_top.get(cat, np.empty(0, dtype=np.int64))
            if exclude is not None and len(exclude):
                top = top[~np.isin(top, exclude)]
            parts['context'].append(top[:pool['category'] // len(cats)])
        result[uid] = {m: np.unique(np.concatenate(p).astype(np.int64)) for m, p in parts.items()}
    return result


def features(context, scores, signals, pool, als_items, gnn_items):
    """One row per (user, candidate of any module): uid, idx, in_<module> flags and FEATURES. Reads no labels."""
    union = {u: np.unique(np.concatenate(list(p.values()))) for u, p in pool.items()}
    frame = ltr.features(context, scores, signals.history, union, als_items, gnn_items)
    flags = {m: np.zeros(len(frame), dtype=bool) for m in MODULES}
    cv, cv_rank, cv_last, share, cat_pos, cat_last = (np.zeros(len(frame)) for _ in range(6))
    starts = np.r_[0, np.cumsum([len(union[u]) for u in union if len(union[u])])]
    users = [u for u in union if len(union[u])]
    if not np.array_equal(frame.uid.to_numpy(), np.repeat(users, np.diff(starts))):
        raise RuntimeError('Feature rows out of order')
    for k, uid in enumerate(users):
        rows = slice(starts[k], starts[k + 1])
        idx = union[uid]
        for m in MODULES:
            flags[m][rows] = np.isin(idx, pool[uid][m])
        positions, values = signals.covisit(uid)
        lookup = pd.Series(values, index=positions)
        cv[rows] = lookup.reindex(idx).fillna(0.).to_numpy()
        order = np.lexsort((positions, -values))
        rank = pd.Series(np.arange(1, len(order) + 1, dtype=np.float64), index=positions[order])
        cv_rank[rows] = np.log1p(rank.reindex(idx).fillna(len(context.ids)).to_numpy())
        recent = signals.history.recent.get(uid, np.empty(0, dtype=np.int64))
        if len(recent):
            cv_last[rows] = signals.graph.matrix[recent[0]].toarray().ravel()[idx]
        cats = signals.cat_index[idx]
        share[rows] = [signals.user_cat_share.get((uid, c), 0.) for c in cats]
        order_c = {c: i for i, c in enumerate(signals.user_categories(uid))}
        cat_pos[rows] = np.log1p([order_c.get(c, 1e4) for c in cats])
        cat_last[rows] = (cats == signals.cat_index[recent[0]]) if len(recent) else 0.
    for m in MODULES:
        frame[f'in_{m}'] = flags[m]
    frame['cv'], frame['cv_rank'], frame['cv_last'] = cv, cv_rank, cv_last
    frame['uc_share'], frame['uc_recent_pos'], frame['uc_last'] = share, cat_pos, cat_last.astype(np.float64)
    for column in signals.item_extra.columns:
        frame[column] = signals.item_extra[column].to_numpy()[frame.idx.to_numpy()]
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError('Non-finite feature')
    return frame


def module_rows(frame, modules):
    """Rows pooled by at least one of ``modules``."""
    return frame[[f'in_{m}' for m in modules]].any(axis=1).to_numpy()


def module_features(modules):
    return [f for m in MODULES if m in modules for f in FEATURE_GROUPS[m]]


def sample_negatives(frame, y, negatives, seed=42):
    """Row mask: every positive row and at most ``negatives`` random negative rows per user (training rows only)."""
    rng = np.random.default_rng(seed)
    key = rng.random(len(frame))
    key[np.asarray(y) == 1] = -1.
    order = pd.DataFrame({'uid': frame.uid.to_numpy(), 'key': key}).groupby('uid').key.rank(method='first')
    positives = pd.Series(np.asarray(y) == 1).groupby(frame.uid.to_numpy()).transform('sum').to_numpy()
    return (order.to_numpy() <= positives + negatives)


def select_params(context, frame, y, columns, grid=ltr.PARAM_GRID, folds=5, seed=42, k=50, sample_weight=None,
                  negatives=100):
    """ltr.select_params with negatives sampled in the training folds only; held-out users keep their full pool."""
    users = np.array(sorted(frame.uid.unique()))
    rng = np.random.default_rng(seed)
    assignment = dict(zip(users, rng.permutation(len(users)) % folds))
    fold_of = frame.uid.map(assignment).to_numpy()
    sampled = sample_negatives(frame, y, negatives, seed)
    rows = []
    for params in grid:
        values = []
        for f in range(folds):
            train, held = (fold_of != f) & sampled, fold_of == f
            model = ltr.fit(frame.loc[train].reset_index(drop=True), y[train], columns, params, seed,
                            None if sample_weight is None else np.asarray(sample_weight)[train])
            held_users = [u for u in users if assignment[u] == f]
            values.append(ltr.ndcg_of(context, ltr.rank(model, frame.loc[held], columns, k), held_users, k))
        ndcg, recall = np.mean(values, axis=0)
        rows.append({**params, 'cv_ndcg': ndcg, 'cv_recall': recall, 'cv_score': (ndcg + recall) / 2})
    table = pd.DataFrame(rows)
    best = table.sort_values('cv_score', ascending=False, kind='stable').iloc[0]
    return {key: (int(best[key]) if key != 'learning_rate' else float(best[key])) for key in grid[0]}, table, sampled


def covisit_ranking(context, signals, k=50):
    """Stand-alone co-visit recommender: the user's co-visit scores, top k (policy exclusions applied)."""
    out = {}
    for uid in context.active:
        positions, values = signals.covisit(uid)
        out[uid] = _top_sparse(positions, values, k, context.exclusion(uid))
    return out
