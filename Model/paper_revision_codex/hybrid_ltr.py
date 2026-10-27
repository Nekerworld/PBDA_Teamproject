"""Stacked Hybrid (v8 candidate): a learned re-ranker over the ALS, BPR and GNN candidates.

Each warm user gets a candidate pool (top items of ALS, BPR and GNN, the user's own recent items under the repeat
policy, and recently popular items). Every (user, candidate) pair is described by the three model scores and ranks,
the user's behaviour with the item (view/cart/purchase counts, decayed weight, recency), similarity of the item to
the user's last items in the ALS and GNN spaces, and item/user activity. A gradient-boosted classifier trained on an
earlier window's labels scores the pool.

Leakage rules kept by construction:
- features read only the training events and training-time item flags of the context (``context.train``,
  ``context.available``); ``check_training_only`` refuses a context whose training events reach its label cutoff;
- labels are read only for the window the ranker is trained on: ``labels`` (its policy truth) or ``TrainingWindow``
  with ``weighted_labels`` (all of its label-period events, behaviour-weighted);
- the ranker's settings are chosen by user-grouped cross-validation inside the training window (``select_params``),
  never on the window it is evaluated on.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from learning import BEHAVIOR_WEIGHTS
from policy_evaluation import top_k

DAY_MS = 86_400_000.
FEATURE_GROUPS = {
    'mf': ['als', 'bpr', 'als_rank', 'bpr_rank', 'sim_als_last'],
    'gnn': ['gnn', 'gnn_rank', 'sim_gnn_last'],
    'history': ['h_views', 'h_carts', 'h_purchases', 'h_weight', 'h_days', 'h_recent_pos', 'h_seen'],
    'context': ['i_pop', 'i_pop7', 'i_pos7', 'i_days_last', 'i_available', 'u_events', 'u_items', 'u_days_last'],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
PARAM_GRID = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves)
              for lr, it, leaves in itertools.product((.05, .1), (100, 300), (15, 31))]
POOL = dict(per_model=100, history=100, popular=50)


def check_training_only(context):
    """The features may only use events before the label window of the context."""
    cutoff = int(context.manifest['cutoff_timestamp'])
    end = int(context.train.timestamp.max())
    if end >= cutoff:
        raise RuntimeError(f'Training events of {context.directory} reach the label cutoff ({end} >= {cutoff})')
    return end


def aligned_item_factors(item_ids, factors, context):
    """Item factor rows ordered like the context catalog, L2-normalised for cosine similarity."""
    order = pd.Index(np.asarray(item_ids, dtype=np.int64)).get_indexer(context.ids)
    if (order < 0).any():
        raise ValueError('Item factors do not cover the catalog')
    rows = np.asarray(factors, dtype=np.float64)[order]
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms == 0, 1, norms)


class History:
    """Per-user and per-item activity of the training window, in catalog positions."""

    def __init__(self, context, half_life_days=7.):
        end = check_training_only(context)
        train = context.train[['visitorid', 'itemid', 'event', 'timestamp']].copy()
        train['idx'] = context.index.get_indexer(train.itemid)
        if (train.idx < 0).any():
            raise ValueError('Training item outside the catalog')
        age = (end - train.timestamp.to_numpy(dtype=np.float64)) / DAY_MS
        train['weight'] = train.event.map(BEHAVIOR_WEIGHTS).to_numpy() * 0.5 ** (age / half_life_days)
        for event in BEHAVIOR_WEIGHTS:
            train[event] = (train.event == event).astype(np.int64)
        pairs = train.groupby(['visitorid', 'idx']).agg(views=('view', 'sum'), carts=('addtocart', 'sum'),
                                                        purchases=('transaction', 'sum'), weight=('weight', 'sum'),
                                                        last=('timestamp', 'max'))
        self.pairs = pairs
        self.end = end
        recent = pairs.reset_index().sort_values(['visitorid', 'last', 'idx'], ascending=[True, False, True],
                                                 kind='stable')
        self.recent = {int(u): g.idx.to_numpy(dtype=np.int64) for u, g in recent.groupby('visitorid', sort=False)}
        n = len(context.ids)
        week = train.timestamp >= end - 7 * DAY_MS
        positive = train.event.isin(('addtocart', 'transaction'))
        self.item = pd.DataFrame({
            'i_pop': np.log1p(np.bincount(train.idx, minlength=n)),
            'i_pop7': np.log1p(np.bincount(train.idx[week], minlength=n)),
            'i_pos7': np.log1p(np.bincount(train.idx[week & positive], minlength=n)),
            'i_days_last': (end - train.groupby('idx').timestamp.max().reindex(range(n)).fillna(-np.inf)
                            .clip(lower=end - 365 * DAY_MS).to_numpy()) / DAY_MS,
            'i_available': np.asarray(context.available, dtype=np.float64)})
        users = train.groupby('visitorid')
        self.user = pd.DataFrame({'u_events': np.log1p(users.size()), 'u_items': np.log1p(users.idx.nunique()),
                                  'u_days_last': (end - users.timestamp.max()) / DAY_MS})
        self.recent_popular = np.bincount(train.idx[week], minlength=n).astype(np.float64)


def candidates(context, scores, history, pool=POOL):
    """uid -> sorted catalog positions of the candidate pool (policy exclusions applied)."""
    result = {}
    for uid in context.active:
        exclude = context.exclusion(uid)
        parts = [top_k(scores.als[uid], pool['per_model'], exclude), top_k(scores.bpr[uid], pool['per_model'], exclude),
                 top_k(scores.gnn[uid], pool['per_model'], exclude),
                 top_k(history.recent_popular, pool['popular'], exclude)]
        if context.policy == 'repeat':
            parts.append(history.recent.get(uid, np.empty(0, dtype=np.int64))[:pool['history']])
        result[uid] = np.unique(np.concatenate(parts).astype(np.int64))
    return result


def _catalog_rank(vector):
    order = np.argsort(-np.asarray(vector, dtype=np.float64), kind='stable')
    rank = np.empty(len(order), dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    return np.log1p(rank)


def features(context, scores, history, pool, als_items, gnn_items, last_n=3):
    """One row per (user, candidate): ``uid``, ``idx`` and FEATURES. Reads no labels."""
    blocks = []
    for uid, idx in pool.items():
        if not len(idx):
            continue
        frame = pd.DataFrame({'uid': uid, 'idx': idx})
        for name in ('als', 'bpr', 'gnn'):
            vector = getattr(scores, name)[uid]
            frame[name] = vector[idx]
            frame[f'{name}_rank'] = _catalog_rank(vector)[idx]
        last = history.recent.get(uid, np.empty(0, dtype=np.int64))[:last_n]
        for name, table in (('als', als_items), ('gnn', gnn_items)):
            frame[f'sim_{name}_last'] = (table[idx] @ table[last].T).max(axis=1) if len(last) else 0.
        try:
            pairs = history.pairs.loc[uid].reindex(idx)
        except KeyError:
            pairs = pd.DataFrame(index=idx, columns=history.pairs.columns, dtype=np.float64)
        frame['h_views'] = pairs.views.fillna(0).to_numpy()
        frame['h_carts'] = pairs.carts.fillna(0).to_numpy()
        frame['h_purchases'] = pairs.purchases.fillna(0).to_numpy()
        frame['h_weight'] = pairs.weight.fillna(0).to_numpy()
        frame['h_days'] = ((history.end - pairs['last'].to_numpy(dtype=np.float64)) / DAY_MS)
        frame['h_days'] = frame.h_days.fillna(365.).clip(upper=365.)
        position = pd.Series(np.arange(len(history.recent.get(uid, []))), index=history.recent.get(uid, []))
        frame['h_recent_pos'] = np.log1p(position.reindex(idx).fillna(1e4).to_numpy())
        frame['h_seen'] = pairs.views.notna().to_numpy(dtype=np.float64)
        for column in history.item.columns:
            frame[column] = history.item[column].to_numpy()[idx]
        user = history.user.loc[uid]
        for column in history.user.columns:
            frame[column] = float(user[column])
        blocks.append(frame)
    out = pd.concat(blocks, ignore_index=True)
    if not np.isfinite(out[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError('Non-finite feature')
    return out


def labels(context, frame):
    """1 where the candidate is a label-window positive of the policy truth (training window only)."""
    truth = context.truth_idx
    return np.fromiter((idx in truth.get(uid, ()) for uid, idx in zip(frame.uid, frame.idx)), dtype=np.int8,
                       count=len(frame))


class TrainingWindow:
    """The training window of the ranker seen as a context for every warm user with any label-window event.

    The evaluation population of a window is the users with an add-to-cart or purchase there (a few hundred in a
    RetailRocket window); the users who only viewed items outnumber them by far. The ranker learns which candidates a
    user interacts with next from all of them, each pair weighted by its strongest label-window event (view 1, cart 6,
    purchase 12, the behaviour weights of the base models). Only the window the ranker is trained on is read here.
    Policy rules follow EvaluationContext: the novel policy never pools a training-catalog item the user touched, and
    its labels leave out every item the user touched in the raw training period."""

    def __init__(self, directory, policy, max_users=None, seed=42):
        from data_protocol import positive_sets, require_protocol
        self.directory, self.policy = Path(directory), policy
        self.manifest = require_protocol(self.directory)
        self.ids = pd.read_csv(self.directory / 'candidate_items.csv').itemid.to_numpy(dtype=np.int64)
        self.index = pd.Index(self.ids)
        self.train = pd.read_csv(self.directory / 'events_train_clean.csv')
        flags = pd.read_csv(self.directory / 'feature_train_inliers.csv').set_index('itemid')
        self.available = flags.reindex(self.ids).has_available.eq(1).to_numpy()
        warm = set(map(int, self.train.visitorid.unique()))
        future = pd.read_csv(self.directory / 'test_event_audit.csv', usecols=['visitorid', 'itemid', 'event'])
        future = future.loc[future.visitorid.isin(warm)]
        raw = pd.read_csv(self.directory / 'train_event_audit.csv', usecols=['visitorid', 'itemid'])
        raw_seen = raw.loc[raw.visitorid.isin(set(future.visitorid))].groupby('visitorid').itemid.agg(set).to_dict()
        original = pd.read_csv(self.directory / 'events_test_original.csv')
        truth = {u: t for u, t in positive_sets(original.loc[original.visitorid.isin(warm)]).items()}
        if policy == 'novel':
            future = future.loc[[i not in raw_seen.get(u, ()) for u, i in zip(future.visitorid, future.itemid)]]
            truth = {u: t - raw_seen.get(u, set()) for u, t in truth.items()}
        truth = {int(u): set(map(int, t)) for u, t in truth.items() if t}
        users = sorted(set(map(int, future.visitorid)) | set(truth))
        if max_users is not None and len(users) > max_users:
            rest = sorted(set(users) - set(truth))
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(rest), size=max(max_users - len(truth), 0), replace=False)
            users = sorted(set(truth) | {rest[i] for i in keep})
        self.active = users
        known = future.loc[future.visitorid.isin(users) & future.itemid.isin(self.index)]
        known = known.assign(idx=self.index.get_indexer(known.itemid), weight=known.event.map(BEHAVIOR_WEIGHTS))
        self.relevance = {int(u): g.groupby('idx').weight.max().to_dict() for u, g in known.groupby('visitorid')}
        self.truth = truth
        self.truth_idx = {u: set(self.index.get_indexer([i for i in t if i in self.index]).tolist()) for u, t in truth.items()}
        seen = self.train.loc[self.train.visitorid.isin(users), ['visitorid', 'itemid']].drop_duplicates()
        self.seen_idx = {int(u): np.sort(self.index.get_indexer(g.itemid)) for u, g in seen.groupby('visitorid')}

    def exclusion(self, uid):
        return self.seen_idx.get(uid) if self.policy == 'novel' else None


def weighted_labels(window, frame):
    """(target, sample weight): target 1 where the user interacts with the candidate in the training window's label
    period, weighted by the strongest event; 0 rows weigh 1."""
    weight = np.fromiter((window.relevance.get(u, {}).get(i, 0.) for u, i in zip(frame.uid, frame.idx)),
                         dtype=np.float64, count=len(frame))
    return (weight > 0).astype(np.int8), np.where(weight > 0, weight, 1.)


def _ranker(params, seed=42):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=params['learning_rate'], max_iter=params['max_iter'],
                                          max_leaf_nodes=params['max_leaf_nodes'], min_samples_leaf=50,
                                          l2_regularization=1., early_stopping=False, random_state=seed)


def fit(frame, y, columns, params, seed=42, sample_weight=None):
    keep = frame.uid.isin(frame.uid[y == 1].unique()).to_numpy()  # users with no positive candidate carry no order
    model = _ranker(params, seed)
    model.fit(frame.loc[keep, columns].to_numpy(dtype=np.float64), y[keep],
              sample_weight=None if sample_weight is None else np.asarray(sample_weight)[keep])
    return model


def rank(model, frame, columns, k=50):
    """uid -> top-k catalog positions by ranker score (ties: lower catalog position)."""
    score = model.predict_proba(frame[columns].to_numpy(dtype=np.float64))[:, 1]
    out = {}
    for uid, rows in frame.assign(score=score).groupby('uid', sort=False):
        idx, s = rows.idx.to_numpy(dtype=np.int64), rows.score.to_numpy()
        out[int(uid)] = idx[np.lexsort((idx, -s))][:k]
    return out


def ndcg_of(context, rankings, users, k=50):
    """Mean NDCG@k and Recall@k over ``users`` that have positive truth (EvaluationContext.details formulas)."""
    discounts = 1 / np.log2(np.arange(2, k + 2))
    values = []
    for uid in users:
        truth = context.truth.get(uid)
        if not truth:
            continue
        ranked = np.asarray(rankings.get(uid, []), dtype=np.int64)[:k]
        hit = np.isin(ranked, list(context.truth_idx[uid])).astype(np.float64)
        values.append(((hit * discounts[:len(hit)]).sum() / discounts[:min(k, len(truth))].sum(), hit.sum() / len(truth)))
    return tuple(np.mean(values, axis=0)) if values else (0., 0.)


def select_params(context, frame, y, columns, grid=PARAM_GRID, folds=5, seed=42, k=50, sample_weight=None):
    """Ranker settings by user-grouped K-fold inside the training window: max mean(NDCG@k, Recall@k) on the
    held-out users' positive truth."""
    users = np.array(sorted(frame.uid.unique()))
    rng = np.random.default_rng(seed)
    assignment = dict(zip(users, rng.permutation(len(users)) % folds))
    fold_of = frame.uid.map(assignment).to_numpy()
    rows = []
    for params in grid:
        values = []
        for f in range(folds):
            train, held = fold_of != f, fold_of == f
            model = fit(frame.loc[train].reset_index(drop=True), y[train], columns, params, seed,
                        None if sample_weight is None else np.asarray(sample_weight)[train])
            held_users = [u for u in users if assignment[u] == f]
            values.append(ndcg_of(context, rank(model, frame.loc[held], columns, k), held_users, k))
        ndcg, recall = np.mean(values, axis=0)
        rows.append({**params, 'cv_ndcg': ndcg, 'cv_recall': recall, 'cv_score': (ndcg + recall) / 2})
    table = pd.DataFrame(rows)
    best = table.sort_values('cv_score', ascending=False, kind='stable').iloc[0]
    return {key: (int(best[key]) if key != 'learning_rate' else float(best[key])) for key in grid[0]}, table
