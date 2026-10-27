"""The stacked Hybrid reads labels only for its training window and features only from its training events."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr
from data_protocol import UnifiedPreprocessor
from policy_evaluation import EvaluationContext, FactorScores
from test_v6_smoke import make_raw


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('ltr')
    make_raw(root / 'raw')
    UnifiedPreprocessor(root / 'raw', root / 'prepared', n_estimators=20).run()
    return root / 'prepared'


def setup(directory, policy, seed=0):
    context = EvaluationContext(directory, policy)
    rng = np.random.default_rng(seed)
    n = len(context.ids)
    scores = FactorScores(context, *({u: rng.normal(size=n) for u in context.active} for _ in range(3)))
    tables = [ltr.aligned_item_factors(context.ids, rng.normal(size=(n, 4)), context) for _ in range(2)]
    history = ltr.History(context)
    pool = ltr.candidates(context, scores, history, dict(per_model=5, history=5, popular=3))
    return context, scores, history, pool, tables


class Sealed(dict):
    def __getattribute__(self, name):
        raise AssertionError(f'labels read through {name}')

    def __getitem__(self, key):
        raise AssertionError('labels read')

    def __iter__(self):
        raise AssertionError('labels read')

    def __len__(self):
        raise AssertionError('labels read')


@pytest.mark.parametrize('policy', ['repeat', 'novel'])
def test_features_read_no_labels(prepared, policy):
    context, scores, history, pool, (als, gnn) = setup(prepared, policy)
    expected = ltr.features(context, scores, history, pool, als, gnn)
    context.truth, context.truth_idx = Sealed(), Sealed()
    pd.testing.assert_frame_equal(ltr.features(context, scores, history, pool, als, gnn), expected)
    assert sorted(expected.columns[2:]) == sorted(ltr.FEATURES)


def test_training_events_reaching_the_label_window_are_refused(prepared):
    context = EvaluationContext(prepared, 'repeat')
    leaked = context.train.iloc[[0]].assign(timestamp=int(context.manifest['cutoff_timestamp']))
    context.train = pd.concat([context.train, leaked], ignore_index=True)
    with pytest.raises(RuntimeError, match='reach the label cutoff'):
        ltr.History(context)


def test_history_features_count_the_training_events_of_the_pair(prepared):
    context, scores, history, pool, (als, gnn) = setup(prepared, 'repeat')
    frame = ltr.features(context, scores, history, pool, als, gnn)
    row = frame.loc[frame.h_seen == 1].iloc[0]
    item = context.ids[int(row.idx)]
    events = context.train.loc[(context.train.visitorid == row.uid) & (context.train.itemid == item)]
    assert (row.h_views, row.h_carts, row.h_purchases) == tuple(
        float((events.event == e).sum()) for e in ('view', 'addtocart', 'transaction'))
    assert row.h_days == pytest.approx((history.end - events.timestamp.max()) / ltr.DAY_MS)


def test_pools_follow_the_policy(prepared):
    context, _, history, pool, _ = setup(prepared, 'novel')
    for uid, idx in pool.items():
        assert not np.isin(idx, context.seen_idx.get(uid, [])).any()
    context, _, history, pool, _ = setup(prepared, 'repeat')
    assert all(np.isin(history.recent[u][:5], pool[u]).all() for u in pool)


def test_labels_mark_exactly_the_policy_truth(prepared):
    context, scores, history, pool, (als, gnn) = setup(prepared, 'repeat')
    frame = ltr.features(context, scores, history, pool, als, gnn)
    y = ltr.labels(context, frame)
    assert y.tolist() == [int(i in context.truth_idx[u]) for u, i in zip(frame.uid, frame.idx)]


def test_ranker_is_deterministic_and_lists_are_valid(prepared):
    context, scores, history, pool, (als, gnn) = setup(prepared, 'novel')
    frame = ltr.features(context, scores, history, pool, als, gnn)
    y = ltr.labels(context, frame)
    params = dict(learning_rate=.1, max_iter=20, max_leaf_nodes=7)
    first = ltr.rank(ltr.fit(frame, y, ltr.FEATURES, params), frame, ltr.FEATURES, 4)
    second = ltr.rank(ltr.fit(frame, y, ltr.FEATURES, params), frame, ltr.FEATURES, 4)
    assert {u: v.tolist() for u, v in first.items()} == {u: v.tolist() for u, v in second.items()}
    assert all(len(v) <= 4 and len(set(v.tolist())) == len(v) for v in first.values())
    context.metrics(first, 4)  # raises on a seen item under the novel policy or a duplicate


@pytest.mark.parametrize('policy', ['repeat', 'novel'])
def test_training_window_weights_each_pair_by_its_strongest_label_period_event(prepared, policy):
    window = ltr.TrainingWindow(prepared, policy)
    audit = pd.read_csv(prepared / 'test_event_audit.csv')
    raw = pd.read_csv(prepared / 'train_event_audit.csv')
    audit = audit.loc[audit.visitorid.isin(set(window.train.visitorid))]
    if policy == 'novel':
        seen = set(zip(raw.visitorid, raw.itemid))
        audit = audit.loc[[(u, i) not in seen for u, i in zip(audit.visitorid, audit.itemid)]]
    assert set(window.active) == set(audit.visitorid) | set(window.truth)
    for uid in window.active:
        rows = audit.loc[(audit.visitorid == uid) & audit.itemid.isin(window.ids)]
        if policy == 'novel':
            rows = rows.loc[~rows.itemid.isin(raw.itemid[raw.visitorid == uid])]
        expected = rows.assign(w=rows.event.map(ltr.BEHAVIOR_WEIGHTS)).groupby('itemid').w.max()
        got = {int(window.ids[i]): w for i, w in window.relevance.get(uid, {}).items()}
        assert got == {int(i): float(w) for i, w in expected.items()}
    context = EvaluationContext(prepared, policy)
    assert {u: t for u, t in window.truth.items() if u in context.active} == {u: context.truth[u] for u in context.active}


def test_weighted_labels_follow_the_window_relevance(prepared):
    window = ltr.TrainingWindow(prepared, 'repeat')
    first = {u: next(iter(r.items())) for u, r in window.relevance.items()}
    frame = pd.DataFrame({'uid': [u for u in first for _ in range(2)],
                          'idx': [i for u in first for i in (first[u][0], -1)]})
    y, weight = ltr.weighted_labels(window, frame)
    assert y.tolist() == [1, 0] * len(first)
    assert weight.tolist() == [w for u in first for w in (first[u][1], 1.)]
    assert {1., 6., 12.} >= {w for r in window.relevance.values() for w in r.values()}


@pytest.fixture(scope='module')
def viewers(tmp_path_factory):
    """A window where six warm users only view items in the label period (no add-to-cart or purchase there)."""
    root = tmp_path_factory.mktemp('ltr_viewers')
    make_raw(root / 'raw')
    events = pd.read_csv(root / 'raw' / 'events.csv')
    users = list(range(41, 47))
    extra = [(t, u, 'view', i, np.nan) for u in users for t, i in zip(range(100_000, 160_000, 10_000), (1, 2, 3, 1, 2, 3))]
    extra += [(999_500 + u, u, 'view', 1, np.nan) for u in users]
    extra = pd.DataFrame(extra, columns=events.columns)
    pd.concat([events, extra]).sort_values('timestamp', kind='stable').to_csv(root / 'raw' / 'events.csv', index=False)
    UnifiedPreprocessor(root / 'raw', root / 'prepared', n_estimators=20).run()
    return root / 'prepared', users


def test_training_window_learns_from_users_who_only_viewed(viewers):
    root, users = viewers
    window = ltr.TrainingWindow(root, 'repeat')
    assert set(users) <= set(window.active) and not set(users) & set(window.truth)
    assert all(list(window.relevance[u].values()) == [1.] for u in users)


def test_training_window_capping_keeps_every_user_with_policy_truth(viewers):
    root, _ = viewers
    window = ltr.TrainingWindow(root, 'repeat')
    assert window.truth and len(window.active) > len(window.truth) + 1
    capped = ltr.TrainingWindow(root, 'repeat', max_users=len(window.truth) + 1)
    assert set(window.truth) <= set(capped.active) and len(capped.active) == len(window.truth) + 1


def test_cv_metric_matches_the_evaluation_context(prepared):
    context, scores, history, pool, (als, gnn) = setup(prepared, 'repeat')
    context.truth = {u: t | {10 ** 12} for u, t in context.truth.items()}  # one positive outside the catalog
    rankings = context.rank_scores(scores.als, 10)
    expected = context.details(rankings, 10)
    assert ltr.ndcg_of(context, rankings, context.active, 10) == pytest.approx(
        (expected.ndcg.mean(), expected.recall.mean()))
