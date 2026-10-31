"""v9 Hybrid modules read only the training events of their window and keep module ablations separate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr
import hybrid_v9 as v9
from data_protocol import UnifiedPreprocessor
from policy_evaluation import EvaluationContext, FactorScores
from test_hybrid_ltr import Sealed
from test_v6_smoke import make_raw

SMALL = dict(per_model=5, conversion=3, history=5, popular=3, category=3)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('v9')
    make_raw(root / 'raw')
    UnifiedPreprocessor(root / 'raw', root / 'prepared', n_estimators=20).run()
    return root / 'prepared'


def setup(directory, policy, seed=0):
    context = EvaluationContext(directory, policy)
    rng = np.random.default_rng(seed)
    n = len(context.ids)
    scores = FactorScores(context, *({u: rng.normal(size=n) for u in context.active} for _ in range(3)))
    tables = [ltr.aligned_item_factors(context.ids, rng.normal(size=(n, 4)), context) for _ in range(2)]
    signals = v9.Signals(context)
    pool = v9.candidates(context, scores, signals, SMALL)
    return context, scores, signals, pool, tables


@pytest.mark.parametrize('policy', ['repeat', 'novel'])
def test_features_read_no_labels(prepared, policy):
    context, scores, signals, pool, (als, gnn) = setup(prepared, policy)
    expected = v9.features(context, scores, signals, pool, als, gnn)
    context.truth, context.truth_idx = Sealed(), Sealed()
    signals = v9.Signals(context)
    pool = v9.candidates(context, scores, signals, SMALL)
    pd.testing.assert_frame_equal(v9.features(context, scores, signals, pool, als, gnn), expected)
    assert set(v9.FEATURES) <= set(expected.columns)


def test_training_events_reaching_the_label_window_are_refused(prepared):
    context = EvaluationContext(prepared, 'repeat')
    leaked = context.train.iloc[[0]].assign(timestamp=int(context.manifest['cutoff_timestamp']))
    context.train = pd.concat([context.train, leaked], ignore_index=True)
    with pytest.raises(RuntimeError, match='reach the label cutoff'):
        v9.CoVisitGraph(context)


def test_covisit_graph_links_items_touched_together_within_the_window(prepared):
    context = EvaluationContext(prepared, 'repeat')
    end = 10 * v9.HOUR_MS
    a, b, c, d = 0, 1, 2, 3
    context.train = pd.DataFrame({
        'visitorid': [7, 7, 7, 8, 8], 'itemid': context.ids[[a, b, c, a, d]],
        'event': ['view', 'addtocart', 'view', 'view', 'view'],
        'timestamp': [end - 5000 - 2 * v9.HOUR_MS, end - 4000 - 2 * v9.HOUR_MS, end - 4000, end, end - 10]})
    context.manifest = {**context.manifest, 'cutoff_timestamp': end + 1}
    graph = v9.CoVisitGraph(context)
    m = graph.matrix.toarray()
    assert np.allclose(m, m.T) and not np.diag(m).any()
    assert m[a, b] > 0 and m[a, d] > 0 and m[b, c] == 0 and m[a, c] == 0
    raw = {(a, b): (1 + 6) / 2 * 0.5 ** ((4000 + 2 * v9.HOUR_MS) / v9.DAY_MS / 7), (a, d): 1. * 0.5 ** 0}
    degree = {a: raw[(a, b)] + raw[(a, d)], b: raw[(a, b)], d: raw[(a, d)]}
    assert m[a, b] == pytest.approx(raw[(a, b)] / np.sqrt(degree[a] * degree[b]))
    assert m[a, d] == pytest.approx(raw[(a, d)] / np.sqrt(degree[a] * degree[d]))


def test_conversion_graph_counts_positive_events_that_follow_a_touch(prepared):
    context = EvaluationContext(prepared, 'repeat')
    end = 100 * v9.HOUR_MS
    a, b, c = 0, 1, 2
    context.train = pd.DataFrame({
        'visitorid': [7, 7, 7, 8, 8, 9, 9], 'itemid': context.ids[[a, b, c, a, b, a, c]],
        'event': ['view', 'addtocart', 'view', 'view', 'view', 'view', 'transaction'],
        'timestamp': [end - 50, end - 40, end - 30, end - 20, end - 10, end - 30 * v9.HOUR_MS, end]})
    context.manifest = {**context.manifest, 'cutoff_timestamp': end + 1}
    graph = v9.ConversionGraph(context, prior=5.)
    m = graph.matrix.toarray()
    decay = lambda t: 0.5 ** ((end - t) / v9.DAY_MS / 7)
    # pairs leaving a: (a,b) and (a,c) of user 7, (a,b) of user 8; user 9's purchase is more than a day later
    out_a = decay(end - 40) + decay(end - 30) + decay(end - 10)
    assert m[a, b] == pytest.approx(decay(end - 40) / (out_a + 5.))
    assert m[a, c] == 0 and m[b, a] == 0 and m[c, a] == 0
    assert m[b, c] == 0  # c is only viewed after b


@pytest.mark.parametrize('policy', ['repeat', 'novel'])
def test_module_pools_follow_the_policy_and_the_flags(prepared, policy):
    context, scores, signals, pool, (als, gnn) = setup(prepared, policy)
    frame = v9.features(context, scores, signals, pool, als, gnn)
    for uid, parts in pool.items():
        union = np.unique(np.concatenate(list(parts.values())))
        if policy == 'novel':
            assert not np.isin(union, context.seen_idx.get(uid, [])).any()
        rows = frame.loc[frame.uid == uid]
        assert rows.idx.tolist() == union.tolist()
        for m in v9.MODULES:
            assert set(rows.idx[rows[f'in_{m}']]) == set(parts[m].tolist())
    if policy == 'repeat':
        assert all(np.isin(signals.history.recent[u][:5], pool[u]['context']).all() for u in pool)


def test_ablations_drop_the_features_and_the_candidates_of_a_module(prepared):
    context, scores, signals, pool, (als, gnn) = setup(prepared, 'repeat')
    frame = v9.features(context, scores, signals, pool, als, gnn)
    columns = v9.module_features(('mf', 'graph'))
    assert not set(columns) & set(v9.FEATURE_GROUPS['context'])
    kept = frame.loc[v9.module_rows(frame, ('mf', 'graph'))]
    assert (kept.in_mf | kept.in_graph).all()
    assert v9.module_features(v9.MODULES) == v9.FEATURES


def test_category_share_is_the_users_history_weight_by_category(prepared):
    context, scores, signals, pool, (als, gnn) = setup(prepared, 'repeat')
    totals = {}
    for (uid, _), share in signals.user_cat_share.items():
        totals[uid] = totals.get(uid, 0.) + share
    assert totals and all(t == pytest.approx(1.) for t in totals.values())


def test_negative_sampling_keeps_every_positive_and_caps_negatives_per_user():
    frame = pd.DataFrame({'uid': np.repeat([1, 2, 3], [10, 4, 6]), 'idx': np.arange(20)})
    y = np.zeros(20, dtype=np.int8)
    y[[0, 5, 11, 12]] = 1
    mask = v9.sample_negatives(frame, y, 2, seed=3)
    assert mask[y == 1].all()
    kept_negatives = pd.Series(mask & (y == 0)).groupby(frame.uid).sum().to_dict()
    assert kept_negatives == {1: 2, 2: 2, 3: 2}
    assert np.array_equal(mask, v9.sample_negatives(frame, y, 2, seed=3))


@pytest.mark.parametrize('stretch', [1, 20_000])
def test_session_and_category_features_recount_the_training_events(prepared, stretch):
    context, scores, _, _, (als, gnn) = setup(prepared, 'repeat')
    context.train = context.train.assign(timestamp=context.train.timestamp * stretch)  # many sessions when stretched
    context.manifest = {**context.manifest, 'cutoff_timestamp': int(context.manifest['cutoff_timestamp']) * stretch}
    signals = v9.Signals(context)
    pool = v9.candidates(context, scores, signals, SMALL)
    frame = v9.features(context, scores, signals, pool, als, gnn)
    sessions = 0
    train = context.train.assign(idx=context.index.get_indexer(context.train.itemid))
    train['cat'] = signals.cat_index[train.idx]
    checked = 0
    for uid, rows in frame.groupby('uid'):
        events = train.loc[train.visitorid == uid].sort_values('timestamp', kind='stable')
        breaks = np.flatnonzero(np.diff(events.timestamp.to_numpy()) > v9.SESSION_GAP_MS)
        last = events.iloc[breaks[-1] + 1:] if len(breaks) else events
        sessions += len(breaks) + 1
        expected = last.idx.value_counts().reindex(rows.idx).fillna(0).to_numpy()
        assert np.allclose(rows.s_events_last, np.log1p(expected))
        weight = signals.history.pairs.loc[uid].weight
        for _, row in rows.loc[rows.h_seen == 1].iterrows():
            cat = signals.cat_index[int(row.idx)]
            same = weight[[i for i in weight.index if signals.cat_index[i] == cat]]
            assert row.uc_item_share == pytest.approx(weight[int(row.idx)] / same.sum())
            assert row.uc_items == pytest.approx(np.log1p(len(same)))
            checked += 1
        assert (rows.loc[rows.h_seen == 0].uc_item_share == 0).all()
    assert checked
    assert (sessions > frame.uid.nunique()) == (stretch > 1)


def test_grades_follow_the_strongest_label_event():
    y = np.array([1, 1, 1, 0, 0])
    assert v9.grades(y, np.array([1., 6., 12., 1., 1.])).tolist() == [1, 2, 3, 0, 0]
    assert v9.grades(y).tolist() == [1, 1, 1, 0, 0]


def test_lambdarank_ranker_is_deterministic_and_lists_are_valid(prepared):
    context, scores, signals, pool, (als, gnn) = setup(prepared, 'novel')
    frame = v9.features(context, scores, signals, pool, als, gnn)
    y = ltr.labels(context, frame)
    params = dict(learning_rate=.1, max_iter=20, max_leaf_nodes=7)
    first, second = (v9.rank(v9.fit(frame, y, v9.FEATURES, params, 'lambdarank'), frame, v9.FEATURES, 4)
                     for _ in range(2))
    assert {u: v.tolist() for u, v in first.items()} == {u: v.tolist() for u, v in second.items()}
    context.metrics(first, 4)
    shuffled = frame.sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match='contiguous'):
        v9.fit(shuffled, y[shuffled.index.to_numpy()], v9.FEATURES, params, 'lambdarank')


def test_covisit_ranking_respects_the_novel_exclusions(prepared):
    context, scores, signals, pool, _ = setup(prepared, 'novel')
    rankings = v9.covisit_ranking(context, signals, 4)
    context.metrics(rankings, 4)  # raises on a seen item or a duplicate
