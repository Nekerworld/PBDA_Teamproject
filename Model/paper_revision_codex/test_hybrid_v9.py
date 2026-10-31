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

SMALL = dict(per_model=5, history=5, popular=3, category=3)


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


def test_covisit_ranking_respects_the_novel_exclusions(prepared):
    context, scores, signals, pool, _ = setup(prepared, 'novel')
    rankings = v9.covisit_ranking(context, signals, 4)
    context.metrics(rankings, 4)  # raises on a seen item or a duplicate
