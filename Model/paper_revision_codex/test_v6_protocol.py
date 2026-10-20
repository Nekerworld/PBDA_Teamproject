"""Regression tests for protocol v6: policies, baselines, statistics, guards."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_protocol import UnifiedPreprocessor, positive_sets
from policy_evaluation import (EvaluationContext, FactorScores, candidate_indices, hybrid_ranking, top_k)
from ranking_evaluation import ranking_metrics
from recommendation_scoring import top_indices, recommendation_frame
import improve_hybrid
import learning
import rolling_protocol
import stats_tools
from audit_tools import leakage_check
from test_revision import make_raw


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('v6')
    raw, out = tmp / 'raw', tmp / 'processed'
    make_raw(raw)
    UnifiedPreprocessor(raw, out, n_estimators=20).run()
    return raw, out


def random_vectors(context, seed=0):
    rng = np.random.default_rng(seed)
    return {u: rng.normal(size=len(context.ids)) for u in context.active}


def test_top_k_matches_reference_tie_rule_and_exclusion():
    rng = np.random.default_rng(1)
    ids = np.arange(100, 400, 3)
    for _ in range(50):
        scores = rng.integers(0, 5, size=len(ids)).astype(float)
        assert np.array_equal(top_k(scores, 20), top_indices(scores, ids, 20))
    scores = np.arange(10, dtype=float)
    picked = top_k(scores, 5, exclude=np.array([9, 8]))
    assert picked.tolist() == [7, 6, 5, 4, 3]
    assert top_k(scores, 20, exclude=np.arange(7)).tolist() == [9, 8, 7]
    assert top_k(scores, 0).tolist() == [] and top_k(scores[:0], 5).tolist() == []
    with pytest.raises(ValueError):
        top_k(np.array([1., np.nan]), 1)
    candidates = np.array([5, 2, 9])
    assert top_k(np.ones(3), 3, tie=candidates).tolist() == [1, 0, 2]


def test_repeat_metrics_equal_reference_implementation(bundle):
    _, out = bundle
    context = EvaluationContext(out, 'repeat')
    ranked = context.rank_scores(random_vectors(context), 5)
    records = [(u, int(context.ids[i]), 0., r) for u, items in ranked.items() for r, i in enumerate(items, 1)]
    reference, _ = ranking_metrics(recommendation_frame(records), context.truth, 5)
    ours = context.metrics(ranked, 5)
    for name in ('ndcg', 'recall', 'precision', 'hit_rate'):
        assert ours[name] == pytest.approx(reference[name])


def test_novel_policy_definitions(bundle):
    _, out = bundle
    repeat, novel = EvaluationContext(out, 'repeat'), EvaluationContext(out, 'novel')
    raw = pd.read_csv(out / 'train_event_audit.csv')
    for uid, items in novel.truth.items():
        seen = set(raw.loc[raw.visitorid == uid, 'itemid'])
        assert items == repeat.truth[uid] - seen and items
    assert set(novel.users) <= set(repeat.users)
    ranked = novel.rank_scores(random_vectors(novel), 50)
    for uid, items in ranked.items():
        assert not np.isin(items, novel.seen_idx.get(uid, [])).any()
    uid = next(u for u in novel.active if len(novel.seen_idx.get(u, [])))
    with pytest.raises(AssertionError):
        novel.details({uid: novel.seen_idx[uid][:1]}, 5)


def test_recent_and_popularity_baselines(bundle):
    _, out = bundle
    context = EvaluationContext(out, 'repeat')
    train = context.train
    counts = train.groupby('itemid').size().reindex(context.ids).fillna(0).to_numpy()
    pop = context.popularity(5)
    assert all(np.array_equal(v, top_k(counts, 5)) for v in pop.values())
    recent = context.recent(50)
    filled = context.recent(50, fill=True)
    for uid in context.active:
        rows = train.loc[train.visitorid == uid].groupby('itemid').timestamp.max().reset_index()
        rows = rows.sort_values(['timestamp', 'itemid'], ascending=[False, True])
        expected = context.index.get_indexer(rows.itemid)[:50]
        assert recent[uid].tolist() == expected.tolist()
        assert len(recent[uid]) == min(50, len(rows))  # fewer than 50 items: shorter list, no fill
        assert len(set(filled[uid].tolist())) == len(filled[uid])
        assert filled[uid][:len(recent[uid])].tolist() == recent[uid].tolist()
        assert len(filled[uid]) == min(50, len(context.ids))
    novel = EvaluationContext(out, 'novel')
    assert all(len(v) == 0 for v in novel.recent(50).values())
    for uid, items in novel.recent(50, fill=True).items():
        assert not np.isin(items, novel.seen_idx.get(uid, [])).any()


def test_candidates_and_hybrid_match_v5_implementation(bundle):
    _, out = bundle
    context = EvaluationContext(out, 'repeat')
    als, bpr, gnn = random_vectors(context, 1), random_vectors(context, 2), random_vectors(context, 3)
    scores = FactorScores(context, als, bpr, gnn)
    cache = SimpleNamespace(ids=context.ids, active=context.active, als=scores.als, bpr=scores.bpr,
                            gnn=scores.gnn, available=context.available,
                            mf=lambda u, a: a * scores.als[u] + (1 - a) * scores.bpr[u])
    for strategy in ('blend', 'balanced_union'):
        for count in (3, 7, len(context.ids)):
            for uid in context.active:
                ours = candidate_indices(scores, uid, .5, count, strategy)
                theirs = improve_hybrid.candidate_indices(cache, uid, .5, count, strategy)
                assert ours.tolist() == theirs.tolist()
            for fusion in ('minmax', 'rrf'):
                config = dict(alpha=.5, candidate_k=count, strategy=strategy, fusion=fusion, omega=.3, gamma=1.25)
                ours = hybrid_ranking(scores, config, 5)
                old = improve_hybrid.config_ranking(cache, config)
                assert all(ours[u].tolist() == old[u][:5].tolist() for u in context.active)


def test_holm_and_paired_tests():
    adjusted = stats_tools.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    rng = np.random.default_rng(0)
    strong = rng.normal(1, .1, 200)
    assert stats_tools.sign_flip_pvalue(strong, 2000) < .01
    assert stats_tools.sign_flip_pvalue(np.zeros(10), 2000) == 1.0
    noise = rng.normal(0, 1, 200)
    assert stats_tools.sign_flip_pvalue(noise - noise.mean(), 2000) > .5
    mean, low, high = stats_tools.paired_bootstrap_ci(strong, 2000)
    assert low < mean < high
    users = pd.Index(range(50), name='visitorid')
    frames = {m: pd.DataFrame({'ndcg': rng.random(50), 'recall': rng.random(50)}, index=users) for m in 'ABC'}
    rows = stats_tools.compare_family(frames, 'A', ['B', 'C'], samples=500)
    assert len(rows) == 4 and all(r['p_holm'] >= r['p_value'] for r in rows)


def test_bpr_checkpoints_do_not_change_training():
    import scipy.sparse as sp
    matrix = sp.random(30, 40, density=.2, random_state=0, format='csr')
    matrix.data = matrix.data + 1
    plain = learning.train_weighted_bpr(matrix, 8, .01, 6, 7, 'cpu', batch_size=16)
    seen = {}
    checked = learning.train_weighted_bpr(matrix, 8, .01, 6, 7, 'cpu', batch_size=16, checkpoint_epochs=(3, 6),
                                          checkpoint_callback=lambda e, u, i: seen.__setitem__(e, (u, i)))
    assert np.array_equal(plain.user_factors, checked.user_factors)
    short = learning.train_weighted_bpr(matrix, 8, .01, 3, 7, 'cpu', batch_size=16)
    assert np.array_equal(short.user_factors, seen[3][0])
    faster = learning.train_weighted_bpr(matrix, 8, .01, 6, 7, 'cpu', batch_size=16, learning_rate=.01)
    assert not np.array_equal(plain.user_factors, faster.user_factors)


def test_gnn_validation_metric_option():
    from model import GNNEmbeddingGenerator
    assert GNNEmbeddingGenerator(device='cpu').validation_metric == 'ndcg'
    with pytest.raises(ValueError):
        GNNEmbeddingGenerator(device='cpu', validation_metric='auc')


def test_checkpoint_loading_only_tolerates_missing_adjacency():
    import torch
    from model import load_without_adjacency

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(3, 2)
            self.register_buffer('adjacency', torch.zeros(1))
    net = Net()
    state = {k: v for k, v in Net().state_dict().items() if k != 'adjacency'}
    load_without_adjacency(net, state)
    assert torch.equal(net.embedding.weight, state['embedding.weight'])
    with pytest.raises(RuntimeError):
        load_without_adjacency(net, {})
    with pytest.raises(RuntimeError):
        load_without_adjacency(net, {**state, 'extra': torch.zeros(1)})


def test_robust_hybrid_backbones_are_explicit():
    import robust_hybrid
    assert robust_hybrid.HISTORICAL_V5_GNN in {n for n, _ in robust_hybrid.HISTORICAL_V5_BACKBONES}
    assert 'after the v3 sealed test' in robust_hybrid.BACKBONE_PROVENANCE


def test_gnn_cap_rule_flags_v5_pattern():
    assert rolling_protocol.at_cap(190, 200, 10)
    assert rolling_protocol.at_cap(400, 400, 10)
    assert not rolling_protocol.at_cap(380, 400, 10)


def test_access_log_blocks_reopening_under_new_plan(tmp_path):
    entry = {'utc': 'x', 'fold': 'F1', 'purpose': 'sealed_test', 'plan_sha256': 'a', 'selection_sha256': 's', 'git': 'g'}
    rolling_protocol.access_log(tmp_path, entry)
    rolling_protocol.access_log(tmp_path, entry)  # same plan may resume
    with pytest.raises(RuntimeError):
        rolling_protocol.access_log(tmp_path, {**entry, 'plan_sha256': 'b'})
    assert len((tmp_path / 'test_access_log.jsonl').read_text().splitlines()) == 2


def test_fold_truncation_is_tie_safe(bundle, tmp_path):
    raw, _ = bundle
    info = rolling_protocol.make_fold_raw(raw, tmp_path / 'fold', .6)
    full = pd.read_csv(raw / 'events.csv')
    kept = pd.read_csv(tmp_path / 'fold' / 'events.csv')
    assert kept.timestamp.max() < info['end_cutoff_timestamp'] <= full.loc[~full.timestamp.isin(kept.timestamp)].timestamp.min()
    assert abs(len(kept) / len(full) - .6) < .01


def test_leakage_check_passes_and_detects_injected_leak(bundle, tmp_path):
    raw, out = bundle
    assert leakage_check.main(out, raw, gnn=True)
    broken = tmp_path / 'broken'
    shutil.copytree(out, broken)
    train = pd.read_csv(broken / 'events_train_clean.csv')
    test = pd.read_csv(broken / 'events_test_original.csv')
    pd.concat([train, test.iloc[:1][train.columns]]).to_csv(broken / 'events_train_clean.csv', index=False)
    assert not leakage_check.main(broken, raw)


def test_plan_freezes_grids_and_rules():
    plan = rolling_protocol.plan_payload()
    assert plan['selection_rule'].startswith('maximize mean(NDCG@50, Recall@50)')
    assert plan['gnn_max_epochs'] == 400 and 'all' in plan['candidate_ks']
    assert any(c == dict(factors=128, regularization=.02, alpha=2., iterations=256) for c in plan['als_grid'])
    assert any(c == dict(layers=3, embedding_dim=64, learning_rate=1e-3) for c in plan['gnn_grid'])
    assert set(rolling_protocol.BASELINES) >= {'Popularity', 'Recent', 'RecentPop', 'GNN'}
