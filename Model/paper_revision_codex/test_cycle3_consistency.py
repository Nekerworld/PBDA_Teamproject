"""Selection-time and test-time scoring agree, paired statistics align users by ID,
selection takes the validation maximum, and inventory flags use the last value before the cutoff."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rolling_protocol as rp
import stats_tools
from data_protocol import UnifiedPreprocessor
from policy_evaluation import POLICIES, EvaluationContext, FactorScores, hybrid_ranking
from test_v6_smoke import make_raw


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    """Synthetic data whose `available` flag changes before the cutoffs and again after them."""
    tmp = tmp_path_factory.mktemp('cycle3')
    raw, out = tmp / 'raw', tmp / 'processed'
    make_raw(raw)
    props = pd.read_csv(raw / 'item_properties_part2.csv')
    flips = pd.DataFrame([(500_000, i, 'available', str(1 - i % 2)) for i in range(1, 11)]
                         + [(999_999, i, 'available', '0') for i in range(1, 31)], columns=props.columns)
    pd.concat([props, flips]).to_csv(raw / 'item_properties_part2.csv', index=False)
    UnifiedPreprocessor(raw, out, n_estimators=20).run()
    return raw, out


def test_novel_truth_removes_items_seen_only_before_core_filtering(tmp_path):
    """An item the user saw in training but the core filter dropped is still 'seen' for the novel policy."""
    raw, out = tmp_path / 'raw', tmp_path / 'processed'
    make_raw(raw)
    events = pd.read_csv(raw / 'events.csv')
    user, rare = int(events.visitorid.iloc[-1]), 777
    extra = pd.DataFrame({'timestamp': [5_000, 999_990], 'visitorid': [user, user], 'event': ['view', 'transaction'],
                          'itemid': [rare, rare], 'transactionid': [np.nan, np.nan]})
    pd.concat([events, extra]).sort_values('timestamp', kind='stable').to_csv(raw / 'events.csv', index=False)
    UnifiedPreprocessor(raw, out, n_estimators=20).run()
    train = pd.read_csv(out / 'events_train_clean.csv')
    audit = pd.read_csv(out / 'train_event_audit.csv')
    assert rare not in set(train.itemid) and rare in set(audit.loc[audit.visitorid == user, 'itemid'])
    repeat, novel = EvaluationContext(out, 'repeat'), EvaluationContext(out, 'novel')
    assert rare in repeat.truth[user]
    assert rare not in novel.truth.get(user, set())


def random_vectors(context, seed):
    rng = np.random.default_rng(seed)
    return {u: rng.normal(size=len(context.ids)) for u in context.active}


def test_available_flag_is_last_value_before_cutoff(prepared):
    raw, out = prepared
    props = pd.concat([pd.read_csv(raw / f'item_properties_part{i}.csv', dtype={'value': str}) for i in (1, 2)])
    for directory in (out, out / 'validation'):
        context = EvaluationContext(directory, 'repeat')
        cutoff = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['cutoff_timestamp']
        past = props.loc[(props.property == 'available') & (props.timestamp < cutoff)]
        last = past.sort_values('timestamp', kind='stable').drop_duplicates('itemid', keep='last')
        expected = last.set_index('itemid').value.astype(float).reindex(context.ids).fillna(0).eq(1).to_numpy()
        assert 0 < expected.sum() < len(expected)
        assert np.array_equal(context.available, expected)


def test_metrics_match_hand_computation_with_unreachable_truth_and_cold_users(prepared):
    """Ideal DCG and Recall count out-of-catalog truth; users without a list score zero but stay in the mean."""
    _, out = prepared
    context = EvaluationContext(out, 'repeat')
    ids = context.ids
    warm, cold = 10_001, 10_002
    context.users, context.active = [warm, cold], [warm]
    context.truth = {warm: {int(ids[0]), int(ids[1]), -5}, cold: {int(ids[2])}}
    context.truth_idx = {warm: {0, 1}, cold: {2}}
    frame = context.details({warm: np.array([1, 5, 0])}, k=3)
    assert frame.visitorid.tolist() == [warm, cold]
    dcg, idcg = 1 + 1 / np.log2(4), 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert frame.ndcg.tolist() == pytest.approx([dcg / idcg, 0.])
    assert frame.recall.tolist() == pytest.approx([2 / 3, 0.])
    assert frame.precision.tolist() == pytest.approx([2 / 3, 0.])
    metrics = context.metrics({warm: np.array([1, 5, 0])}, k=3)
    assert metrics['users'] == 2 and metrics['hit_users'] == 1
    assert metrics['ndcg'] == pytest.approx(dcg / idcg / 2) and metrics['recall'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('policy', POLICIES)
def test_selection_grid_scores_equal_test_time_ranking(prepared, policy, monkeypatch):
    """rolling_protocol.hybrid_grid (validation selection) must score exactly like hybrid_ranking (sealed test)."""
    _, out = prepared
    context = EvaluationContext(out / 'validation', policy)
    assert context.available.any() and not context.available.all()
    als, bpr, gnn = (random_vectors(context, seed) for seed in (1, 2, 3))
    grid = [dict(alpha=a, strategy=s, candidate_k=c, fusion=f, omega=w, gamma=g)
            for a in (.25, .75) for c, s in ((3, 'blend'), (5, 'balanced_union'), ('all', 'blend'))
            for f in ('minmax', 'rrf') for w in (.2, .8) for g in (1., 1.5)]
    monkeypatch.setattr(rp, 'HYBRID_GRID', grid)
    rows = rp.hybrid_grid(context, als, bpr, gnn, 'backbone')
    scores = FactorScores(context, als, bpr, gnn)
    assert len(rows) == len(grid)
    distinct = set()
    for row, config in zip(rows, grid):
        assert {k: row[k] for k in config} == config
        expected = context.metrics(hybrid_ranking(scores, config, rp.PRIMARY_K), rp.PRIMARY_K)
        for name, value in expected.items():
            assert row[name] == pytest.approx(value, abs=1e-12), (config, name)
        distinct.add(round(row['score'], 12))
    assert len(distinct) > 1


def test_paired_statistics_align_users_by_id_not_position():
    rng = np.random.default_rng(0)
    users = pd.Index(np.arange(100, 300), name='visitorid')
    base = pd.DataFrame({'ndcg': rng.random(200), 'recall': rng.random(200)}, index=users)
    shuffled = base.sample(frac=1, random_state=1)
    for row in stats_tools.compare_family({'A': base, 'B': shuffled}, 'A', ['B'], samples=500):
        assert row['difference'] == 0 and row['users_tied'] == 200
        assert row['p_value'] == 1.0 and row['claim'] == 'not significant'
    shifted = (base - .01).sample(frac=1, random_state=2)
    for row in stats_tools.compare_family({'A': base, 'B': shifted}, 'A', ['B'], samples=500):
        assert row['difference'] == pytest.approx(.01) and row['users_better'] == 200 and row['significant_holm']
    with pytest.raises(ValueError):
        stats_tools.compare_family({'A': base, 'B': base.iloc[1:]}, 'A', ['B'], samples=100)


def test_sign_flip_is_two_sided():
    d = np.random.default_rng(3).normal(1, .1, 100)
    assert stats_tools.sign_flip_pvalue(-d, 2000) < .01
    assert stats_tools.sign_flip_pvalue(d, 2000) == stats_tools.sign_flip_pvalue(-d, 2000)
    # Exact: 12 of the 32 sign patterns of [1, 1, 1, 1, -1] have |sum| >= 3 (one-sided: 6 of 32).
    assert stats_tools.sign_flip_pvalue(np.array([1., 1., 1., 1., -1.]), 20000) == pytest.approx(12 / 32, abs=.02)


def test_code_hashes_ignore_line_endings_but_not_content(tmp_path):
    files = {'model.py': b'x = 1\nprint(x)\n', 'audit_tools/check.py': b'y = 2\n'}
    roots = {}
    for name, convert in (('lf', lambda b: b), ('crlf', lambda b: b.replace(b'\n', b'\r\n'))):
        roots[name] = tmp_path / name
        for relative, text in files.items():
            (roots[name] / relative).parent.mkdir(parents=True, exist_ok=True)
            (roots[name] / relative).write_bytes(convert(text))
    lf, crlf = rp.code_hashes(roots['lf']), rp.code_hashes(roots['crlf'])
    assert set(lf) == set(files) and lf == crlf
    assert rp.code_hashes(roots['lf'], 'sha256-raw') != rp.code_hashes(roots['crlf'], 'sha256-raw')
    (roots['crlf'] / 'model.py').write_bytes(b'x = 2\r\nprint(x)\r\n')
    assert rp.code_hashes(roots['crlf'])['model.py'] != lf['model.py']
    with pytest.raises(ValueError):
        rp.code_hashes(roots['lf'], 'md5')


@pytest.mark.parametrize('scheme', ['sha256-lf', 'sha256-raw'])
def test_run_accepts_unchanged_code_and_refuses_changed_code(tmp_path, monkeypatch, scheme):
    """New plans record the scheme; plans without it (v6, v7) are checked against raw-byte hashes."""
    monkeypatch.setattr(rp, 'write_reports', lambda root: None)
    plan = rp.plan_payload()
    stored = {'plan_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), 'plan': plan,
              'code_sha256': rp.code_hashes(scheme=scheme)}
    if scheme == 'sha256-lf':
        stored['code_hash_scheme'] = scheme
    (tmp_path / 'plan.json').write_text(json.dumps(stored), encoding='utf-8')
    rp.run(tmp_path, tmp_path, folds=['no-fold'])
    stored['code_sha256']['policy_evaluation.py'] = '0' * 64
    (tmp_path / 'plan.json').write_text(json.dumps(stored), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Code changed'):
        rp.run(tmp_path, tmp_path, folds=['no-fold'])


def test_best_row_takes_highest_score_and_first_on_ties():
    rows = [dict(name='a', score=.1), dict(name='b', score=.3), dict(name='c', score=.3), dict(name='d', score=.2)]
    assert rp.best_row(rows, ['name', 'score']) == {'name': 'b', 'score': .3}


def test_mf_validation_selects_the_validation_maximum(prepared, tmp_path, monkeypatch):
    _, out = prepared
    monkeypatch.setattr(rp, 'ALS_GRID', [dict(factors=f, regularization=.02, alpha=2., iterations=i)
                                         for f in (2, 8) for i in (1, 6)])
    monkeypatch.setattr(rp, 'BPR_GRID', [dict(factors=4, learning_rate=lr, regularization=.03) for lr in (1e-4, 5e-2)])
    monkeypatch.setattr(rp, 'BPR_EPOCH_CHECKPOINTS', (1, 4))
    validation = out / 'validation'
    record = rp.mf_validation(tmp_path, validation, {p: EvaluationContext(validation, p) for p in POLICIES})
    grid = pd.read_csv(tmp_path / 'validation_mf_grid.csv')
    for family in ('ALS', 'BPR'):
        part = grid.loc[grid.family == family]
        assert part.repeat_score.nunique() > 1
        assert record['validation'][family]['repeat_score'] == part.repeat_score.max()
    best = grid.loc[grid.family == 'BPR'].sort_values('repeat_score', ascending=False, kind='stable').iloc[0]
    assert record['bpr_epochs'] == int(best.epochs) and record['bpr']['learning_rate'] == best.learning_rate
