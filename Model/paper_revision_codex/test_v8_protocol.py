"""Protocol v8: the v7 design with the stacked Hybrid re-rankers, without changing the sealed v6/v7 plans."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr
import rolling_protocol as rp
import stats_tools
from test_v6_smoke import make_raw, tiny_settings

RESULTS = HERE / 'audit'
LTR_MODELS = {'Hybrid_LTR', 'Hybrid_LTR_W'} | {f'{m}_{v}' for m in ('Hybrid_LTR', 'Hybrid_LTR_W') for v in ltr.ABLATIONS}


def plan_hash():
    return hashlib.sha256(json.dumps(rp.plan_payload(), sort_keys=True).encode()).hexdigest()


def use(monkeypatch, name):
    for key, value in rp.PROTOCOLS[name].items():
        monkeypatch.setattr(rp, key, value)


def tiny_v8(monkeypatch):
    use(monkeypatch, 'v8')
    tiny_settings(monkeypatch)
    monkeypatch.setattr(ltr, 'PARAM_GRID', [dict(learning_rate=.1, max_iter=5, max_leaf_nodes=4),
                                            dict(learning_rate=.1, max_iter=10, max_leaf_nodes=4)])
    monkeypatch.setattr(ltr, 'POOL', dict(per_model=6, history=6, popular=4))
    monkeypatch.setattr(rp, 'RERANKER_CV_FOLDS', 2)
    monkeypatch.setattr(rp, 'gnn_device', lambda: 'cpu')  # CUDA scatter-adds are not bit-reproducible


@pytest.mark.parametrize('name', ['v6', 'v7'])
def test_v8_code_keeps_the_sealed_v6_and_v7_plans(monkeypatch, name):
    use(monkeypatch, name)
    stored = json.loads((RESULTS / f'results_{name}' / 'plan.json').read_text(encoding='utf-8'))
    assert plan_hash() == stored['plan_sha256'] and 'reranker' not in rp.plan_payload()


def test_v8_plan_records_the_rerankers_and_one_holm_family(monkeypatch):
    use(monkeypatch, 'v8')
    plan = rp.plan_payload()
    assert plan['protocol'] == 'rolling-origin-v8' and plan['posthoc_test_analyses'] == 'none'
    assert [f['name'] for f in plan['folds']] == ['TB1_full'] and plan['folds'][0]['end_fraction'] == 1.0
    assert plan['als_grid'] == rp.V7['ALS_GRID'] and plan['gnn_grid'] == rp.V7['GNN_GRID']
    assert plan['bpr_grid'] == rp.V7['BPR_GRID'] and plan['hybrid_grid_size'] == len(rp.HYBRID_GRID)
    assert plan['reranker']['param_grid'] == ltr.PARAM_GRID and set(plan['reranker']['models'].values()) == {'Hybrid_LTR', 'Hybrid_LTR_W'}
    assert set(plan['reranker']['ablations']) == LTR_MODELS - {'Hybrid_LTR', 'Hybrid_LTR_W'}
    assert rp.HYBRIDS == ['Hybrid', 'Hybrid_LTR', 'Hybrid_LTR_W']
    assert 'Hybrid, Hybrid_LTR and Hybrid_LTR_W each vs every baseline' in plan['statistics']


def test_compare_families_is_one_holm_family():
    rng = np.random.default_rng(0)
    users = pd.Index(range(300), name='visitorid')
    frames = {m: pd.DataFrame({'ndcg': rng.random(300) + shift, 'recall': rng.random(300)}, index=users)
              for m, shift in (('A', .05), ('B', 0.), ('C', .02), ('D', 0.))}
    single = stats_tools.compare_family(frames, 'A', ['B', 'D'], samples=500)
    assert stats_tools.compare_families(frames, ['A'], ['B', 'D'], samples=500) == single
    rows = stats_tools.compare_families(frames, ['A', 'C'], ['B', 'D'], samples=500)
    assert len(rows) == 8
    assert np.allclose([r['p_holm'] for r in rows], stats_tools.holm([r['p_value'] for r in rows]))
    separate = stats_tools.compare_family(frames, 'C', ['B', 'D'], samples=500)
    assert [r['p_value'] for r in rows[4:]] == [r['p_value'] for r in separate]
    assert all(a['p_holm'] >= b['p_holm'] for a, b in zip(rows[4:], separate))


def scramble_test_period(raw, end_fraction=1., test_size=.25):
    """Same raw input before the outer test cutoff; the test-period events point at other items."""
    events = pd.read_csv(raw / 'events.csv')
    order = np.lexsort((np.arange(len(events)), events.timestamp.to_numpy()))
    cutoff = events.timestamp.to_numpy()[order[int(len(events) * (1 - test_size))]]
    late = events.timestamp >= cutoff
    events.loc[late, 'itemid'] = np.random.default_rng(5).permutation(events.loc[late, 'itemid'].to_numpy())
    events.loc[late, 'event'] = np.random.default_rng(6).permutation(events.loc[late, 'event'].to_numpy())
    events.to_csv(raw / 'events.csv', index=False)
    return int(late.sum())


def test_v8_end_to_end_seals_rerankers_before_the_test(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw, users=60, events=9000)
    tiny_v8(monkeypatch)
    root = tmp_path / 'run'
    rp.run(root, raw)
    fold_dir = root / 'TB1_full'
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']
    assert plan['protocol'] == 'rolling-origin-v8' and plan['reranker']['param_grid'] == ltr.PARAM_GRID
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    for policy in ('repeat', 'novel'):
        models = selection['rerankers'][policy]['models']
        assert set(models) == LTR_MODELS
        for name, entry in models.items():
            assert entry['sha256'] == rp.sha256(fold_dir / 'rerankers' / entry['model'])
            full = 'Hybrid_LTR_W' if name.startswith('Hybrid_LTR_W') else 'Hybrid_LTR'
            assert entry['params'] == models[full]['params'] and entry['params'] in ltr.PARAM_GRID
        assert set(selection['rerankers'][policy]['training']) == {'Hybrid_LTR', 'Hybrid_LTR_W'}
        for name, (model, entry) in rp.load_rerankers(fold_dir, selection, policy).items():
            fitted = model.get_params()
            assert {k: fitted[k] for k in entry['params']} == entry['params']
            assert model.n_features_in_ == len(entry['features'])
        cv = pd.read_csv(fold_dir / 'rerankers' / f'cv_{policy}_Hybrid_LTR.csv')
        best = cv.sort_values('cv_score', ascending=False, kind='stable').iloc[0]
        assert {k: type(v)(best[k]) for k, v in models['Hybrid_LTR']['params'].items()} == models['Hybrid_LTR']['params']
    stages = [json.loads(line)['stage'] for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert stages.index('rerankers_trained') < stages.index('selection_sealed') < stages.index('test_evaluated')
    metrics = pd.read_csv(fold_dir / 'test_metrics.csv')
    assert set(metrics.model) >= {'Hybrid', 'ALS', 'BPR', 'ALS+BPR', 'GNN'} | LTR_MODELS
    tests = pd.read_csv(fold_dir / 'report' / 'paired_tests.csv')
    for policy, part in tests.groupby('policy'):
        confirmatory = part.loc[part.family == 'confirmatory']
        assert len(confirmatory) == 3 * len(rp.BASELINES) * 2
        assert set(confirmatory.comparison.str.split(' - ').str[0]) == {'Hybrid', 'Hybrid_LTR', 'Hybrid_LTR_W'}
        assert np.allclose(confirmatory.p_holm, stats_tools.holm(confirmatory.p_value))
        secondary = part.loc[part.family == 'secondary']
        assert sorted(secondary.comparison.unique()) == ['Hybrid_LTR - Hybrid', 'Hybrid_LTR_W - Hybrid']
        assert np.allclose(secondary.p_holm, stats_tools.holm(secondary.p_value))
    access = (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['purpose'] for line in access] == ['sealed_test']
    stamp = (fold_dir / 'rerankers' / 'repeat.json').stat().st_mtime_ns
    rp.run(root, raw)  # resuming retrains nothing and opens nothing
    assert (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines() == access
    assert (fold_dir / 'rerankers' / 'repeat.json').stat().st_mtime_ns == stamp
    # A run interrupted after some policies were trained reuses their records instead of retraining them.
    files = sorted((fold_dir / 'rerankers').iterdir())
    stamps = [p.stat().st_mtime_ns for p in files]
    records = rp.train_rerankers(fold_dir, fold_dir / 'val_mf', selection['hybrid_backbone']['job_dir'])
    assert records == selection['rerankers']
    assert sorted((fold_dir / 'rerankers').iterdir()) == files and [p.stat().st_mtime_ns for p in files] == stamps

    # The outer test period never reaches the selection or the re-rankers.
    other_raw = tmp_path / 'raw_scrambled'
    shutil.copytree(raw, other_raw)
    assert scramble_test_period(other_raw) > 0
    other = tmp_path / 'run_scrambled'
    rp.run(other, other_raw)
    other_dir = other / 'TB1_full'
    again = json.loads((other_dir / 'selection.json').read_text(encoding='utf-8'))
    for policy in ('repeat', 'novel'):
        assert again['policies'][policy]['hybrid'] == selection['policies'][policy]['hybrid']
        for name, entry in selection['rerankers'][policy]['training'].items():
            other_entry = again['rerankers'][policy]['training'][name]
            assert {k: v for k, v in entry.items() if k != 'seconds'} == {k: v for k, v in other_entry.items() if k != 'seconds'}
            cv = [(d / 'rerankers' / f'cv_{policy}_{name}.csv').read_bytes() for d in (fold_dir, other_dir)]
            assert cv[0] == cv[1]
        for name, entry in selection['rerankers'][policy]['models'].items():
            assert again['rerankers'][policy]['models'][name] == entry  # same model file bytes (sha256)
    assert not pd.read_csv(fold_dir / 'test_metrics.csv').equals(pd.read_csv(other_dir / 'test_metrics.csv'))


def test_load_rerankers_refuses_a_replaced_model(tmp_path):
    fold_dir = tmp_path / 'fold'
    (fold_dir / 'rerankers').mkdir(parents=True)
    path = fold_dir / 'rerankers' / 'repeat_Hybrid_LTR.pkl'
    import pickle
    path.write_bytes(pickle.dumps({'model': 1}))
    selection = {'rerankers': {'repeat': {'models': {'Hybrid_LTR': {'model': path.name, 'sha256': rp.sha256(path),
                                                                    'features': ltr.FEATURES}}}}}
    assert rp.load_rerankers(fold_dir, selection, 'repeat')['Hybrid_LTR'][0] == {'model': 1}
    path.write_bytes(pickle.dumps({'model': 2}))
    with pytest.raises(RuntimeError, match='not the re-ranker sealed'):
        rp.load_rerankers(fold_dir, selection, 'repeat')


def test_selection_decision_covers_the_rerankers(tmp_path):
    base = {'plan_sha256': 'p', 'fold': 'f', 'mf': {'als': {}, 'bpr': {}, 'bpr_epochs': 1},
            'gnn': {'config': {}, 'epochs': 1}, 'hybrid_backbone': {'config': {}, 'epochs': 1},
            'policies': {'repeat': {'mf_alpha': .5, 'hybrid': {k: 1 for k in rp.HYBRID_DECISION}}}}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(base), encoding='utf-8')
    without = rp.selection_decision(path)
    entry = {'features': ['als'], 'params': {'max_iter': 5}, 'sha256': 'x', 'model': 'm.pkl'}
    path.write_text(json.dumps({**base, 'rerankers': {'repeat': {'models': {'Hybrid_LTR': entry}}}}), encoding='utf-8')
    with_rerankers = rp.selection_decision(path)
    path.write_text(json.dumps({**base, 'rerankers': {'repeat': {'models': {
        'Hybrid_LTR': {**entry, 'params': {'max_iter': 10}}}}}}), encoding='utf-8')
    assert len({without, with_rerankers, rp.selection_decision(path)}) == 3
