"""Protocol v9: the module ladder of hybrid_v9 sealed before the test, without changing the v6/v7/v8 plans."""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr
import hybrid_v9 as v9
import rolling_protocol as rp
import stats_tools
from test_v6_smoke import make_raw, tiny_settings
from test_v8_protocol import RESULTS, plan_hash, scramble_test_period, use

SMALL = dict(per_model=6, covisit=5, covisit2=0, conversion=3, history=6, popular=4, category=3)
LADDER_MODELS = set(v9.RANKERS)


def tiny_v9(monkeypatch):
    use(monkeypatch, 'v9')
    tiny_settings(monkeypatch)
    monkeypatch.setattr(rp, 'HYBRID_BACKBONES', 1)
    monkeypatch.setattr(ltr, 'PARAM_GRID', [dict(learning_rate=.1, max_iter=5, max_leaf_nodes=4),
                                            dict(learning_rate=.1, max_iter=10, max_leaf_nodes=4)])
    monkeypatch.setitem(v9.DESIGNS, 'a02', dict(v9.DESIGNS['a02'], pool=SMALL))
    monkeypatch.setattr(rp, 'RERANKER_CV_FOLDS', 2)
    monkeypatch.setattr(rp, 'LADDER_NEGATIVES', 5)
    monkeypatch.setattr(rp, 'gnn_device', lambda: 'cpu')  # CUDA scatter-adds are not bit-reproducible


@pytest.mark.parametrize('name', ['v6', 'v7', 'v8'])
def test_v9_code_keeps_the_sealed_plans(monkeypatch, name):
    use(monkeypatch, name)
    stored = json.loads((RESULTS / f'results_{name}' / 'plan.json').read_text(encoding='utf-8'))
    assert plan_hash() == stored['plan_sha256'] and 'ladder' not in rp.plan_payload()
    assert not rp.LADDER and 'CoVisit' not in rp.BASELINES


def test_v9_plan_records_the_ladder(monkeypatch):
    use(monkeypatch, 'v9')
    plan = rp.plan_payload()
    assert plan['protocol'] == 'rolling-origin-v9' and plan['posthoc_test_analyses'] == 'none'
    assert [f['name'] for f in plan['folds']] == ['TB2_full'] and '[300, 500)' in plan['folds'][0]['test_window']
    v8 = json.loads((RESULTS / 'results_v8' / 'plan.json').read_text(encoding='utf-8'))['plan']
    for key in ('als_grid', 'bpr_grid', 'gnn_grid', 'gnn_fixed', 'gnn_max_epochs', 'mf_alphas', 'mf_common',
                'bpr_epoch_checkpoints', 'bpr_negatives', 'seeds', 'selection_seed', 'ks', 'preprocessing'):
        assert plan[key] == v8[key], key
    assert not {'hybrid_grid_size', 'hybrid_grid_axes', 'candidate_ks', 'hybrid_backbones', 'reranker'} & set(plan)
    ladder = plan['ladder']
    assert ladder['variant'] == 'a02' and ladder['pool'] == v9.DESIGNS['a02']['pool']
    assert ladder['dropped_features'] == ['cv2'] and ladder['param_grid'] == ltr.PARAM_GRID
    assert ladder['rankers'] == {n: list(m) for n, m in v9.RANKERS.items()}
    assert plan['baselines'] == rp.V8['BASELINES'] + ['CoVisit'] and rp.HYBRIDS == ['Hybrid']


def test_compare_pairs_is_one_holm_family():
    rng = np.random.default_rng(0)
    users = pd.Index(range(300), name='visitorid')
    frames = {m: pd.DataFrame({'ndcg': rng.random(300) + shift, 'recall': rng.random(300)}, index=users)
              for m, shift in (('A', .05), ('B', 0.), ('C', .02))}
    rows = stats_tools.compare_pairs(frames, [('A', 'B'), ('B', 'C')], samples=500)
    assert [r['comparison'] for r in rows] == ['A - B', 'A - B', 'B - C', 'B - C']
    assert np.allclose([r['p_holm'] for r in rows], stats_tools.holm([r['p_value'] for r in rows]))
    alone = stats_tools.compare_family(frames, 'B', ['C'], samples=500)
    assert [r['p_value'] for r in rows[2:]] == [r['p_value'] for r in alone]


def test_ladder_summary_applies_the_ten_percent_rule():
    base = {'ALS': .05, 'BPR': .04, 'GNN': .045, 'ALS+BPR': .06, 'Popularity': .01, 'CoVisit': .05}
    ok = v9.ladder_summary({**base, 'Hybrid_MF_Graph': .066, 'Hybrid': .0727}, list(base))
    assert ok['pass'] and ok['min_step'] == pytest.approx(1.1, abs=1e-3)
    assert not v9.ladder_summary({**base, 'Hybrid_MF_Graph': .0659, 'Hybrid': .08}, list(base))['pass']
    assert not v9.ladder_summary({**base, 'ALS+BPR': .049, 'Hybrid_MF_Graph': .06, 'Hybrid': .07}, list(base))['pass']
    assert not v9.ladder_summary({**base, 'Hybrid_MF_Graph': .07, 'Hybrid': .08, 'CoVisit': .09}, list(base))['pass']


def test_v9_end_to_end_seals_the_ladder_before_the_test(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw, users=60, events=9000)
    tiny_v9(monkeypatch)
    root = tmp_path / 'run'
    rp.run(root, raw)
    fold_dir = root / 'TB2_full'
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']
    assert plan['protocol'] == 'rolling-origin-v9' and plan['ladder']['pool'] == SMALL
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    assert selection['hybrid_backbone'] == selection['gnn'] and selection['budget']['evaluated_configs']['Hybrid'] == 0
    assert not list(fold_dir.glob('validation_hybrid_*.csv'))
    for policy in ('repeat', 'novel'):
        assert set(selection['policies'][policy]) == {'mf_alpha'}
        models = selection['ladder'][policy]['models']
        assert set(models) == LADDER_MODELS
        for name, entry in models.items():
            assert entry['sha256'] == rp.sha256(fold_dir / 'ladder' / entry['model'])
            assert entry['params'] in ltr.PARAM_GRID and entry['modules'] == list(v9.RANKERS[name])
            assert entry['features'] == v9.module_features(v9.RANKERS[name], ('cv2',))
            cv = pd.read_csv(fold_dir / 'ladder' / f'cv_{policy}_{name}.csv')
            best = cv.sort_values('cv_score', ascending=False, kind='stable').iloc[0]
            assert {k: type(v)(best[k]) for k, v in entry['params'].items()} == entry['params']
        for name, (model, entry) in rp.load_ladder(fold_dir, selection, policy).items():
            assert model.booster_.num_feature() == len(entry['features'])
    stages = [json.loads(line)['stage'] for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert stages.index('ladder_trained') < stages.index('selection_sealed') < stages.index('test_evaluated')
    metrics = pd.read_csv(fold_dir / 'test_metrics.csv')
    assert set(metrics.model) == set(rp.BASELINES) | LADDER_MODELS
    tests = pd.read_csv(fold_dir / 'report' / 'paired_tests.csv')
    for policy, part in tests.groupby('policy'):
        for family in ('confirmatory', 'ladder', 'ablation'):
            rows = part.loc[part.family == family]
            assert np.allclose(rows.p_holm, stats_tools.holm(rows.p_value))
        assert sorted(part.loc[part.family == 'confirmatory'].comparison.unique()) == sorted(
            f'Hybrid - {m}' for m in rp.BASELINES)
        assert sorted(part.loc[part.family == 'ladder'].comparison.unique()) == sorted(
            ['Hybrid - Hybrid_MF_Graph', 'Hybrid_MF_Graph - ALS+BPR', 'ALS+BPR - ALS', 'ALS+BPR - BPR', 'ALS+BPR - GNN'])
        assert sorted(part.loc[part.family == 'ablation'].comparison.unique()) == sorted(
            ['Hybrid - LTR_MF', 'Hybrid - Hybrid_no_graph', 'Hybrid - Hybrid_no_mf'])
    ladder = pd.read_csv(fold_dir / 'report' / 'ladder.csv').set_index('policy')
    summary = pd.read_csv(fold_dir / 'report' / 'test_summary.csv')
    for policy in ('repeat', 'novel'):
        ndcg = summary.loc[(summary.policy == policy) & (summary.k == rp.PRIMARY_K)].set_index('model').ndcg_mean
        assert ladder.loc[policy, 'L3'] == pytest.approx(ndcg['Hybrid'])
        assert ladder.loc[policy, 'best_baseline'] == pytest.approx(ndcg[rp.BASELINES].max())
    access = (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['purpose'] for line in access] == ['sealed_test']
    stamp = (fold_dir / 'ladder' / 'repeat.json').stat().st_mtime_ns
    rp.run(root, raw)  # resuming retrains nothing and opens nothing
    assert (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines() == access
    assert (fold_dir / 'ladder' / 'repeat.json').stat().st_mtime_ns == stamp
    files = sorted((fold_dir / 'ladder').iterdir())
    stamps = [p.stat().st_mtime_ns for p in files]
    assert rp.train_ladder(fold_dir, fold_dir / 'val_mf', selection['hybrid_backbone']['job_dir']) == selection['ladder']
    assert sorted((fold_dir / 'ladder').iterdir()) == files and [p.stat().st_mtime_ns for p in files] == stamps

    # The outer test period never reaches the selection or the ladder rankers.
    other_raw = tmp_path / 'raw_scrambled'
    shutil.copytree(raw, other_raw)
    assert scramble_test_period(other_raw) > 0
    other = tmp_path / 'run_scrambled'
    rp.run(other, other_raw)
    other_dir = other / 'TB2_full'
    again = json.loads((other_dir / 'selection.json').read_text(encoding='utf-8'))
    assert {p: v['mf_alpha'] for p, v in again['policies'].items()} == {p: v['mf_alpha'] for p, v in selection['policies'].items()}
    for policy in ('repeat', 'novel'):
        assert again['ladder'][policy]['models'] == selection['ladder'][policy]['models']  # same model bytes
        drop_seconds = lambda t: {n: {k: v for k, v in e.items() if k != 'seconds'} if isinstance(e, dict) else e
                                  for n, e in t.items()}
        assert drop_seconds(again['ladder'][policy]['training']) == drop_seconds(selection['ladder'][policy]['training'])
    assert rp.selection_decision(other_dir / 'selection.json') == rp.selection_decision(fold_dir / 'selection.json')
    assert not pd.read_csv(fold_dir / 'test_metrics.csv').equals(pd.read_csv(other_dir / 'test_metrics.csv'))


def test_load_ladder_refuses_a_replaced_model(tmp_path):
    fold_dir = tmp_path / 'fold'
    (fold_dir / 'ladder').mkdir(parents=True)
    path = fold_dir / 'ladder' / 'repeat_Hybrid.pkl'
    path.write_bytes(pickle.dumps({'model': 1}))
    selection = {'ladder': {'repeat': {'models': {'Hybrid': {'model': path.name, 'sha256': rp.sha256(path)}}}}}
    assert rp.load_ladder(fold_dir, selection, 'repeat')['Hybrid'][0] == {'model': 1}
    path.write_bytes(pickle.dumps({'model': 2}))
    with pytest.raises(RuntimeError, match='not the ladder ranker sealed'):
        rp.load_ladder(fold_dir, selection, 'repeat')


def test_selection_decision_covers_the_ladder(tmp_path):
    base = {'plan_sha256': 'p', 'fold': 'f', 'mf': {'als': {}, 'bpr': {}, 'bpr_epochs': 1},
            'gnn': {'config': {}, 'epochs': 1}, 'hybrid_backbone': {'config': {}, 'epochs': 1},
            'policies': {'repeat': {'mf_alpha': .5}}}
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(base), encoding='utf-8')
    without = rp.selection_decision(path)
    entry = {'modules': ['mf'], 'features': ['als'], 'params': {'max_iter': 5}, 'sha256': 'x', 'model': 'm.pkl'}
    path.write_text(json.dumps({**base, 'ladder': {'repeat': {'models': {'Hybrid': entry}}}}), encoding='utf-8')
    with_ladder = rp.selection_decision(path)
    path.write_text(json.dumps({**base, 'ladder': {'repeat': {'models': {
        'Hybrid': {**entry, 'params': {'max_iter': 10}}}}}}), encoding='utf-8')
    decisions = {without, with_ladder, rp.selection_decision(path)}
    assert None not in decisions and len(decisions) == 3
