"""Protocol v10: the v8 Hybrid family and the v9 ladder in one RetailRocket run, repeat only, earlier plans unchanged."""
import importlib.util
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
import hybrid_v9 as v9
import rolling_protocol as rp
import stats_tools
from test_v6_smoke import make_raw, tiny_settings
from test_v8_protocol import LTR_MODELS, RESULTS, plan_hash, scramble_test_period, use
from test_v9_protocol import SMALL

LADDER_REPORTED = set(rp.V10_LADDER_NAMES.values())
V6_V7 = {json.loads((RESULTS / f'results_{n}' / 'plan.json').read_text(encoding='utf-8'))['plan_sha256']
         for n in ('v6', 'v7')}


def primary_family_tool():
    spec = importlib.util.spec_from_file_location('primary_family', HERE / 'audit' / 'v10protocol' / 'primary_family.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_v10(monkeypatch):
    use(monkeypatch, 'v10')
    tiny_settings(monkeypatch)
    monkeypatch.setattr(rp, 'FOLDS', rp.V10['FOLDS'][:1])
    monkeypatch.setattr(ltr, 'PARAM_GRID', [dict(learning_rate=.1, max_iter=5, max_leaf_nodes=4),
                                            dict(learning_rate=.1, max_iter=10, max_leaf_nodes=4)])
    monkeypatch.setattr(ltr, 'POOL', dict(per_model=6, history=6, popular=4))
    monkeypatch.setitem(v9.DESIGNS, 'a02', dict(v9.DESIGNS['a02'], pool=SMALL))
    monkeypatch.setattr(rp, 'RERANKER_CV_FOLDS', 2)
    monkeypatch.setattr(rp, 'LADDER_NEGATIVES', 5)
    monkeypatch.setattr(rp, 'gnn_device', lambda: 'cpu')  # CUDA scatter-adds are not bit-reproducible


@pytest.mark.parametrize('name', ['v6', 'v7', 'v8', 'v9'])
def test_v10_code_keeps_the_sealed_plans(monkeypatch, name):
    use(monkeypatch, name)
    stored = json.loads((RESULTS / f'results_{name}' / 'plan.json').read_text(encoding='utf-8'))
    plan = rp.plan_payload()
    assert plan_hash() == stored['plan_sha256']
    assert not {'policies', 'acknowledged_test_records'} & set(plan) and rp.POLICIES == ('repeat', 'novel')
    assert rp.ACKNOWLEDGED_TEST_RECORDS == {} and rp.LADDER_NAMES == {}


def test_v10_plan_records_the_combined_design(monkeypatch):
    use(monkeypatch, 'v10')
    plan = rp.plan_payload()
    assert plan['protocol'] == 'rolling-origin-v10' and plan['posthoc_test_analyses'] == 'none'
    assert plan['policies'] == ['repeat'] and rp.POLICIES == ('repeat',)
    v6 = json.loads((RESULTS / 'results_v6' / 'plan.json').read_text(encoding='utf-8'))['plan']
    v7 = json.loads((RESULTS / 'results_v7' / 'plan.json').read_text(encoding='utf-8'))['plan']
    for mine, theirs in zip(plan['folds'], v6['folds'] + v7['folds']):  # the same windows, new descriptions
        assert {k: v for k, v in mine.items() if k != 'test_window'} == {k: v for k, v in theirs.items() if k != 'test_window'}
    assert set(plan['acknowledged_test_records']) == V6_V7
    v8 = json.loads((RESULTS / 'results_v8' / 'plan.json').read_text(encoding='utf-8'))['plan']
    for key in ('als_grid', 'bpr_grid', 'gnn_grid', 'gnn_fixed', 'gnn_max_epochs', 'gnn_cap_rule', 'mf_alphas',
                'mf_common', 'bpr_epoch_checkpoints', 'bpr_negatives', 'seeds', 'selection_seed', 'ks',
                'preprocessing', 'hybrid_grid_size', 'hybrid_grid_axes', 'candidate_ks', 'hybrid_backbones', 'reranker'):
        assert plan[key] == v8[key], key
    v9_plan = json.loads((RESULTS / 'results_v9' / 'plan.json').read_text(encoding='utf-8'))['plan']
    assert {k: v for k, v in plan['ladder'].items() if k not in ('reported_names', 'graph_gnn')} == v9_plan['ladder']
    assert plan['ladder']['reported_names'] == rp.V10_LADDER_NAMES
    assert plan['baselines'] == rp.V8['BASELINES'] + ['CoVisit']
    assert rp.HYBRIDS == ['Hybrid', 'Hybrid_LTR', 'Hybrid_LTR_W', 'Hybrid_v9'] and rp.FUSION and rp.RERANKER and rp.LADDER


def write_record(base, name, plan_sha, events_sha, folds, fold_opened):
    folder = base / name
    folder.mkdir(parents=True)
    (folder / 'plan.json').write_text(json.dumps({'plan_sha256': plan_sha, 'raw_input_sha256': {'events.csv': events_sha},
                                                  'plan': {'folds': folds}}), encoding='utf-8')
    (folder / 'test_access_log.jsonl').write_text(json.dumps({'fold': fold_opened, 'selection_sha256': 's'}) + '\n',
                                                  encoding='utf-8')


def test_only_the_named_plans_may_have_opened_the_windows(tmp_path, monkeypatch):
    use(monkeypatch, 'v10')
    folds = rp.V10['FOLDS']
    acknowledged = sorted(V6_V7)[0]
    write_record(tmp_path, 'named', acknowledged, 'E', folds[:1], folds[0]['name'])
    raw = {'events.csv': 'E'}
    assert rp.committed_test_conflicts(raw, 'mine', folds, records=tmp_path) == []
    write_record(tmp_path, 'other', 'f' * 64, 'E', folds[:1], folds[0]['name'])
    found = rp.committed_test_conflicts(raw, 'mine', folds, records=tmp_path)
    assert [c['record'] for c in found] == [str(tmp_path / 'other')] and found[0]['fold'] == folds[0]['name']
    use(monkeypatch, 'v9')  # without the plan's names the v6/v7 openings still block
    assert len(rp.committed_test_conflicts(raw, 'mine', folds, records=tmp_path)) == 2


def test_the_committed_records_hold_only_the_named_openings(monkeypatch):
    use(monkeypatch, 'v10')
    stored = json.loads((RESULTS / 'results_v6' / 'plan.json').read_text(encoding='utf-8'))
    raw = {'events.csv': stored['raw_input_sha256']['events.csv']}
    assert rp.committed_test_conflicts(raw, plan_hash(), rp.FOLDS, records=RESULTS) == []
    use(monkeypatch, 'v9')
    blocked = rp.committed_test_conflicts(raw, plan_hash(), rp.V10['FOLDS'], records=RESULTS)
    assert {c['fold'] for c in blocked} == {f['name'] for f in rp.V10['FOLDS']}


def test_clone_retries_a_refused_rename_and_then_gives_up(monkeypatch, tmp_path):
    import experiments
    calls, waits = [], []
    real = experiments.os.replace

    def refuse_twice(source, destination):
        calls.append(Path(destination).name)
        if len(calls) <= 2:
            raise PermissionError(5, 'Access is denied')
        return real(source, destination)

    monkeypatch.setattr(experiments.os, 'replace', refuse_twice)
    monkeypatch.setattr(experiments.time, 'sleep', waits.append)
    source, target = tmp_path / 'a.partial', tmp_path / 'a'
    source.mkdir()
    experiments.replace_with_retry(source, target)
    assert target.is_dir() and not source.exists() and len(calls) == 3 and waits == [.5, 1.]

    def refuse(source, destination):
        raise PermissionError(5, 'Access is denied')
    monkeypatch.setattr(experiments.os, 'replace', refuse)
    (tmp_path / 'b.partial').mkdir()
    with pytest.raises(PermissionError):
        experiments.replace_with_retry(tmp_path / 'b.partial', tmp_path / 'b')
    assert len(waits) == 2 + 5


def test_v10_rankings_put_the_ladder_on_the_standalone_gnn(monkeypatch):
    class Scores:
        def __init__(self, ctx, als, bpr, gnn):
            self.als, self.bpr, self.gnn = als, bpr, gnn

        def mf(self, uid, alpha):
            return uid

    class Context:
        active = []

        def rank_scores(self, scores, k):
            return 'ranked'

        def popularity(self, k):
            return 'pop'

        def recent(self, k, fill=False):
            return 'recent'

    seen = {}

    def ladder_rankings(context, std, hyb, alpha, ladder, item_tables):
        seen.update(std=std.gnn, hyb=hyb.gnn, tables=item_tables)
        return {'ALS': 'other', 'CoVisit': 'cv', 'Hybrid_v9': 'l3'}

    monkeypatch.setattr(rp, 'FactorScores', Scores)
    monkeypatch.setattr(rp, 'hybrid_ranking', lambda scores, config, k, mf_source='mf': (scores.gnn, mf_source))
    monkeypatch.setattr(rp, 'ladder_rankings', ladder_rankings)
    rankings = rp.model_rankings(Context(), 'als', 'bpr', 'standalone', 'backbone', {'mf_alpha': .5, 'hybrid': {}},
                                 ladder={'Hybrid': None}, ladder_tables='std_tables')
    assert seen == {'std': 'standalone', 'hyb': 'standalone', 'tables': 'std_tables'}
    assert rankings['Hybrid'] == ('backbone', 'mf') and rankings['ALS'] == 'ranked'
    assert rankings['CoVisit'] == 'cv' and rankings['Hybrid_v9'] == 'l3'
    seen.clear()
    rp.model_rankings(Context(), 'als', 'bpr', 'standalone', 'backbone', {'mf_alpha': .5}, item_tables='hyb_tables',
                      ladder={'Hybrid': None})  # v9: the ladder alone, on the Hybrid backbone
    assert seen == {'std': 'standalone', 'hyb': 'backbone', 'tables': 'hyb_tables'}


def test_v10_trains_the_ladder_on_the_selected_gnn_and_the_rerankers_on_the_backbone(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw, users=60, events=9000)
    tiny_v10(monkeypatch)
    fold_dir = tmp_path / 'run' / rp.FOLDS[0]['name']
    best_row, seen = rp.best_row, {}

    def other_backbone(rows, keys):  # the v1 Hybrid grid picks a backbone that is not the top GNN
        if 'backbone' in keys:
            top = json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8'))[0]['name']
            rows = [r for r in rows if r['backbone'] != top]
        return best_row(rows, keys)

    class Stop(Exception):
        pass

    def ladder(fold, val_mf, gnn_dir):
        seen['ladder'] = gnn_dir
        return {}

    def rerankers(fold, val_mf, gnn_dir):
        seen['rerankers'] = gnn_dir
        raise Stop

    monkeypatch.setattr(rp, 'best_row', other_backbone)
    monkeypatch.setattr(rp, 'train_ladder', ladder)
    monkeypatch.setattr(rp, 'train_rerankers', rerankers)
    with pytest.raises(Stop):
        rp.run(tmp_path / 'run', raw)
    rows = json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8'))
    assert seen['ladder'] == rows[0]['job_dir'] and seen['rerankers'] == rows[1]['job_dir']


def test_v10_end_to_end_seals_both_families_before_the_test(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw, users=60, events=9000)
    tiny_v10(monkeypatch)
    root = tmp_path / 'run'
    rp.run(root, raw)
    fold = rp.FOLDS[0]['name']
    fold_dir = root / fold
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']
    assert plan['protocol'] == 'rolling-origin-v10' and plan['policies'] == ['repeat']
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    assert set(selection['policies']) == {'repeat'} and set(selection['ladder']) == set(selection['rerankers']) == {'repeat'}
    assert 'hybrid' in selection['policies']['repeat'] and list(fold_dir.glob('validation_hybrid_repeat_*.csv'))
    assert not list(fold_dir.glob('*novel*'))
    assert selection['budget']['evaluated_configs']['Hybrid'] == len(rp.HYBRID_GRID) * rp.HYBRID_BACKBONES
    assert all('novel' not in row for row in json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8')))
    for name, entry in selection['ladder']['repeat']['models'].items():
        assert entry['sha256'] == rp.sha256(fold_dir / 'ladder' / entry['model']) and name in v9.RANKERS
    for name, entry in selection['rerankers']['repeat']['models'].items():
        assert entry['sha256'] == rp.sha256(fold_dir / 'rerankers' / entry['model']) and name in LTR_MODELS
    stages = [json.loads(line)['stage'] for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert max(stages.index('ladder_trained'), stages.index('rerankers_trained')) < stages.index('selection_sealed') \
        < stages.index('test_evaluated')
    metrics = pd.read_csv(fold_dir / 'test_metrics.csv')
    assert set(metrics.policy) == {'repeat'}
    assert set(metrics.model) == set(rp.BASELINES) | {'Hybrid', 'Hybrid_ALS_only'} | LTR_MODELS | LADDER_REPORTED
    tests = pd.read_csv(fold_dir / 'report' / 'paired_tests.csv')
    assert set(tests.policy) == {'repeat'}
    for family in ('confirmatory', 'ladder', 'ablation', 'secondary'):
        rows = tests.loc[tests.family == family]
        assert len(rows) and np.allclose(rows.p_holm, stats_tools.holm(rows.p_value))
    by = lambda family: sorted(tests.loc[tests.family == family].comparison.unique())
    assert by('confirmatory') == sorted(f'{h} - {b}' for h in rp.HYBRIDS for b in rp.BASELINES)
    assert by('ladder') == sorted(['Hybrid_v9 - Hybrid_v9_MF_Graph', 'Hybrid_v9_MF_Graph - ALS+BPR', 'ALS+BPR - ALS',
                                   'ALS+BPR - BPR', 'ALS+BPR - GNN'])
    assert by('ablation') == sorted(['Hybrid_v9 - Hybrid_v9_MF', 'Hybrid_v9 - Hybrid_v9_no_graph',
                                     'Hybrid_v9 - Hybrid_v9_no_mf'])
    assert by('secondary') == sorted(['Hybrid_LTR - Hybrid', 'Hybrid_LTR_W - Hybrid'])
    ladder = pd.read_csv(fold_dir / 'report' / 'ladder.csv').set_index('policy')
    summary = pd.read_csv(fold_dir / 'report' / 'test_summary.csv')
    ndcg = summary.loc[summary.k == rp.PRIMARY_K].set_index('model').ndcg_mean
    assert list(ladder.index) == ['repeat']
    assert ladder.loc['repeat', 'L3'] == pytest.approx(ndcg['Hybrid_v9'])
    assert ladder.loc['repeat', 'L2'] == pytest.approx(ndcg['Hybrid_v9_MF_Graph'])
    assert ladder.loc['repeat', 'best_baseline'] == pytest.approx(ndcg[rp.BASELINES].max())
    family = primary_family_tool()
    assert family.BASELINES == rp.BASELINES and set(family.PRIMARY) <= set(rp.HYBRIDS)
    frame = family.primary_family(root, [fold])
    assert len(frame) == 29 and np.allclose(frame.p_holm_primary, stats_tools.holm(frame.p_value))
    one_fold = tests.loc[tests.metric == 'ndcg'].set_index(['family', 'comparison'])
    assert np.allclose(frame.p_value, [one_fold.loc[(f, c), 'p_value'] for f, c in family.wanted()])
    ladder_verdict, _ = family.verdicts(root, frame, [fold])
    assert ladder_verdict.loc[0, 'L3'] == pytest.approx(ndcg['Hybrid_v9'])
    access = (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert [json.loads(line)['purpose'] for line in access] == ['sealed_test']
    rp.run(root, raw)  # resuming retrains nothing and opens nothing
    assert (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines() == access

    # The outer test period never reaches the selection, the re-rankers or the ladder rankers.
    other_raw = tmp_path / 'raw_scrambled'
    shutil.copytree(raw, other_raw)
    assert scramble_test_period(other_raw, test_size=rp.FOLDS[0]['test_size']) > 0
    other = tmp_path / 'run_scrambled'
    rp.run(other, other_raw)
    again = json.loads((other / fold / 'selection.json').read_text(encoding='utf-8'))
    for key in ('ladder', 'rerankers'):
        assert again[key]['repeat']['models'] == selection[key]['repeat']['models']  # same model bytes
    assert rp.selection_decision(other / fold / 'selection.json') == rp.selection_decision(fold_dir / 'selection.json')
    assert not pd.read_csv(fold_dir / 'test_metrics.csv').equals(pd.read_csv(other / fold / 'test_metrics.csv'))
