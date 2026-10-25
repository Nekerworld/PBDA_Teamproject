"""MF selection data, GNN refits and the test-time model assembly are pinned to what the protocol selected."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model
import rolling_protocol as rp
from test_v6_smoke import make_raw, tiny_settings


class Stop(Exception):
    pass


def manifest(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'manifest.json').write_text(json.dumps(fields), encoding='utf-8')
    return folder


def test_mf_selection_refuses_data_that_is_not_the_inner_validation_of_its_fold(tmp_path):
    fold = tmp_path / 'F2'
    outer = manifest(fold / 'prepared', role='test', protocol_id='outer')
    inner = manifest(outer / 'validation', role='validation', protocol_id='inner')
    copy = manifest(fold / 'val_mf', role='validation', protocol_id='inner')
    other = manifest(tmp_path / 'F1' / 'prepared' / 'validation', role='validation', protocol_id='other')
    for source in (inner, copy):
        rp.require_inner_validation(fold, source, 'x')
    for source in (outer, other):
        with pytest.raises(RuntimeError, match='inner validation split of F2'):
            rp.mf_validation(fold, source, {})


def selection_run(tmp_path, monkeypatch):
    """run_fold up to the sealed selection, with stub training whose scores make the best standalone GNN differ from
    the backbone of the best repeat Hybrid row, and the novel Hybrid best differ from the repeat one."""
    raw = tmp_path / 'raw'
    make_raw(raw)
    tiny_settings(monkeypatch)
    names = [rp.config_name('gnn', c) for c in rp.GNN_GRID]
    seen = {'mf_grid': [], 'mf_fit': []}
    mf = {'als': rp.ALS_GRID[0], 'bpr': rp.BPR_GRID[0], 'bpr_epochs': 2, 'fits': {'ALS': 1, 'BPR': 1},
          'evaluated_configs': {'ALS': 1, 'BPR': 2}, 'train_seconds': {'ALS': 0., 'BPR': 0.}}

    def mf_validation(fold_dir, val_dir, contexts):
        seen['mf_grid'].append(Path(val_dir).resolve())
        return mf

    class Scores:
        def __init__(self, ctx, als, bpr, gnn=None):
            self.n = len(ctx.ids)

        def mf(self, user, alpha):
            return np.zeros(self.n)

    def grid(path, compute):
        policy, backbone = Path(path).stem.split('_', 3)[2:]
        best = {('repeat', names[1]): 1, ('novel', names[1]): 2, ('novel', names[0]): 0}.get((policy, backbone))
        top = {'repeat': .8, 'novel': .9 if backbone == names[0] else .7}[policy]
        return [dict(c, backbone=backbone, ndcg=0., recall=0., score=top if i == best else .1)
                for i, c in enumerate(rp.HYBRID_GRID)], True

    def sealed(fold_dir, plan_hash):
        raise Stop

    monkeypatch.setattr(rp, 'run_gnn_jobs', lambda root, specs, stop=None: [
        dict(chosen_epochs=s['epochs'], at_cap=False, seconds=1., evaluated_checkpoints=1) for s in specs])
    monkeypatch.setattr(rp, 'mf_validation', mf_validation)
    monkeypatch.setattr(rp, 'fit_mf', lambda directory, selection, seed: seen['mf_fit'].append(Path(directory)))
    monkeypatch.setattr(rp, 'load_factors', lambda directory, context: (defaultdict(int), defaultdict(int)))
    monkeypatch.setattr(rp, 'load_gnn', lambda directory, context: Path(directory).name)
    monkeypatch.setattr(rp, 'evaluate_vectors', lambda contexts, vectors: {
        p: {'ndcg': .9 if vectors == names[0] else .1, 'recall': .9 if vectors == names[0] else .1}
        for p in rp.POLICIES})
    monkeypatch.setattr(rp, 'FactorScores', Scores)
    monkeypatch.setattr(rp, 'cached_grid', grid)
    monkeypatch.setattr(rp, 'sealed_selection', sealed)
    return raw, tmp_path / 'run', names, seen


def test_run_fold_selects_on_the_inner_validation_and_records_the_repeat_hybrid_backbone(tmp_path, monkeypatch):
    raw, root, names, seen = selection_run(tmp_path, monkeypatch)
    with pytest.raises(Stop):
        rp.run_fold(root, rp.FOLDS[0], raw, 'plan')
    fold = root / rp.FOLDS[0]['name']
    assert seen['mf_grid'] == [(fold / 'prepared' / 'validation').resolve()]
    assert seen['mf_fit'] == [fold / 'val_mf']
    selection = json.loads((fold / 'selection.json').read_text(encoding='utf-8'))
    assert selection['gnn']['name'] == names[0]
    assert selection['hybrid_backbone']['name'] == names[1]
    hybrid = {p: selection['policies'][p]['hybrid'] for p in rp.POLICIES}
    assert {h['backbone'] for h in hybrid.values()} == {names[1]}
    for policy, index in (('repeat', 1), ('novel', 2)):
        assert {k: hybrid[policy][k] for k in rp.HYBRID_GRID[index]} == rp.HYBRID_GRID[index]
    assert selection['policies']['novel']['unrestricted_best_backbone'] == names[0]


def test_hybrid_selection_fit_on_the_outer_split_is_refused(tmp_path, monkeypatch):
    raw, root, names, seen = selection_run(tmp_path, monkeypatch)
    fold = root / rp.FOLDS[0]['name']
    clone = rp.clone_protocol
    monkeypatch.setattr(rp, 'clone_protocol', lambda source, destination: clone(
        fold / 'prepared' if Path(destination).name == 'val_mf' else source, destination))
    with pytest.raises(RuntimeError, match='inner validation split'):
        rp.run_fold(root, rp.FOLDS[0], raw, 'plan')
    assert seen['mf_fit'] == []


def test_model_rankings_score_the_hybrid_with_the_hybrid_backbone(monkeypatch):
    class Scores:
        def __init__(self, ctx, als, bpr, gnn):
            self.als, self.bpr, self.gnn = als, bpr, gnn

        def mf(self, user, alpha):
            return 0.

    class Context:
        active = []

        def rank_scores(self, scores, k):
            return ('ranked', scores)

        def popularity(self, k):
            return 'popularity'

        def recent(self, k, fill=False):
            return 'recent'

    monkeypatch.setattr(rp, 'FactorScores', Scores)
    monkeypatch.setattr(rp, 'hybrid_ranking', lambda scores, config, k, mf_source='mf': (scores.gnn, mf_source))
    rankings = rp.model_rankings(Context(), 'als', 'bpr', 'standalone', 'backbone', {'mf_alpha': .5, 'hybrid': {}})
    assert rankings['GNN'] == ('ranked', 'standalone')
    assert rankings['Hybrid'] == ('backbone', 'mf') and rankings['Hybrid_ALS_only'] == ('backbone', 'als')


def test_fold_test_uses_each_policy_selection_and_the_hybrid_backbone_refit(tmp_path, monkeypatch):
    calls = []

    class Context:
        def __init__(self, directory, policy):
            self.policy = policy

        def population(self):
            return {}

    def rankings(context, als, bpr, gnn_std, gnn_hybrid, policy_selection):
        calls.append((context.policy, policy_selection, gnn_std, gnn_hybrid))
        if len(calls) == len(rp.POLICIES) * len(rp.SEEDS):
            raise Stop
        return {}

    monkeypatch.setattr(rp, 'open_test', lambda *args: True)
    monkeypatch.setattr(rp, 'EvaluationContext', Context)
    monkeypatch.setattr(rp, 'load_factors', lambda directory, context: (None, None))
    monkeypatch.setattr(rp, 'load_gnn', lambda directory, context: str(directory))
    monkeypatch.setattr(rp, 'model_rankings', rankings)
    monkeypatch.setattr(rp, 'fold_statistics', lambda policy, details: (None, None))
    selection = {'policies': {p: {'policy': p} for p in rp.POLICIES}}
    refits = {s: {'mf': str(tmp_path / f'mf_{s}'), 'gnn': 'standalone', 'hybrid_gnn': 'backbone'} for s in rp.SEEDS}
    with pytest.raises(Stop):
        rp.evaluate_fold_test(tmp_path, {'name': 'F'}, None, selection, refits, 'plan')
    assert len(calls) == len(rp.POLICIES) * len(rp.SEEDS)
    for policy, policy_selection, standalone, backbone in calls:
        assert policy_selection is selection['policies'][policy]
        assert (standalone, backbone) == ('standalone', 'backbone')


def refit_job(tmp_path, monkeypatch, mode_for):
    seen = []

    class Generator:
        def __init__(self, processed_dir, **kwargs):
            self.processed_dir, self.kwargs = Path(processed_dir), kwargs
            seen.append(kwargs)

        def run(self):
            mode, chosen = mode_for(self.kwargs)
            (self.processed_dir / 'gnn_training_history.json').write_text(
                json.dumps({'mode': mode, 'chosen_epochs': chosen}), encoding='utf-8')

    monkeypatch.setattr(model, 'GNNEmbeddingGenerator', Generator)
    job = manifest(tmp_path / 'runs' / 'gnn_seed_42', role='test')
    spec = {'job_dir': str(job), 'source': str(job), 'config': {}, 'fixed': rp.GNN_FIXED, 'device': 'cpu',
            'seed': 42, 'mode': 'refit', 'epochs': 90}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec), encoding='utf-8')
    return job, path, seen


def test_gnn_refit_trains_the_fixed_epochs_without_early_stopping(tmp_path, monkeypatch):
    job, path, seen = refit_job(tmp_path, monkeypatch, lambda kwargs: (
        'fixed_epochs', kwargs['epochs']) if kwargs['early_stopping_patience'] is None else ('validation_selection', 40))
    rp.gnn_job(path)
    assert seen[0]['early_stopping_patience'] is None and seen[0]['epochs'] == 90
    assert json.loads((job / 'job_result.json').read_text(encoding='utf-8'))['chosen_epochs'] == 90


def test_gnn_refit_that_chose_its_own_epochs_is_refused(tmp_path, monkeypatch):
    job, path, _ = refit_job(tmp_path, monkeypatch, lambda kwargs: ('chronological_validation_then_refit', 40))
    with pytest.raises(RuntimeError, match='fixed 90 epochs'):
        rp.gnn_job(path)
    assert not (job / 'job_result.json').exists()
