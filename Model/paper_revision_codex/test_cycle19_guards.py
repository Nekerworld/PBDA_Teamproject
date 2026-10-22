"""GNN selection jobs, which early-stop on the labels of their data, are built and checked on the inner validation split."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'report_tools'))
import rolling_protocol as rp
import check_cached_runs
from test_v6_smoke import make_raw, tiny_settings


def manifest(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'manifest.json').write_text(json.dumps(fields), encoding='utf-8')
    return folder


def test_selection_spec_refuses_data_that_is_not_an_inner_validation_split(tmp_path):
    outer = manifest(tmp_path / 'prepared', role='test')
    inner = manifest(outer / 'validation', role='validation')
    assert rp.gnn_spec(tmp_path / 'job', inner, {}, 42, 'select', 4)['source'] == str(inner.resolve())
    with pytest.raises(RuntimeError, match='inner validation split'):
        rp.gnn_spec(tmp_path / 'job', outer, {}, 42, 'select', 4)
    assert rp.gnn_spec(tmp_path / 'job', outer, {}, 42, 'refit', 4)['mode'] == 'refit'


class Stop(Exception):
    pass


def test_run_fold_builds_the_gnn_selection_jobs_on_the_inner_validation_split(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw)
    tiny_settings(monkeypatch)
    requested = []
    monkeypatch.setattr(rp, 'run_gnn_jobs', lambda root, specs, stop=None: requested.extend(specs))

    def no_mf(*args):
        raise Stop
    monkeypatch.setattr(rp, 'mf_validation', no_mf)
    root = tmp_path / 'run'
    with pytest.raises(Stop):
        rp.run_fold(root, rp.FOLDS[0], raw, 'plan')
    validation = root / rp.FOLDS[0]['name'] / 'prepared' / 'validation'
    assert len(requested) == len(rp.GNN_GRID)
    assert {(s['mode'], s['source']) for s in requested} == {('select', str(validation.resolve()))}


def test_cached_selection_job_trained_on_the_outer_split_is_reported(tmp_path):
    config = {'layers': 1, 'embedding_dim': 8, 'learning_rate': .01}
    plan = {'gnn_fixed': {'reg': 1e-4}, 'selection_seed': 42, 'seeds': [42], 'gnn_grid': [config],
            'gnn_max_epochs': 400, 'gnn_cap_rule': f'cap {rp.GNN_CAP_EXTENSION_EPOCHS}'}
    (tmp_path / 'plan.json').write_text(json.dumps({'plan': plan}), encoding='utf-8')
    fold = tmp_path / 'F'
    mf = {'als': {'factors': 4}, 'bpr': {'factors': 4}, 'bpr_epochs': 2}
    row = {'name': 'g', 'config': config, 'epochs': 90}
    for d in (fold / 'val_mf', fold / 'runs' / 'seed_42'):
        d.mkdir(parents=True)
        (d / 'mf_fit.json').write_text(json.dumps(dict(mf, seed=42)), encoding='utf-8')
    (fold / 'selection.json').write_text(json.dumps({'mf': mf, 'gnn': row, 'hybrid_backbone': row}), encoding='utf-8')
    (fold / 'validation_gnn.json').write_text(json.dumps([dict(row, job_dir='x/g')]), encoding='utf-8')
    manifest(fold / 'prepared', protocol_id='outer')
    manifest(fold / 'prepared' / 'validation', protocol_id='inner')
    jobs = {Path('gnn_val') / rp.config_name('gnn', config): dict(mode='select', epochs=400),
            Path('runs') / 'gnn_seed_42': dict(mode='refit', epochs=90)}
    for job, spec in jobs.items():
        manifest(fold / job, protocol_id='inner' if spec['mode'] == 'select' else 'outer')
        (fold / job / 'job_result.json').write_text(json.dumps({'spec': dict(spec, config=config, seed=42,
                                                                              fixed=plan['gnn_fixed'])}), encoding='utf-8')
    assert check_cached_runs.main(tmp_path, tmp_path / 'ok.json')['gnn_mismatches'] == 0
    manifest(fold / next(iter(jobs)), protocol_id='outer')
    leaked = check_cached_runs.main(tmp_path, tmp_path / 'leaked.json')
    assert leaked['gnn_mismatches'] == 1 and leaked['gnn'][0]['differs'] == ['data']
