"""Test re-opening needs the same plan and code, overlapping windows count, local roots are guarded,
--raw must match the plan, every opening of a fold uses one selection, cached fits must match
the configuration that asks for them, and an interrupted post-hoc step leaves no completion marker."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle4_guards import committed_record, raw_folder

RAW = {'events.csv': 'r'}
FOLD = {'name': 'A', 'end_fraction': .6, 'test_size': .25}


def test_code_matches_ignores_line_endings_but_not_content_or_file_set(tmp_path):
    (tmp_path / 'audit_tools').mkdir()
    (tmp_path / 'a.py').write_bytes(b'x = 1\nprint(x)\n')
    lf = rp.code_hashes(tmp_path, 'sha256-lf')
    crlf_raw = {'a.py': rp.hashlib.sha256(b'x = 1\r\nprint(x)\r\n').hexdigest()}
    assert rp.code_matches(lf, 'sha256-lf', tmp_path)
    assert rp.code_matches(crlf_raw, 'sha256-raw', tmp_path)
    (tmp_path / 'a.py').write_bytes(b'x = 1\r\nprint(x)\r\n')
    assert rp.code_matches(lf, 'sha256-lf', tmp_path)
    (tmp_path / 'a.py').write_bytes(b'x = 2\nprint(x)\n')
    assert not rp.code_matches(lf, 'sha256-lf', tmp_path)
    (tmp_path / 'a.py').write_bytes(b'x = 1\nprint(x)\n')
    (tmp_path / 'b.py').write_bytes(b'')
    assert not rp.code_matches(lf, 'sha256-lf', tmp_path)


def test_same_plan_with_changed_code_cannot_reopen_a_window(tmp_path):
    stale = dict(rp.code_hashes(), **{'rolling_protocol.py': '0' * 64})
    committed_record(tmp_path, RAW, 'p1', [FOLD], opened=['A'], code=stale)
    assert [c['fold'] for c in rp.committed_test_conflicts(RAW, 'p1', [FOLD], tmp_path)] == ['A']


def test_overlapping_windows_conflict_and_touching_windows_do_not(tmp_path):
    committed_record(tmp_path, RAW, 'p1', [FOLD], opened=['A'])  # window .45-.60
    def conflicts(end, size):
        return rp.committed_test_conflicts(RAW, 'p2', [{'name': 'N', 'end_fraction': end, 'test_size': size}], tmp_path)
    assert conflicts(.5, .25)          # .375-.50 overlaps the start
    assert conflicts(.65, .2)          # .52-.65 overlaps the end
    assert not conflicts(.7, .1)       # .63-.70 lies after it
    assert not conflicts(.45, .25)     # .3375-.45 touches it
    assert not conflicts(.8, .25)      # .60-.80 touches it


def test_local_roots_are_guarded_as_well_as_committed_ones(tmp_path):
    committed, local = tmp_path / 'audit', tmp_path / 'audit_runs'
    committed.mkdir()
    committed_record(local, RAW, 'p1', [FOLD], opened=['A'], name='v8')
    assert rp.committed_test_conflicts(RAW, 'p2', [FOLD], (committed, local))
    assert not rp.committed_test_conflicts(RAW, 'p2', [FOLD], (committed,))
    assert rp.RESULT_RECORDS == (HERE / 'audit', HERE / 'audit_runs')


def test_run_refuses_a_raw_folder_other_than_the_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'write_reports', lambda root: None)
    monkeypatch.setattr(rp, 'RESULT_RECORDS', ())
    plan = rp.plan_payload()
    plan_hash = rp.hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    planned, other = raw_folder(tmp_path / 'planned'), raw_folder(tmp_path / 'other')
    (other / 'events.csv').write_text('different\n', encoding='utf-8')
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'plan.json').write_text(json.dumps({'plan_sha256': plan_hash, 'plan': plan,
                                                'raw_input_sha256': rp.raw_hashes(planned),
                                                'code_hash_scheme': rp.CODE_HASH_SCHEME,
                                                'code_sha256': rp.code_hashes()}), encoding='utf-8')
    rp.run(root, planned, folds=['no-fold'])
    with pytest.raises(RuntimeError, match='differs from the raw input'):
        rp.run(root, other, folds=['no-fold'])
    started = [json.loads(line) for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [e for e in started if e['stage'] == 'run_started'][0]['native_threads']


def test_a_fold_keeps_one_selection_across_all_test_purposes(tmp_path):
    entry = {'utc': 'x', 'fold': 'F1', 'purpose': 'sealed_test', 'plan_sha256': 'a', 'selection_sha256': 's', 'git': 'g'}
    rp.access_log(tmp_path, entry)
    with pytest.raises(RuntimeError, match='selection'):
        rp.access_log(tmp_path, {**entry, 'purpose': 'posthoc_if_ablation', 'selection_sha256': 't'})
    with pytest.raises(RuntimeError, match='plan'):
        rp.access_log(tmp_path, {**entry, 'purpose': 'posthoc_candidate_gamma', 'plan_sha256': 'b'})
    rp.access_log(tmp_path, {**entry, 'purpose': 'posthoc_candidate_gamma'}, record=False)
    assert len((tmp_path / 'test_access_log.jsonl').read_text().splitlines()) == 1


def test_finished_fold_is_still_checked_against_the_log(tmp_path):
    fold_dir = tmp_path / 'F1'
    fold_dir.mkdir()
    (fold_dir / 'selection.json').write_text('{"sealed": 1}', encoding='utf-8')
    (fold_dir / 'test_metrics.csv').write_text('done\n', encoding='utf-8')
    rp.access_log(tmp_path, {'utc': 'x', 'fold': 'F1', 'purpose': 'sealed_test', 'plan_sha256': 'p',
                             'selection_sha256': rp.sha256(fold_dir / 'selection.json'), 'git': 'g'})
    rp.evaluate_fold_test(tmp_path, {'name': 'F1'}, None, None, None, 'p')
    (fold_dir / 'selection.json').write_text('{"sealed": 2}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='selection'):
        rp.evaluate_fold_test(tmp_path, {'name': 'F1'}, None, None, None, 'p')
    assert len((tmp_path / 'test_access_log.jsonl').read_text().splitlines()) == 1


def test_a_selection_sealed_under_another_plan_is_refused(tmp_path):
    (tmp_path / 'selection.json').write_text(json.dumps({'plan_sha256': 'p', 'gnn': {}}), encoding='utf-8')
    assert rp.sealed_selection(tmp_path, 'p') == {'plan_sha256': 'p', 'gnn': {}}
    with pytest.raises(RuntimeError, match='sealed under plan'):
        rp.sealed_selection(tmp_path, 'q')


def test_cached_mf_fit_must_match_the_requested_configuration(tmp_path):
    selection = {'als': {'factors': 8}, 'bpr': {'factors': 4}, 'bpr_epochs': 16}
    info = {'seed': 42, **selection, 'train_seconds': {}}
    (tmp_path / 'als_artifacts.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'mf_fit.json').write_text(json.dumps(info), encoding='utf-8')
    assert rp.fit_mf(tmp_path, selection, 42) == info
    with pytest.raises(RuntimeError, match='another configuration'):
        rp.fit_mf(tmp_path, dict(selection, bpr_epochs=32), 42)
    with pytest.raises(RuntimeError, match='another configuration'):
        rp.fit_mf(tmp_path, selection, 43)


def test_cached_gnn_job_must_match_the_requested_specification(tmp_path):
    spec = {'job_dir': str(tmp_path / 'job'), 'source': 's', 'config': {'layers': 1}, 'fixed': {'reg': 1e-4},
            'device': 'cpu', 'seed': 42, 'mode': 'refit', 'epochs': 90}
    (tmp_path / 'job').mkdir()
    (tmp_path / 'job' / 'job_result.json').write_text(json.dumps({'spec': dict(spec, device='cuda')}), encoding='utf-8')
    assert rp.run_gnn_jobs(tmp_path, [spec])[0]['spec']['epochs'] == 90  # device may differ on resume
    with pytest.raises(RuntimeError, match='another specification'):
        rp.run_gnn_jobs(tmp_path, [dict(spec, epochs=100)])


def test_score_precision_tool_loads_blas_single_threaded():
    code = ('import json, sys; sys.path.insert(0, "report_tools"); import check_score_precision as c; '
            'print(json.dumps(c.rp.native_threads()))')
    many = {name: '4' for name in rp.THREAD_VARIABLES}
    result = subprocess.run([sys.executable, '-c', code], cwd=HERE, capture_output=True, text=True,
                            env={**os.environ, **many}, check=True)
    threads = json.loads(result.stdout.strip().splitlines()[-1])
    assert threads and all(count == 1 for api, count in threads if api in ('openblas', 'mkl'))


def test_records_match_on_events_whatever_the_line_endings_or_other_raw_files(tmp_path):
    original = raw_folder(tmp_path / 'raw')
    (original / 'events.csv').write_bytes(b'timestamp,visitorid\n1,2\n')
    committed_record(tmp_path / 'records', rp.raw_hashes(original), 'p1', [FOLD], opened=['A'])
    copy = raw_folder(tmp_path / 'copy')
    (copy / 'events.csv').write_bytes(b'timestamp,visitorid\r\n1,2\r\n')
    (copy / 'category_tree.csv').write_bytes(b'categoryid,parentid\r\n')

    def opened(raw, forms=None):
        return [c['fold'] for c in rp.committed_test_conflicts(rp.raw_hashes(raw), 'p2', [FOLD], tmp_path / 'records',
                                                               event_forms=forms)]
    assert opened(copy) == []  # byte hash alone misses a line-ending copy
    assert opened(copy, rp.event_digests(copy)) == ['A']
    (copy / 'events.csv').write_bytes(b'timestamp,visitorid\n1,2\n')
    assert opened(copy) == ['A']  # other raw files do not define the window
    (copy / 'events.csv').write_bytes(b'timestamp,visitorid\n1,3\n')
    assert opened(copy, rp.event_digests(copy)) == []


def test_git_state_keeps_the_status_column_of_the_first_line(monkeypatch):
    class Done:
        stdout = ' M audit_runs/v7/x.json\n M rolling_protocol.py\n'
    monkeypatch.setattr(rp.subprocess, 'run', lambda *a, **k: Done())
    assert rp.git_state()['dirty_files'] == [' M rolling_protocol.py']


def test_cache_audit_compares_with_what_the_plan_and_selection_request(tmp_path):
    sys.path.insert(0, str(HERE / 'report_tools'))
    import check_cached_runs
    config = {'layers': 1, 'embedding_dim': 8, 'learning_rate': .01}
    plan = {'gnn_fixed': {'reg': 1e-4}, 'selection_seed': 42, 'seeds': [42], 'gnn_grid': [config],
            'gnn_max_epochs': 400, 'gnn_cap_rule': f'cap {rp.GNN_CAP_EXTENSION_EPOCHS}'}
    (tmp_path / 'plan.json').write_text(json.dumps({'plan': plan}), encoding='utf-8')
    fold = tmp_path / 'F'
    mf = {'als': {'factors': 4}, 'bpr': {'factors': 4}, 'bpr_epochs': 2}
    gnn_row = {'name': 'g', 'config': config, 'epochs': 90}
    (fold / 'runs' / 'seed_42').mkdir(parents=True)
    (fold / 'val_mf').mkdir()
    for d in (fold / 'val_mf', fold / 'runs' / 'seed_42'):
        (d / 'mf_fit.json').write_text(json.dumps(dict(mf, seed=42)), encoding='utf-8')
    (fold / 'selection.json').write_text(json.dumps({'mf': mf, 'gnn': gnn_row, 'hybrid_backbone': gnn_row}), encoding='utf-8')
    (fold / 'validation_gnn.json').write_text(json.dumps([dict(gnn_row, job_dir='x/g')]), encoding='utf-8')

    def trained(job, **spec):
        (fold / job).mkdir(parents=True, exist_ok=True)
        (fold / job / 'job_result.json').write_text(json.dumps({'spec': dict(spec, fixed=plan['gnn_fixed'])}), encoding='utf-8')
    trained(Path('gnn_val') / rp.config_name('gnn', config), config=config, seed=42, mode='select', epochs=400)
    trained(Path('runs') / 'gnn_seed_42', config=config, seed=42, mode='refit', epochs=90)
    clean = check_cached_runs.main(tmp_path, tmp_path / 'ok.json')
    assert (clean['gnn_requested'], clean['gnn_mismatches'], clean['mf_mismatches']) == (2, 0, 0)
    trained(Path('runs') / 'gnn_seed_42', config=dict(config, layers=4), seed=42, mode='refit', epochs=400)
    stale = check_cached_runs.main(tmp_path, tmp_path / 'bad.json')
    assert stale['gnn_mismatches'] == 1 and set(stale['gnn'][1]['differs']) == {'config', 'epochs'}


class CrashOnIteration(list):
    def __iter__(self):
        raise KeyboardInterrupt('interrupted during the post-hoc gamma sweep')


def test_interrupted_posthoc_leaves_no_completion_marker(tmp_path, monkeypatch):
    from test_v6_smoke import make_raw, tiny_settings
    raw = tmp_path / 'raw'
    make_raw(raw)
    tiny_settings(monkeypatch)
    monkeypatch.setattr(rp, 'FOLDS', rp.FOLDS[:1])
    gammas = rp.POSTHOC_GAMMAS
    monkeypatch.setattr(rp, 'POSTHOC_GAMMAS', CrashOnIteration(gammas))
    root = tmp_path / 'run'
    with pytest.raises(KeyboardInterrupt):
        rp.run(root, raw)
    report = root / rp.FOLDS[0]['name'] / 'report'
    assert not (report / 'posthoc_candidate_depth.csv').exists()
    monkeypatch.setattr(rp, 'POSTHOC_GAMMAS', gammas)
    rp.run(root, raw)
    assert (report / 'posthoc_candidate_depth.csv').exists() and (report / 'posthoc_gamma.csv').exists()
