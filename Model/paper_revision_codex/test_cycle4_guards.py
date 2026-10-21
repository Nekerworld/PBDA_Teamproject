"""Resume and test-access guards: a stored failed leakage check stops the run, a test cannot be reopened
under another selection or (across roots) another committed plan, capped GNN backbones are extended
until none is left, and the run records its thread and device settings."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from audit_tools import leakage_check
from test_v6_smoke import make_raw


def test_leakage_gate_reads_the_stored_verdict(tmp_path, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError('a stored verdict must be reused, not recomputed')
    monkeypatch.setattr(leakage_check, 'main', must_not_run)
    (tmp_path / 'leakage_check.json').write_text(json.dumps({'all_pass': False}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='Leakage check failed'):
        rp.leakage_gate(tmp_path / 'prepared', tmp_path)
    (tmp_path / 'leakage_check.json').write_text(json.dumps({'all_pass': True}), encoding='utf-8')
    rp.leakage_gate(tmp_path / 'prepared', tmp_path)


def test_leakage_gate_fails_on_a_fresh_failing_check(tmp_path, monkeypatch):
    def failing(prepared, raw, gnn, output):
        Path(output).write_text(json.dumps({'all_pass': False}), encoding='utf-8')
        return False
    monkeypatch.setattr(leakage_check, 'main', failing)
    with pytest.raises(RuntimeError, match='Leakage check failed'):
        rp.leakage_gate(tmp_path / 'prepared', tmp_path)
    with pytest.raises(RuntimeError, match='Leakage check failed'):  # resuming must not skip it
        rp.leakage_gate(tmp_path / 'prepared', tmp_path)


def test_access_log_refuses_reopening_under_another_selection(tmp_path):
    entry = {'utc': 'x', 'fold': 'F1', 'purpose': 'sealed_test', 'plan_sha256': 'a', 'selection_sha256': 's', 'git': 'g'}
    rp.access_log(tmp_path, entry)
    rp.access_log(tmp_path, entry)
    with pytest.raises(RuntimeError, match='selection'):
        rp.access_log(tmp_path, {**entry, 'selection_sha256': 't'})
    rp.access_log(tmp_path, {**entry, 'fold': 'F2', 'selection_sha256': 't'})
    assert len((tmp_path / 'test_access_log.jsonl').read_text().splitlines()) == 3


def committed_record(records, raw_hashes, plan_hash, folds, opened):
    target = records / 'results_vx'
    target.mkdir(parents=True)
    (target / 'plan.json').write_text(json.dumps({'plan_sha256': plan_hash, 'plan': {'folds': folds},
                                                  'raw_input_sha256': raw_hashes}), encoding='utf-8')
    (target / 'test_access_log.jsonl').write_text(
        ''.join(json.dumps({'fold': name, 'purpose': 'sealed_test', 'plan_sha256': plan_hash}) + '\n' for name in opened),
        encoding='utf-8')


def test_committed_test_conflicts_match_raw_input_and_window(tmp_path):
    raw = {'events.csv': 'r'}
    old = [{'name': 'A', 'end_fraction': .6, 'test_size': .25}, {'name': 'B', 'end_fraction': .8, 'test_size': .25}]
    committed_record(tmp_path, raw, 'p1', old, opened=['A'])
    same_window = [{'name': 'renamed', 'end_fraction': .6, 'test_size': .25}]
    assert rp.committed_test_conflicts(raw, 'p2', same_window, tmp_path) == [
        {'fold': 'renamed', 'opened_as': 'A', 'record': str(tmp_path / 'results_vx')}]
    assert rp.committed_test_conflicts(raw, 'p1', same_window, tmp_path) == []  # same plan: reproduction
    assert rp.committed_test_conflicts({'events.csv': 'other'}, 'p2', same_window, tmp_path) == []
    assert rp.committed_test_conflicts(None, 'p2', same_window, tmp_path) == []
    assert rp.committed_test_conflicts(raw, 'p2', [{'name': 'B', 'end_fraction': .8, 'test_size': .25}], tmp_path) == []
    assert rp.committed_test_conflicts(raw, 'p2', [{'name': 'C', 'end_fraction': .6, 'test_size': .2}], tmp_path) == []


def test_committed_v6_v7_records_guard_their_windows():
    """On the real committed records: v7 does not overlap v6's windows, and a new plan cannot reopen F1-F4."""
    plans = {v: json.loads((HERE / 'audit' / f'results_{v}' / 'plan.json').read_text(encoding='utf-8')) for v in ('v6', 'v7')}
    for v, plan in plans.items():
        assert rp.committed_test_conflicts(plan['raw_input_sha256'], plan['plan_sha256'], plan['plan']['folds']) == []
    raw = plans['v6']['raw_input_sha256']
    assert plans['v7']['raw_input_sha256'] == raw
    every_fold = plans['v6']['plan']['folds'] + plans['v7']['plan']['folds']
    conflicts = rp.committed_test_conflicts(raw, 'new-plan', every_fold)
    assert sorted(c['fold'] for c in conflicts) == ['F1_hist', 'F2_mid', 'F3_early', 'F4_unseen']


def test_run_refuses_a_window_opened_under_another_committed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'write_reports', lambda root: None)
    plan = rp.plan_payload()
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    raw = {'events.csv': 'r'}
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'plan.json').write_text(json.dumps({'plan_sha256': plan_hash, 'plan': plan, 'raw_input_sha256': raw,
                                                'code_hash_scheme': rp.CODE_HASH_SCHEME,
                                                'code_sha256': rp.code_hashes()}), encoding='utf-8')
    records = tmp_path / 'records'
    fold = rp.FOLDS[0]
    committed_record(records, raw, 'older-plan', [dict(fold, name='old')], opened=['old'])
    monkeypatch.setattr(rp, 'COMMITTED_RESULTS', records)
    with pytest.raises(RuntimeError, match='already opened under another committed plan'):
        rp.run(root, tmp_path, folds=[fold['name']])


def test_capped_backbones_are_extended_until_none_is_left(monkeypatch):
    monkeypatch.setattr(rp, 'HYBRID_BACKBONES', 3)
    rows = [dict(name='a', score=.9, at_cap=True), dict(name='b', score=.8, at_cap=False),
            dict(name='c', score=.7, at_cap=False), dict(name='d', score=.65, at_cap=True)]
    new_scores = {'a': .6, 'd': .66}
    calls = []

    def extend(todo):
        calls.append([r['name'] for r in todo])
        return [dict(score=new_scores[r['name']], at_cap=False, result=r['name']) for r in todo]
    results = rp.extend_capped_backbones(rows, extend)
    assert calls == [['a'], ['d']]
    assert results == ['a', 'd']
    assert [r['name'] for r in rows] == ['b', 'c', 'd', 'a']
    assert all(r.get('extended') for r in rows if r['name'] in 'ad')


def test_capped_backbone_still_capped_after_extension_is_not_extended_again(monkeypatch):
    monkeypatch.setattr(rp, 'HYBRID_BACKBONES', 3)
    rows = [dict(name='a', score=.9, at_cap=True)]
    results = rp.extend_capped_backbones(rows, lambda todo: [dict(score=.95, at_cap=True, result=1) for _ in todo])
    assert results == [1] and rows[0]['extended']


def test_environment_records_threads_and_device():
    env = rp.environment()
    assert env['thread_env'] == {name: '1' for name in rp.THREAD_VARIABLES}
    assert env['device'] in ('cuda', 'cpu') and env['torch_threads'] >= 1
    assert {api for api, _ in env['native_threads']} >= {'openblas'}


def test_blas_runs_single_threaded_even_if_the_caller_set_more_threads():
    """The variables read back as '1' either way (experiments.py rewrites them after numpy has loaded),
    so the check reads the thread count of the loaded libraries."""
    code = 'import json, rolling_protocol as rp; print(json.dumps(rp.native_threads()))'
    many = {name: '4' for name in rp.THREAD_VARIABLES}
    result = subprocess.run([sys.executable, '-c', code], cwd=HERE, capture_output=True, text=True,
                            env={**os.environ, **many}, check=True)
    threads = json.loads(result.stdout.strip().splitlines()[-1])
    assert threads and all(count == 1 for api, count in threads if api in ('openblas', 'mkl'))


def test_leakage_check_end_fraction_checks_a_fold_against_the_full_raw_data(tmp_path):
    raw = tmp_path / 'raw'
    make_raw(raw)
    fold = {'name': 'F', 'end_fraction': .6, 'test_size': .25, 'validation_size': .2}
    prepared, _ = rp.prepare_fold(fold, tmp_path, raw)
    assert leakage_check.main(prepared, tmp_path / 'F' / 'fold_raw')
    assert leakage_check.main(prepared, raw, end_fraction=.6)
    assert not leakage_check.main(prepared, raw)


def test_verify_plan_code_reports_changes_and_fails_only_when_asked():
    command = [sys.executable, str(HERE / 'report_tools' / 'verify_plan_code.py'), '--plan', 'audit/results_v6/plan.json']
    assert subprocess.run(command, cwd=HERE, capture_output=True).returncode == 0
    assert subprocess.run(command + ['--require-unchanged'], cwd=HERE, capture_output=True).returncode == 1
