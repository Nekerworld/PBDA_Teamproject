"""Code edited during a run is refused before a GNN job or test opening, the opening re-check matches a
line-ending copy of events.csv, a run's own root is not taken for another run, a cut-off test record does not
hide or block openings, and a reused Hybrid grid is bit-exact."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle4_guards import committed_record

FOLD = {'name': 'A', 'end_fraction': .6, 'test_size': .25}
RAW = {'events.csv': 'r'}


def run_root(path, edited=False):
    code = rp.code_hashes()
    if edited:
        code['policy_evaluation.py'] = '0' * 64
    path.mkdir(parents=True)
    (path / 'plan.json').write_text(json.dumps({'plan_sha256': 'p1', 'plan': {'folds': rp.FOLDS}, 'raw_input_sha256': RAW,
                                                'code_hash_scheme': rp.CODE_HASH_SCHEME, 'code_sha256': code}),
                                    encoding='utf-8')
    (path / rp.FOLDS[0]['name']).mkdir()
    (path / rp.FOLDS[0]['name'] / 'selection.json').write_text('{}', encoding='utf-8')
    return path


def test_code_edited_mid_run_is_refused_before_a_test_opening_or_gnn_job(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (tmp_path,))
    root = run_root(tmp_path / 'mine', edited=True)
    with pytest.raises(RuntimeError, match='Code changed since the plan was frozen'):
        rp.open_test(root, rp.FOLDS[0]['name'], 'sealed_test', 'p1', done=False)
    assert not (root / 'test_access_log.jsonl').exists()
    spec = {'job_dir': str(root / 'job'), 'source': 's', 'config': {}, 'fixed': {}, 'device': 'cpu', 'seed': 42,
            'mode': 'refit', 'epochs': 1}
    with pytest.raises(RuntimeError, match='Code changed since the plan was frozen'):
        rp.run_gnn_jobs(root, [spec])
    assert not list((root / 'job_specs').iterdir())
    unedited = run_root(tmp_path / 'unedited')
    assert rp.open_test(unedited, rp.FOLDS[0]['name'], 'sealed_test', 'p1', done=False)


def test_test_opening_recheck_matches_another_line_ending_copy_of_events(tmp_path, monkeypatch):
    records = tmp_path / 'records'
    records.mkdir()
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (records,))
    root = run_root(tmp_path / 'mine')
    committed_record(records, {'events.csv': 'crlf'}, 'p2', [rp.FOLDS[0]], opened=[rp.FOLDS[0]['name']])
    monkeypatch.setattr(rp, 'RUN_EVENT_FORMS', {'r', 'crlf'})
    with pytest.raises(RuntimeError, match='overlapping test window'):
        rp.open_test(root, rp.FOLDS[0]['name'], 'sealed_test', 'p1', done=False)


def test_own_root_is_not_reported_as_another_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (tmp_path,))
    root = run_root(tmp_path / 'mine')
    fold = rp.FOLDS[0]['name']
    assert rp.open_test(root, fold, 'sealed_test', 'p1', done=False)
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    plan['raw_input_sha256'] = dict(RAW, **{'category_tree.csv': 'c'})  # no longer an exact reproduction of itself
    (root / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    assert rp.committed_test_conflicts(RAW, 'p1', [rp.FOLDS[0]], tmp_path)
    assert rp.committed_test_conflicts(RAW, 'p1', [rp.FOLDS[0]], tmp_path, exclude=root) == []
    assert rp.open_test(root, fold, 'posthoc_candidate_gamma', 'p1', done=False)


def test_cut_off_test_record_counts_as_every_fold_opened(tmp_path):
    other = {'name': 'B', 'end_fraction': .9, 'test_size': .2}
    committed_record(tmp_path, RAW, 'p1', [FOLD, other], opened=['A'])
    log = tmp_path / 'results_vx' / 'test_access_log.jsonl'
    assert rp.committed_test_conflicts(RAW, 'p2', [other], tmp_path) == []
    log.write_text(log.read_text(encoding='utf-8') + '{"fold": "B", "purp', encoding='utf-8')
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p2', [other], tmp_path)] == ['B']
    disjoint = {'name': 'N', 'end_fraction': .3, 'test_size': .2}
    assert rp.committed_test_conflicts(RAW, 'p2', [disjoint], tmp_path) == []


def test_unreadable_plan_or_own_log_names_the_file(tmp_path):
    committed_record(tmp_path, RAW, 'p1', [FOLD], opened=['A'])
    plan = tmp_path / 'results_vx' / 'plan.json'
    plan.write_text(plan.read_text(encoding='utf-8')[:40], encoding='utf-8')
    with pytest.raises(RuntimeError, match='cannot be read'):
        rp.committed_test_conflicts(RAW, 'p2', [FOLD], tmp_path)
    (tmp_path / 'test_access_log.jsonl').write_text('{"fold": "A", "purp', encoding='utf-8')
    with pytest.raises(RuntimeError, match='cannot be read'):
        rp.access_log(tmp_path, {'fold': 'A', 'plan_sha256': 'p1'}, record=False)


def test_reused_hybrid_grid_is_bit_exact(tmp_path, monkeypatch):
    grid = [{'omega': w} for w in (.1, .2, .3)]
    monkeypatch.setattr(rp, 'HYBRID_GRID', grid)
    full = [{'backbone': 'b', **c, 'ndcg': 0.18943147324069307 + i / 3, 'score': (0.1 + 0.2) * (i + 1) / 7}
            for i, c in enumerate(grid)]
    path = tmp_path / 'validation_hybrid_repeat_b.csv'
    computed, reused = rp.cached_grid(path, lambda: full)
    assert not reused
    rows, reused = rp.cached_grid(path, lambda: pytest.fail('recomputed'))
    assert reused and rows == full
    assert pd.read_csv(path).to_dict('records') != full  # the default parser is not exact on these values
