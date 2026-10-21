"""An exact reproduction in another root may reopen a fold only with a selection deciding the same, the opening re-check
compares actual test windows (event timestamps), step-completion CSVs are written atomically, and an error in a
background GNN step surfaces with its cause."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp

RAW = {'events.csv': 'r'}
FOLD = {'name': 'A', 'end_fraction': .36, 'test_size': .25}


def record(base, plan_hash, folds, entries, code=None, name='other'):
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    (target / 'plan.json').write_text(json.dumps({'plan_sha256': plan_hash, 'plan': {'folds': folds}, 'raw_input_sha256': RAW,
                                                  'code_hash_scheme': 'sha256-lf', 'code_sha256': code or {}}),
                                      encoding='utf-8')
    (target / 'test_access_log.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries), encoding='utf-8')
    return target


def test_exact_reproduction_reopens_only_with_the_same_selection(tmp_path):
    from test_cycle10_guards import selection
    theirs = selection(tmp_path / 'other' / 'A' / 'selection.json')
    record(tmp_path, 'p1', [FOLD], [{'fold': 'A', 'plan_sha256': 'p1', 'selection_sha256': rp.sha256(theirs)}],
           code=rp.code_hashes())
    assert rp.committed_test_conflicts(RAW, 'p1', [FOLD], tmp_path) == []  # run start: selection not made yet
    same = rp.selection_decision(theirs)
    assert rp.committed_test_conflicts(RAW, 'p1', [FOLD], tmp_path, selections={'A': same}) == []
    changed = rp.selection_decision(selection(tmp_path / 'changed.json', gamma=1.25))
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p1', [FOLD], tmp_path, selections={'A': changed})] == ['A']


def test_overlap_uses_actual_test_windows_when_recorded(tmp_path):
    target = record(tmp_path, 'p1', [FOLD], [{'fold': 'A', 'plan_sha256': 'p1', 'selection_sha256': 's1'}])
    (target / 'A' / 'fold_raw').mkdir(parents=True)
    (target / 'A' / 'fold_raw' / 'fold_raw.json').write_text(json.dumps({'end_cutoff_timestamp': 100}), encoding='utf-8')
    (target / 'A' / 'prepared').mkdir()
    (target / 'A' / 'prepared' / 'manifest.json').write_text(json.dumps({'cutoff_timestamp': 50}), encoding='utf-8')
    assert rp.recorded_test_window(target / 'A') == (50, 100)
    touching = {'name': 'N', 'end_fraction': .4, 'test_size': 1 - .36 / .4}  # nominal window only touches A's
    assert rp.committed_test_conflicts(RAW, 'p2', [touching], tmp_path) == []
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p2', [touching], tmp_path, test_windows={'N': (99, 120)})] == ['A']
    assert rp.committed_test_conflicts(RAW, 'p2', [touching], tmp_path, test_windows={'N': (100, 120)}) == []
    (target / 'A' / 'prepared' / 'manifest.json').unlink()  # no recorded window: touching counts as overlapping
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p2', [touching], tmp_path, test_windows={'N': (100, 120)})] == ['A']


def test_test_opening_passes_its_selection_and_actual_window(tmp_path, monkeypatch):
    from test_cycle8_guards import run_root
    from test_cycle10_guards import selection
    records = tmp_path / 'records'
    records.mkdir()
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (records,))
    root = run_root(tmp_path / 'mine')
    fold = rp.FOLDS[0]
    selection(root / fold['name'] / 'selection.json')
    twin = record(records, 'p1', rp.FOLDS, [{'fold': fold['name'], 'plan_sha256': 'p1', 'selection_sha256': 'other'}],
                  code=rp.code_hashes(), name='twin')
    theirs = selection(twin / fold['name'] / 'selection.json', locked='2026-03-03T00:00:00', job_dir='D:/b', seconds=5.0)
    with pytest.raises(RuntimeError, match='overlapping test window'):
        rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=False)
    (twin / 'test_access_log.jsonl').write_text(json.dumps({'fold': fold['name'], 'selection_sha256': rp.sha256(theirs)}) + '\n',
                                                encoding='utf-8')
    assert rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=False)
    (root / 'test_access_log.jsonl').unlink()
    before = {'name': 'Z', 'end_fraction': fold['end_fraction'] * (1 - fold['test_size']), 'test_size': .25}
    earlier = record(records, 'p2', [before], [{'fold': 'Z', 'plan_sha256': 'p2', 'selection_sha256': 's'}], name='earlier')
    for base, start, end in ((earlier / 'Z', 50, 100), (root / fold['name'], 99, None)):
        (base / 'fold_raw').mkdir(parents=True)
        (base / 'fold_raw' / 'fold_raw.json').write_text(json.dumps({'end_cutoff_timestamp': end}), encoding='utf-8')
        (base / 'prepared').mkdir()
        (base / 'prepared' / 'manifest.json').write_text(json.dumps({'cutoff_timestamp': start}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='overlapping test window'):  # nominal windows only touch; event 99 is shared
        rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=False)


def test_completion_csv_is_never_left_cut_under_its_name(tmp_path, monkeypatch):
    out = tmp_path / 'test_metrics.csv'

    def cut_write(self, path, **kwargs):
        Path(path).write_text('model,nd', encoding='utf-8')
        raise RuntimeError('interrupted')
    monkeypatch.setattr(pd.DataFrame, 'to_csv', cut_write)
    with pytest.raises(RuntimeError, match='interrupted'):
        rp.write_csv_atomic(pd.DataFrame({'model': ['a']}), out)
    assert not out.exists()
    monkeypatch.undo()
    rp.write_csv_atomic(pd.DataFrame({'model': ['a']}), out)
    assert pd.read_csv(out).model.tolist() == ['a'] and not (tmp_path / 'test_metrics.csv.partial').exists()


def test_background_step_error_keeps_its_cause():
    assert rp.alongside(lambda stop: 7, lambda: 3) == (3, 7)

    def fails(stop):
        raise RuntimeError('Code changed since the plan was frozen: [x]')
    with pytest.raises(RuntimeError, match='Code changed since the plan was frozen'):
        rp.alongside(fails, lambda: 3)


def test_access_log_entry_that_is_not_an_opening_names_the_file(tmp_path):
    (tmp_path / 'test_access_log.jsonl').write_text('[1, 2]\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='cannot be read'):
        rp.access_log(tmp_path, {'fold': 'A', 'plan_sha256': 'p1'}, record=False)
