"""A test opening re-checks other roots, an exact reproduction needs the same raw files, a cut-short
Hybrid grid cache is recomputed, and a preregistered plan cannot be created without git."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle4_guards import committed_record, raw_folder


def test_test_opening_rechecks_roots_that_opened_the_window_after_this_run_started(tmp_path, monkeypatch):
    fold = rp.FOLDS[0]
    raw = {'events.csv': 'r'}
    root = tmp_path / 'mine'
    (root / fold['name']).mkdir(parents=True)
    (root / fold['name'] / 'selection.json').write_text('{}', encoding='utf-8')
    (root / 'plan.json').write_text(json.dumps({'plan_sha256': 'p1', 'raw_input_sha256': raw}), encoding='utf-8')
    records = tmp_path / 'records'
    records.mkdir()
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (records,))
    assert rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=False)
    (root / 'test_access_log.jsonl').unlink()
    committed_record(records, raw, 'p2', [fold], opened=[fold['name']], name='parallel')
    with pytest.raises(RuntimeError, match='overlapping test window'):
        rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=False)
    assert not (root / 'test_access_log.jsonl').exists()
    assert rp.open_test(root, fold['name'], 'sealed_test', 'p1', done=True) is False  # outputs exist: no new opening


def test_exact_reproduction_needs_the_same_raw_files(tmp_path):
    fold = {'name': 'A', 'end_fraction': .6, 'test_size': .25}
    stored = {'events.csv': 'r', 'item_properties_part1.csv': 'x'}
    committed_record(tmp_path, stored, 'p1', [fold], opened=['A'], code=rp.code_hashes())
    assert rp.committed_test_conflicts(stored, 'p1', [fold], tmp_path) == []
    edited = dict(stored, **{'item_properties_part1.csv': 'y'})  # drives `available`, hence gamma
    assert [c['fold'] for c in rp.committed_test_conflicts(edited, 'p1', [fold], tmp_path)] == ['A']


def test_cut_short_hybrid_grid_cache_is_recomputed(tmp_path, monkeypatch):
    grid = [{'omega': w} for w in (.1, .2, .3, .4)]
    monkeypatch.setattr(rp, 'HYBRID_GRID', grid)
    path = tmp_path / 'validation_hybrid_repeat_b.csv'
    full = [{'backbone': 'b', **c, 'score': i} for i, c in enumerate(grid)]
    calls = []

    def compute():
        calls.append(1)
        return full
    pd.DataFrame(full[:2]).to_csv(path, index=False)
    rows, reused = rp.cached_grid(path, compute)
    assert (reused, len(rows), len(calls)) == (False, 4, 1)
    assert len(pd.read_csv(path)) == 4 and not path.with_name(path.name + '.partial').exists()
    rows, reused = rp.cached_grid(path, compute)
    assert (reused, len(rows), len(calls)) == (True, 4, 1)


def test_plan_is_not_created_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'git_state', lambda: {'commit': None, 'branch': None, 'dirty_files': []})
    with pytest.raises(RuntimeError, match='needs its code in git'):
        rp.run(tmp_path / 'root', raw_folder(tmp_path / 'raw'))
    assert not (tmp_path / 'root' / 'plan.json').exists()
