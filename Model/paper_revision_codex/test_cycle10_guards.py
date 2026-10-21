"""An exact reproduction in another root is matched on what its selection decides, not on the selection file's
bytes; an odd recorded split field leaves a record without a window instead of raising; GNN jobs stop when the
step beside them fails; and an unknown fold name is refused before a plan is made."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle9_guards import FOLD, RAW, record


def selection(path, locked='2026-01-01T00:00:00', job_dir='C:/a', seconds=1.0, gamma=1.1):
    backbone = {'name': 'g', 'config': {'layers': 1}, 'epochs': 10, 'job_dir': job_dir, 'seconds': seconds,
                'repeat': {'ndcg': .1 + seconds}}
    hybrid = {'backbone': 'g', 'alpha': .5, 'strategy': 'balanced_union', 'candidate_k': 500, 'fusion': 'minmax',
              'omega': .7, 'gamma': gamma, 'ndcg': .2 + seconds}
    payload = {'plan_sha256': 'p1', 'fold': 'A', 'locked_utc': locked,
               'mf': {'als': {'factors': 4}, 'bpr': {'factors': 4}, 'bpr_epochs': 2, 'train_seconds': {'ALS': seconds}},
               'gnn': backbone, 'hybrid_backbone': backbone,
               'policies': {'repeat': {'mf_alpha': .9, 'hybrid': hybrid}}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding='utf-8')
    return Path(path)


def test_exact_reproduction_is_matched_on_the_selection_decision(tmp_path):
    records = tmp_path / 'records'
    records.mkdir()
    theirs = selection(records / 'other' / 'A' / 'selection.json')
    record(records, 'p1', [FOLD], [{'fold': 'A', 'plan_sha256': 'p1', 'selection_sha256': rp.sha256(theirs)}],
           code=rp.code_hashes())
    same = selection(tmp_path / 'mine' / 'same.json', locked='2026-02-02T00:00:00', job_dir='D:/b', seconds=9.0)
    assert rp.sha256(same) != rp.sha256(theirs)
    assert rp.selection_decision(same) == rp.selection_decision(theirs)
    assert rp.committed_test_conflicts(RAW, 'p1', [FOLD], records, selections={'A': rp.selection_decision(same)}) == []
    other = selection(tmp_path / 'mine' / 'other.json', gamma=1.25)
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p1', [FOLD], records,
                                                                selections={'A': rp.selection_decision(other)})] == ['A']
    assert rp.selection_decision(tmp_path / 'missing.json') is None
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p1', [FOLD], records, selections={'A': None})] == ['A']
    # the record's selection.json must be the one its log says it opened with
    (records / 'other' / 'test_access_log.jsonl').write_text(json.dumps({'fold': 'A', 'selection_sha256': 'x'}) + '\n',
                                                             encoding='utf-8')
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p1', [FOLD], records,
                                                                selections={'A': rp.selection_decision(same)})] == ['A']


def test_exact_reproduction_from_raw_reaches_its_test_opening(tmp_path, monkeypatch):
    from test_v6_smoke import make_raw, tiny_settings
    raw = tmp_path / 'raw'
    make_raw(raw, events=12000)
    for name in ('PROTOCOL', 'FOLDS', 'POSTHOC', 'PLAN_EXTRA'):
        monkeypatch.setattr(rp, name, rp.V7[name])
    tiny_settings(monkeypatch)
    records = tmp_path / 'audit_runs'
    records.mkdir()
    monkeypatch.setattr(rp, 'RESULT_RECORDS', (records,))
    rp.run(records / 'A', raw)
    rp.run(records / 'B', raw)
    fold = rp.FOLDS[0]['name']
    a, b = (records / r / fold / 'selection.json' for r in 'AB')
    assert rp.sha256(a) != rp.sha256(b) and rp.selection_decision(a) == rp.selection_decision(b)
    assert (records / 'B' / fold / 'test_metrics.csv').exists()


@pytest.mark.parametrize('start, end', [(None, 100), ('1.4e12', 100), (50, 'n/a'), (50.0, 100)])
def test_odd_recorded_split_fields_leave_the_record_without_a_window(tmp_path, start, end):
    target = record(tmp_path, 'p1', [FOLD], [{'fold': 'A', 'plan_sha256': 'p1', 'selection_sha256': 's1'}])
    (target / 'A' / 'fold_raw').mkdir(parents=True)
    (target / 'A' / 'fold_raw' / 'fold_raw.json').write_text(json.dumps({'end_cutoff_timestamp': end}), encoding='utf-8')
    (target / 'A' / 'prepared').mkdir()
    (target / 'A' / 'prepared' / 'manifest.json').write_text(json.dumps({'cutoff_timestamp': start}), encoding='utf-8')
    assert rp.recorded_test_window(target / 'A') is None
    touching = {'name': 'N', 'end_fraction': .4, 'test_size': 1 - .36 / .4}
    assert [c['opened_as'] for c in rp.committed_test_conflicts(RAW, 'p2', [touching], tmp_path,
                                                                test_windows={'N': (100, 120)})] == ['A']


def test_failed_step_stops_the_background_jobs_before_its_error_propagates():
    seen = {}

    def background(stop):
        seen['stopped'] = stop.wait(10)
        return 'done'

    def foreground():
        time.sleep(.1)
        raise RuntimeError('MF fit failed')
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match='MF fit failed'):
        rp.alongside(background, foreground)
    assert seen == {'stopped': True} and time.perf_counter() - start < 5
    assert rp.alongside(lambda stop: 7, lambda: 3) == (3, 7)

    def fails(stop):
        raise RuntimeError('Code changed since the plan was frozen: [x]')
    with pytest.raises(RuntimeError, match='Code changed since the plan was frozen'):
        rp.alongside(fails, lambda: 3)

    def exits(stop):
        raise SystemExit(3)
    with pytest.raises(SystemExit):
        rp.alongside(exits, lambda: 3)


def test_stopped_gnn_jobs_are_terminated(tmp_path, monkeypatch):
    stop = threading.Event()
    launched = []

    class Job:
        def __init__(self, *args, **kwargs):
            self.terminated = False
            launched.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self):
            return 1
    monkeypatch.setattr(rp.subprocess, 'Popen', Job)
    monkeypatch.setattr(rp, 'require_frozen_code', lambda root: None)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) < 3, 'the job loop ignored stop'
        stop.set()
    monkeypatch.setattr(rp.time, 'sleep', sleep)
    specs = [{'job_dir': str(tmp_path / f'job{i}')} for i in range(rp.MAX_PARALLEL_GNN + 1)]
    with pytest.raises(RuntimeError, match='stopped'):
        rp.run_gnn_jobs(tmp_path, specs, stop)
    assert len(launched) == rp.MAX_PARALLEL_GNN and all(job.terminated for job in launched)


def test_unknown_fold_name_is_refused_before_a_plan_is_made(tmp_path):
    with pytest.raises(ValueError, match='Unknown folds'):
        rp.run(tmp_path / 'root', tmp_path / 'raw', folds=[rp.FOLDS[0]['name'].replace('_', '-')])
    assert not (tmp_path / 'root').exists()
