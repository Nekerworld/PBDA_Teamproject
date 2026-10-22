"""Preprocessing is only marked complete at its very end, an empty fold list is refused, and a GNN job that
ended before it could join the job object reports its own failure."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_protocol
import rolling_protocol as rp
from test_v6_smoke import make_raw


def test_interrupted_preprocessing_is_not_taken_as_complete(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw)
    fold = rp.FOLDS[-1]

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(data_protocol.UnifiedPreprocessor, 'finalize_outer', interrupted)
    with pytest.raises(KeyboardInterrupt):
        rp.prepare_fold(fold, tmp_path / 'root', raw)
    prepared = tmp_path / 'root' / fold['name'] / 'prepared'
    assert (prepared / 'validation' / 'manifest.json').exists() and not (prepared / 'manifest.json').exists()
    monkeypatch.undo()
    with pytest.raises(FileExistsError, match='Incomplete preprocessing'):
        rp.prepare_fold(fold, tmp_path / 'root', raw)
    prepared, _ = rp.prepare_fold(fold, tmp_path / 'fresh', raw)
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['input_hashes']) == set(rp.RAW_FILES) and 'raw_events' in manifest
    assert 'raw_valid' in set(pd.read_csv(prepared / 'dataset_statistics.csv').iloc[:, 0].astype(str))
    data_protocol.require_protocol(prepared)
    assert not (prepared / 'manifest.json.partial').exists()


def test_empty_fold_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match='empty fold list'):
        rp.run(tmp_path / 'root', tmp_path / 'raw', folds=[])
    assert not (tmp_path / 'root').exists()
    cli = subprocess.run([sys.executable, str(HERE / 'rolling_protocol.py'), 'run', '--root', str(tmp_path / 'r2'),
                          '--raw', str(tmp_path / 'raw'), '--folds'], capture_output=True, text=True)
    assert cli.returncode == 2 and 'expected at least one argument' in cli.stderr
    assert not (tmp_path / 'r2').exists()


class EndedJob:
    returncode = 1

    def __init__(self, *args, **kwargs):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self):
        return self.returncode


class RunningJob(EndedJob):
    returncode = None
    terminated = False

    def terminate(self):
        self.terminated = True


def refuse(process):
    raise PermissionError(5, 'Access is denied')


def test_job_that_ended_before_joining_reports_its_own_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'require_frozen_code', lambda root: None)
    monkeypatch.setattr(rp, 'kill_with_parent', refuse)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) < 5, 'the job loop kept polling a job whose launch should have failed'
    monkeypatch.setattr(rp.time, 'sleep', sleep)
    monkeypatch.setattr(rp.subprocess, 'Popen', EndedJob)
    with pytest.raises(RuntimeError, match='GNN job failed'):
        rp.run_gnn_jobs(tmp_path, [{'job_dir': str(tmp_path / 'job0')}])
    launched = []
    monkeypatch.setattr(rp.subprocess, 'Popen', lambda *a, **k: launched.append(RunningJob()) or launched[-1])
    with pytest.raises(PermissionError):
        rp.run_gnn_jobs(tmp_path, [{'job_dir': str(tmp_path / 'job1')}])
    assert launched[0].terminated


@pytest.mark.skipif(os.name != 'nt', reason='job objects are Windows only')
def test_an_ended_process_cannot_join_the_job_object():
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    with pytest.raises(OSError):
        rp.kill_with_parent(process)
