"""An interrupted protocol clone leaves nothing a resume would take as complete, and GNN jobs end with the run
that started them even when it is killed outright."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiments
import rolling_protocol as rp
from data_protocol import VERSION, sha256


def protocol(folder, names=('events_train_clean.csv', 'events_test.csv', 'items.csv')):
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text(f'{name}\n', encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps({'version': VERSION, 'audit_hashes': {},
                                                      'hashes': {n: sha256(folder / n) for n in names}}),
                                          encoding='utf-8')
    return folder


def test_interrupted_clone_leaves_no_destination_and_is_redone(tmp_path, monkeypatch):
    source = protocol(tmp_path / 'source')
    protocol(source / 'validation')
    link, made = os.link, []

    def cut_link(src, dst):
        if len(made) == 5:
            raise OSError('interrupted')
        made.append(dst)
        link(src, dst)
    monkeypatch.setattr(experiments.os, 'link', cut_link)
    target = tmp_path / 'runs' / 'job'
    with pytest.raises(OSError):
        experiments.clone_protocol(source, target)
    assert not target.exists()
    monkeypatch.undo()
    experiments.clone_protocol(source, target)
    for folder in (target, target / 'validation'):
        assert sorted(p.name for p in folder.iterdir() if p.is_file()) == sorted(
            ['manifest.json', 'events_train_clean.csv', 'events_test.csv', 'items.csv'])
    assert not (tmp_path / 'runs' / 'job.partial').exists() and (source / 'items.csv').exists()


class FakeJob:
    returncode = 0

    def __init__(self, *args, **kwargs):
        pass

    def poll(self):
        return 0

    def terminate(self):
        pass

    def wait(self):
        return 0


def test_every_gnn_job_is_tied_to_the_run_and_a_failed_launch_closes_its_log(tmp_path, monkeypatch):
    tied = []
    monkeypatch.setattr(rp, 'require_frozen_code', lambda root: None)
    monkeypatch.setattr(rp, 'kill_with_parent', tied.append)
    monkeypatch.setattr(rp.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(rp.subprocess, 'Popen', FakeJob)
    specs = [{'job_dir': str(tmp_path / f'job{i}')} for i in range(2)]
    with pytest.raises(RuntimeError, match='GNN job failed'):  # no job_result.json is written by the fake
        rp.run_gnn_jobs(tmp_path, specs)
    assert len(tied) == 2 and all(isinstance(job, FakeJob) for job in tied)
    opened = []
    real_open = open

    def tracked_open(*args, **kwargs):
        opened.append(real_open(*args, **kwargs))
        return opened[-1]

    def no_launch(*args, **kwargs):
        raise OSError('cannot start')
    monkeypatch.setattr(rp, 'open', tracked_open, raising=False)
    monkeypatch.setattr(rp.subprocess, 'Popen', no_launch)
    with pytest.raises(OSError, match='cannot start'):
        rp.run_gnn_jobs(tmp_path, [{'job_dir': str(tmp_path / 'job9')}])
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.skipif(os.name != 'nt', reason='job objects are Windows only')
def test_gnn_job_dies_with_a_killed_parent(tmp_path):
    beat = tmp_path / 'beat.txt'
    child = f"import time\nwhile True:\n    open(r'{beat}', 'a').write('.')\n    time.sleep(.05)\n"
    parent = (f"import subprocess, sys, time\nsys.path.insert(0, r'{HERE}')\nimport rolling_protocol as rp\n"
              f"p = subprocess.Popen([sys.executable, '-c', {child!r}])\nrp.kill_with_parent(p)\n"
              "print(p.pid, flush=True)\ntime.sleep(120)\n")
    run = subprocess.Popen([sys.executable, '-c', parent], stdout=subprocess.PIPE, text=True)
    pid = None
    try:
        pid = int(run.stdout.readline())
        deadline = time.time() + 30
        while (not beat.exists() or beat.stat().st_size < 3) and time.time() < deadline:
            time.sleep(.1)
        assert beat.exists()
        run.kill()  # TerminateProcess: no Python cleanup runs in the parent
        run.wait()
        time.sleep(1)
        size = beat.stat().st_size
        time.sleep(1)
        assert beat.stat().st_size == size, 'the GNN job outlived its killed parent'
    finally:
        if run.poll() is None:
            run.kill()
        if pid is not None:
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
