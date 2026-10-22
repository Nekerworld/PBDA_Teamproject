"""A git status that cannot be read refuses a preregistered run instead of passing as clean."""
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle4_guards import raw_folder
from test_cycle16_guards import git

sys.path.insert(0, str(HERE / 'report_tools'))
import launch_detached


def test_dirty_check_reads_unescaped_non_ascii_paths(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'core.quotePath', 'false')
    (tmp_path / '회의록.md').write_text('x', encoding='utf-8')
    monkeypatch.setattr(rp, 'HERE', tmp_path)
    assert rp.git_state()['dirty_files'] == ['?? 회의록.md']


def test_detached_launch_refuses_another_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match='Run launch_detached.py from'):
        launch_detached.main(tmp_path / 'run.log', [sys.executable, '-c', 'pass'])
    assert not (tmp_path / 'run.log').exists()


def test_detached_launch_reports_a_command_that_ends_at_once(tmp_path, monkeypatch):
    class Ended:
        pid = 1

        def wait(self, timeout):
            return 3
    monkeypatch.chdir(launch_detached.HERE)
    monkeypatch.setattr(launch_detached.subprocess, 'Popen', lambda *a, **k: Ended())
    with pytest.raises(SystemExit, match='exit code 3'):
        launch_detached.main(tmp_path / 'run.log', [sys.executable, '-c', 'pass'])


def test_failed_git_status_is_not_a_clean_tree(monkeypatch):
    def run(args, **kwargs):
        if 'status' in args:
            raise subprocess.CalledProcessError(129, args)
        return subprocess.CompletedProcess(args, 0, stdout='abc\n')
    monkeypatch.setattr(rp.subprocess, 'run', run)
    state = rp.git_state()
    assert state['commit'] == 'abc' and state['dirty_files'] is None


def test_plan_is_not_created_when_the_dirty_files_cannot_be_read(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, 'git_state', lambda: {'commit': 'abc', 'branch': 'dev', 'dirty_files': None})
    with pytest.raises(RuntimeError, match='needs its code in git'):
        rp.run(tmp_path / 'root', raw_folder(tmp_path / 'raw'))
    assert not (tmp_path / 'root' / 'plan.json').exists()
