"""The dirty check lists uncommitted code whatever the git settings, and GNN jobs find their specs from any
working directory."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp
from test_cycle16_guards import git


def test_dirty_check_ignores_untracked_setting_and_renames_out_of_audit_runs(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'status.showUntrackedFiles', 'no')
    (tmp_path / 'audit_runs').mkdir()
    (tmp_path / 'audit_runs' / 'x.py').write_text('')
    (tmp_path / 'audit_runs' / 'my run.txt').write_text('a')
    git(tmp_path, 'add', '-f', 'audit_runs')
    git(tmp_path, 'commit', '-q', '-m', 'base')
    git(tmp_path, 'mv', 'audit_runs/x.py', 'newcode2.py')
    (tmp_path / 'audit_runs' / 'my run.txt').write_text('b')
    (tmp_path / 'newcode.py').write_text('')
    monkeypatch.setattr(rp, 'HERE', tmp_path)
    assert sorted(rp.git_state()['dirty_files']) == ['?? newcode.py', 'A  newcode2.py']


def test_gnn_job_spec_path_is_absolute_for_a_relative_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'myroot').mkdir()
    monkeypatch.setattr(rp, 'require_frozen_code', lambda root: None)
    launched = []

    def popen(args, **kwargs):
        launched.append((args, kwargs))
        raise RuntimeError('stop after launch')
    monkeypatch.setattr(rp.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='stop after launch'):
        rp.run_gnn_jobs('myroot', [{'job_dir': str(tmp_path / 'myroot' / 'job0')}])
    (args, kwargs), = launched
    assert kwargs['cwd'] == rp.HERE and Path(args[-1]).is_absolute() and Path(args[-1]).exists()
