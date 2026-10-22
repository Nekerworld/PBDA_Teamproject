"""Only the run-output folder is exempt from the dirty check before a preregistered run."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp


def git(repo, *args):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=repo, check=True,
                   capture_output=True)


def test_dirty_check_exempts_only_the_audit_runs_folder(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    (tmp_path / 'model.py').write_text('a = 1\n')
    git(tmp_path, 'add', 'model.py')
    git(tmp_path, 'commit', '-q', '-m', 'base')
    (tmp_path / 'model.py').write_text('a = 2\n')
    (tmp_path / 'audit_runs_patch.py').write_text('')
    (tmp_path / 'audit_runs' / 'v8').mkdir(parents=True)
    (tmp_path / 'audit_runs' / 'v8' / 'x.json').write_text('{}')
    monkeypatch.setattr(rp, 'HERE', tmp_path)
    assert sorted(line[3:] for line in rp.git_state()['dirty_files']) == ['audit_runs_patch.py', 'model.py']
