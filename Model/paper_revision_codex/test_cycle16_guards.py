"""Only the run-output folder is exempt from the dirty check before a preregistered run."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp


def test_dirty_check_exempts_only_the_audit_runs_folder(monkeypatch):
    porcelain = '\n'.join([
        '?? Model/paper_revision_codex/audit_runs_patch.py',
        '?? audit_runs_tools.py',
        ' M Model/paper_revision_codex/model.py',
        '?? Model/paper_revision_codex/audit_runs/v8/x.json',
        '?? audit_runs/v8/',
    ]) + '\n'

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=porcelain if 'status' in args else 'x\n')

    monkeypatch.setattr(rp.subprocess, 'run', fake_run)
    assert rp.git_state()['dirty_files'] == [
        '?? Model/paper_revision_codex/audit_runs_patch.py',
        '?? audit_runs_tools.py',
        ' M Model/paper_revision_codex/model.py',
    ]
