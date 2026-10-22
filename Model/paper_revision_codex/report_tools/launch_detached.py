"""Start a long run (e.g. `rolling_protocol.py run`) as a process that survives the launching shell.

On Windows the run must not share the shell's console: Ctrl+C / console-close events killed two v6
attempts, and with DETACHED_PROCESS the Intel Fortran runtime inside the BLAS/LAPACK stack aborted
with "forrtl: error (200): window-CLOSE event". The flags below give the child its own process group,
take it out of the shell's job object and give it a hidden console of its own; the Fortran runtime's
console handler is disabled as well. Thread variables are pinned to 1 because ALS depends on them.

Usage: python report_tools/launch_detached.py --log audit_runs/v7_run.log -- \
           python -u rolling_protocol.py run --root audit_runs/v7 --raw ../../data --protocol v7
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CREATE_NEW_PROCESS_GROUP, CREATE_BREAKAWAY_FROM_JOB, CREATE_NO_WINDOW = 0x200, 0x01000000, 0x08000000
PINNED = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
          'NUMEXPR_NUM_THREADS': '1', 'FOR_DISABLE_CONSOLE_CTRL_HANDLER': '1', 'PYTHONUNBUFFERED': '1'}
STARTUP_SECONDS = 10


def main(log, command):
    # The command runs in HERE; from another folder its relative paths (script, --root, --raw) and --log would
    # resolve against different folders, and a --root could land outside audit_runs/ where other roots are checked.
    if Path.cwd().resolve() != HERE:
        raise SystemExit(f'Run launch_detached.py from {HERE}: the command runs there, so its relative paths must too')
    log = Path(log).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW if os.name == 'nt' else 0
    with open(log, 'ab') as handle:
        process = subprocess.Popen(command, cwd=HERE, env={**os.environ, **PINNED}, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, creationflags=flags,
                                   start_new_session=os.name != 'nt')
    try:
        code = process.wait(timeout=STARTUP_SECONDS)
    except subprocess.TimeoutExpired:
        print(f'pid {process.pid}, log {log}')
        return
    if code:
        raise SystemExit(f'The command ended within {STARTUP_SECONDS} s with exit code {code}; see {log}')
    print(f'The command finished within {STARTUP_SECONDS} s; log {log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('command required after --')
    if command[0] == 'python':
        command[0] = sys.executable
    main(args.log, command)
