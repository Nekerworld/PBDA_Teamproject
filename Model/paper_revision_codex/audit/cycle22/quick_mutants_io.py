import ast, importlib.util, os, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'PYTHONIOENCODING={os.environ.get("PYTHONIOENCODING")}']

# every target occurs exactly once in its file
M = g.M
counts = {k: (src / f).read_text(encoding='utf-8').count(old) for k, (f, old, new) in M.items()}
lines.append(f'targets {len(M)}, all exactly once: {all(c == 1 for c in counts.values())}')

run = lambda d, tests: subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], cwd=d,
                                      capture_output=True, encoding='utf-8', errors='replace',
                                      env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
tmp = Path(tempfile.mkdtemp(prefix='qio_'))
tests = ['test_cycle22_guards.py']
lines.append(f'tests {tests}')
d = g.fresh(tmp); r = run(d, tests)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name in [n for n in g.M if int(n.split()[0][1:]) >= 88]:
    f, old, new = g.M[name]
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = run(d, tests)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')

# the four mutants the Cycle 22 run recorded as aborted, through the fixed run_tests (full guard test list)
lines.append('rerun through the fixed guard_mutation_check.run_tests (guard test list):')
for short in ('H11', 'H69', 'H80', 'H86'):
    name = next(n for n in g.M if n.split()[0] == short)
    f, old, new = g.M[name]
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
shutil.rmtree(tmp)
lines.append(f'finished_utc {datetime.now(timezone.utc).isoformat()}')
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
