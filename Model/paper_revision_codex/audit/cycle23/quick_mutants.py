"""Cycle 23 guards (H87, H94, H95) put back one at a time in a temporary copy, against the tests that cover them."""
import importlib.util, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
tests = ['test_cycle5_guards.py', 'test_cycle19_guards.py', 'test_cycle22_guards.py']
g.args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests]
tmp = Path(tempfile.mkdtemp(prefix='q23_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'tests {tests}']
d = g.fresh(tmp); r = g.run_tests(d)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name in [n for n in g.M if n.split()[0] in ('H87', 'H94', 'H95')]:
    f, old, new = g.M[name]
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, name
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
