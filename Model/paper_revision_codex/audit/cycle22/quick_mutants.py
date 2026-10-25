import importlib.util, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
tests = ['test_cycle22_guards.py']
tmp = Path(tempfile.mkdtemp(prefix='q22_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'tests {tests}']
args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests]
d = g.fresh(tmp); r = subprocess.run(args, cwd=d, capture_output=True, text=True)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name in [n for n in g.M if int(n.split()[0][1:]) >= 78]:
    f, old, new = g.M[name]
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = subprocess.run(args, cwd=d, capture_output=True, text=True)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
d = g.fresh(tmp)
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_cycle19_guards.py', 'test_cycle20_guards.py', 'test_cycle21_guards.py', 'test_compare_outputs.py', 'test_v6_smoke.py', 'test_cycle3_consistency.py'],
                   cwd=d, capture_output=True, text=True)
lines.append(f'other (test_cycle19/20/21_guards, test_compare_outputs, test_v6_smoke, test_cycle3_consistency) rc={r.returncode} {g.summary(r.stdout)}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
