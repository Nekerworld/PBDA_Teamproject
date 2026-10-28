"""Guard mutants named on the command line (report_tools/guard_mutation_check.py), against its guard tests.

Usage: python audit/v8protocol/guard_subset.py <output.txt> H59 H84"""
import importlib.util, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
names = sys.argv[2:]
tmp = Path(tempfile.mkdtemp(prefix='gsub_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'mutants {names}']
d = g.fresh(tmp); r = g.run_tests(d)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name, (f, old, new) in g.M.items():
    if name.split()[0] not in names:
        continue
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, name
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
    print(lines[-1], flush=True)
lines.append(f'finished_utc {datetime.now(timezone.utc).isoformat()}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
