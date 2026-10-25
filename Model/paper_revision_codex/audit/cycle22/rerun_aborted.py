import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(src / 'report_tools'))
import guard_mutation_check as g  # noqa: E402

names = sys.argv[1:]
print('started_utc', datetime.now(timezone.utc).isoformat(), 'head', g.git('rev-parse', 'HEAD'), flush=True)
tmp = Path(tempfile.mkdtemp(prefix='h_rerun_'))
for name in names:
    key = next(k for k in g.M if k.split()[0] == name)
    f, old, new = g.M[key]
    d = g.fresh(tmp)
    t = (d / f).read_text(encoding='utf-8')
    assert t.count(old) == 1, (key, t.count(old))
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = subprocess.run(g.args, cwd=d, capture_output=True, text=True)
    print(json.dumps({'mutant': key, 'result': g.classify(r.returncode, r.stdout), 'returncode': r.returncode,
                      'summary': g.summary(r.stdout), 'stdout_chars': len(r.stdout or ''),
                      'stderr_tail': (r.stderr or '')[-1500:]}), flush=True)
print('finished_utc', datetime.now(timezone.utc).isoformat(), flush=True)
