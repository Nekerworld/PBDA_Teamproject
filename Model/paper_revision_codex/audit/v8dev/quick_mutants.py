"""hybrid_ltr leakage and label rules broken one at a time in a temporary copy, against test_hybrid_ltr.py."""
import importlib.util, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_hybrid_ltr.py']
F = 'hybrid_ltr.py'
M = {
    'L1 features read the label sets': (F, "    blocks = []\n    for uid, idx in pool.items():",
                                         "    blocks = []\n    peek = len(context.truth_idx)\n    for uid, idx in pool.items():"),
    'L2 training events at the cutoff accepted': (F, "    if end >= cutoff:", "    if False:"),
    'L3 novel window keeps raw-seen items': (F, "        if policy == 'novel':\n            future", "        if policy == 'never':\n            future"),
    'L4 pair weight from the weakest event': (F, "g.groupby('idx').weight.max()", "g.groupby('idx').weight.min()"),
    'L5 window labels only from the evaluation users': (F, "'test_event_audit.csv', usecols=['visitorid', 'itemid', 'event']",
                                                          "'events_test_original.csv', usecols=['visitorid', 'itemid', 'event']"),
    'L6 positive pairs unweighted': (F, "np.where(weight > 0, weight, 1.)", "np.where(weight > 0, 1., 1.)"),
    'L7 CV recall over reachable truth only': (F, "hit.sum() / len(truth)", "hit.sum() / max(len(context.truth_idx[uid]), 1)"),
    'L8 capping drops users with policy truth': (F, "users = sorted(set(truth) | {rest[i] for i in keep})",
                                                 "users = sorted({rest[i] for i in keep})"),
    'L9 novel pool ignores the exclusion': (F, "        exclude = context.exclusion(uid)\n        parts",
                                            "        exclude = None\n        parts"),
}
tmp = Path(tempfile.mkdtemp(prefix='qv8_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'tests {g.args[6:]}']
d = g.fresh(tmp); r = g.run_tests(d)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name, (f, old, new) in M.items():
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, name
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
lines.append(f'finished_utc {datetime.now(timezone.utc).isoformat()}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
