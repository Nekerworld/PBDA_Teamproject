"""v10 protocol rules (rolling_protocol: combined v8 family + v9 ladder, repeat only, named earlier openings) broken
one at a time in a temporary copy, against test_v10_protocol.py."""
import importlib.util, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_v10_protocol.py']
R = 'rolling_protocol.py'
M = {
    'Q1 named earlier openings not skipped': (
        R, "        if stored['plan_sha256'] in ACKNOWLEDGED_TEST_RECORDS:", "        if False:"),
    'Q2 every earlier opening skipped': (
        R, "        if stored['plan_sha256'] in ACKNOWLEDGED_TEST_RECORDS:", "        if True:"),
    'Q3 v10 ladder scored on the Hybrid backbone': (
        R, "extra = ladder_rankings(context, std, std, alpha, ladder, ladder_tables)",
        "extra = ladder_rankings(context, std, hyb, alpha, ladder, item_tables)"),
    'Q4 ladder trained on the Hybrid backbone': (
        R, "ladder = train_ladder(fold_dir, val_mf, selection['gnn']['job_dir'])",
        "ladder = train_ladder(fold_dir, val_mf, selection['hybrid_backbone']['job_dir'])"),
    'Q5 secondary family over every other Hybrid': (
        R, "compare_families(per_user, list(RERANKER_NAMES.values()), ['Hybrid'])",
        "compare_families(per_user, HYBRIDS[1:], HYBRIDS[:1])"),
    'Q6 ladder.csv reads the v1 Hybrid as L3': (
        R, "        original = {reported: name for name, reported in LADDER_NAMES.items()}", "        original = {}"),
    'Q7 v10 computes both policies': (R, "    POLICIES=('repeat',),\n", ""),
    'Q8 v10 drops the ladder rankings': (
        R, "        rankings.update({name: ranked for name, ranked in extra.items() if name not in rankings})", "        pass"),
    'Q9 ladder rankings keep the hybrid_v9 names': (R, "rankings[LADDER_NAMES.get(name, name)] = v9.rank(", "rankings[name] = v9.rank("),
    'Q10 ladder family on the hybrid_v9 names': (
        R, "        lad = {k: tuple(LADDER_NAMES.get(m, m) for m in v) if isinstance(v, tuple) else LADDER_NAMES.get(v, v)",
        "        lad = {k: tuple(m for m in v) if isinstance(v, tuple) else v"),
    'Q11 plan omits the named openings': (
        R, "        **({'acknowledged_test_records': ACKNOWLEDGED_TEST_RECORDS} if ACKNOWLEDGED_TEST_RECORDS else {}),\n", ""),
    'Q12 v10 without the v1 fusion grid': (R, "    LADDER=True,\n    FUSION=True,\n", "    LADDER=True,\n    FUSION=False,\n"),
    'Q13 ladder trained after the selection is sealed': (
        R, "    if LADDER and not (fold_dir / 'selection.json').exists():\n        ladder",
        "    if LADDER and (fold_dir / 'selection.json').exists():\n        ladder"),
    'Q14 earlier plans get the policies record': (
        R, "{'policies': list(POLICIES)} if tuple(POLICIES) != ALL_POLICIES else {}", "{'policies': list(POLICIES)}"),
}



def fresh(tmp):  # the v10 tests also read the v8/v9 plans and the primary family tool
    d = g.fresh(tmp)
    for v in ('v8', 'v9'):
        (d / 'audit' / f'results_{v}').mkdir(parents=True)
        shutil.copy2(src / 'audit' / f'results_{v}' / 'plan.json', d / 'audit' / f'results_{v}' / 'plan.json')
    (d / 'audit' / 'v10protocol').mkdir()
    shutil.copy2(src / 'audit' / 'v10protocol' / 'primary_family.py', d / 'audit' / 'v10protocol' / 'primary_family.py')
    return d


only = sys.argv[2:]
if only:
    M = {k: v for k, v in M.items() if k.split()[0] in only}
tmp = Path(tempfile.mkdtemp(prefix='qv10p_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'tests {g.args[6:]}']
d = fresh(tmp); r = g.run_tests(d)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
print(lines[-1], flush=True)
for name, (f, old, new) in M.items():
    d = fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, name
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
    print(lines[-1], flush=True)
lines.append(f'finished_utc {datetime.now(timezone.utc).isoformat()}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
