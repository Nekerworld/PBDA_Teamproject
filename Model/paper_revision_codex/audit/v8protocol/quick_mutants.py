"""v8 protocol rules (rolling_protocol re-rankers, stats_tools Holm family) broken one at a time in a temporary
copy, against test_v8_protocol.py."""
import importlib.util, shutil, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

src = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('g', src / 'report_tools' / 'guard_mutation_check.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_v8_protocol.py']
R, S = 'rolling_protocol.py', 'stats_tools.py'
M = {
    'P1 re-ranker labels from the outer test split': (
        R, "window = context if design == 'policy_truth' else ltr.TrainingWindow(val_mf, policy)",
        "window = context if design == 'policy_truth' else ltr.TrainingWindow(Path(val_mf).parent / 'prepared', policy)"),
    'P2 re-rankers trained after the selection is sealed': (
        R, "    if RERANKER and not (fold_dir / 'selection.json').exists():\n        rerankers",
        "    if RERANKER and (fold_dir / 'selection.json').exists():\n        rerankers"),
    'P3 Holm per Hybrid instead of one family': (
        R, "    frame = pd.DataFrame(compare_families(per_user, HYBRIDS, BASELINES))",
        "    frame = pd.concat([pd.DataFrame(compare_families(per_user, [h], BASELINES)) for h in HYBRIDS], ignore_index=True)"),
    'P4 secondary family dropped': (R, "    if len(HYBRIDS) > 1:\n        secondary", "    if False:\n        secondary"),
    'P5 sealed model hash not checked': (R, "        if sha256(path) != entry['sha256']:", "        if False:"),
    'P6 decision ignores the re-rankers': (R, "        if 'rerankers' in s:\n", "        if False:\n"),
    'P7 plan omits the re-rankers': (R, "        **({'reranker': reranker_plan()} if RERANKER else {}),\n", ""),
    'P8 v7 plan gets the re-ranker record': (R, "{'reranker': reranker_plan()} if RERANKER else {}",
                                             "{'reranker': reranker_plan()} if True else {}"),
    'P9 CV grid from the definition-time default': (R, "ltr.FEATURES, grid=ltr.PARAM_GRID,", "ltr.FEATURES,"),
    'P10 resume retrains the re-rankers': (R, "        if path.exists():\n            records[policy] = json.loads",
                                           "        if False:\n            records[policy] = json.loads"),
    'P11 ablations get their own (first-grid) settings': (
        R, "pickle.dump(ltr.fit(frame, y, columns, params, sample_weight=weight), stream)",
        "pickle.dump(ltr.fit(frame, y, columns, params if variant == 'full' else ltr.PARAM_GRID[0], sample_weight=weight), stream)"),
    'P12 joint Holm skipped in compare_families': (
        S, "            for row in compare_family(per_user, reference, competitors, metrics, alpha, samples)]\n    return _apply_holm(rows, alpha)",
        "            for row in compare_family(per_user, reference, competitors, metrics, alpha, samples)]\n    return rows"),
}
only = sys.argv[2:]  # mutant name prefixes to rerun, e.g. P10
if only:
    M = {k: v for k, v in M.items() if k.split()[0] in only}
tmp = Path(tempfile.mkdtemp(prefix='qv8p_'))
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'tests {g.args[6:]}']
d = g.fresh(tmp); r = g.run_tests(d)
lines.append(f'unmutated rc={r.returncode} {g.summary(r.stdout)}')
for name, (f, old, new) in M.items():
    d = g.fresh(tmp); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, name
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = g.run_tests(d)
    lines.append(f'{name}: rc={r.returncode} {g.classify(r.returncode, r.stdout)} | {g.summary(r.stdout)}')
    print(lines[-1], flush=True)
lines.append(f'finished_utc {datetime.now(timezone.utc).isoformat()}')
shutil.rmtree(tmp)
out = Path(sys.argv[1]); out.write_text('\n'.join(lines) + '\n', encoding='utf-8'); print(out.read_text(encoding='utf-8'))
