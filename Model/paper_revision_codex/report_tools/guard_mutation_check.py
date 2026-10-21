"""Put the pre-fix behaviour of each cycle-5 guard back into a temporary copy, one at a time, and check
that test_cycle4_guards.py / test_cycle5_guards.py fail. Output: audit/cycle5/guard_mutants.json.

Usage: python report_tools/guard_mutation_check.py
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
src = Path(__file__).resolve().parents[1]
M = {
 'H1 reproduction exempt by plan hash only': ('rolling_protocol.py', "        if stored['plan_sha256'] == plan_hash and code_matches(", "        if stored['plan_sha256'] == plan_hash or not code_matches("),
 'H2 exact window match instead of overlap': ('rolling_protocol.py', "                if max(low, windows[name][0]) < min(high, windows[name][1]) - 1e-9:", "                if (low, high) == windows[name]:"),
 'H3 committed records only': ('rolling_protocol.py', "RESULT_RECORDS = (HERE / 'audit', HERE / 'audit_runs')", "RESULT_RECORDS = (HERE / 'audit',)"),
 'H4 raw input not re-verified': ('rolling_protocol.py', "    if 'raw_input_sha256' in stored and raw_hashes(raw) != stored['raw_input_sha256']:", "    if False:"),
 'H5 selection compared within a purpose only': ('rolling_protocol.py', "        if item['fold'] != entry['fold']:", "        if item['fold'] != entry['fold'] or item['purpose'] != entry['purpose']:"),
 'H6 finished fold skips the log check': ('rolling_protocol.py', "    if not open_test(root, fold['name'], 'sealed_test', plan_hash, (fold_dir / 'test_metrics.csv').exists()):", "    if (fold_dir / 'test_metrics.csv').exists() or not open_test(root, fold['name'], 'sealed_test', plan_hash, False):"),
 'H7 cached MF fit not checked': ('rolling_protocol.py', "        if {k: cached.get(k) for k in wanted} != wanted:", "        if False:"),
 'H8 cached GNN job not checked': ('rolling_protocol.py', "        if any(result['spec'].get(k) != spec[k] for k in GNN_JOB_IDENTITY):", "        if False:"),
 'H9 precision tool imports numpy first': ('report_tools/check_score_precision.py', "import argparse\n", "import argparse\nimport numpy\n"),
 'H10 leakage gate on fold_raw only': ('rolling_protocol.py', "        leakage_main(prepared, raw, gnn=True, output=path, end_fraction=end_fraction)", "        leakage_main(prepared, Path(fold_dir) / 'fold_raw', gnn=True, output=path, end_fraction=None)"),
 'H11 posthoc marker written first': ('rolling_protocol.py', "    frame['note'] = 'Post-hoc on sealed test, seed 42, repeat policy; never used for selection.'\n", "    frame['note'] = 'Post-hoc on sealed test, seed 42, repeat policy; never used for selection.'\n    frame.to_csv(out, index=False)\n"),
 'H12 selection plan hash not checked': ('rolling_protocol.py', "    if selection['plan_sha256'] != plan_hash:\n        raise", "    if False:\n        raise"),
}
tmp = Path(tempfile.mkdtemp(prefix='h_mut_'))
def fresh():
    d = tmp / 'p'
    if d.exists(): shutil.rmtree(d)
    d.mkdir()
    for f in src.glob('*.py'): shutil.copy2(f, d)
    shutil.copy2(src / 'pytest.ini', d)
    for s in ('audit_tools', 'report_tools'): shutil.copytree(src / s, d / s, ignore=shutil.ignore_patterns('__pycache__'))
    for v in ('v6', 'v7'):
        (d / 'audit' / f'results_{v}').mkdir(parents=True)
        for n in ('plan.json', 'test_access_log.jsonl'): shutil.copy2(src / 'audit' / f'results_{v}' / n, d / 'audit' / f'results_{v}' / n)
    return d
args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_cycle4_guards.py', 'test_cycle5_guards.py',
        '--deselect', 'test_cycle4_guards.py::test_verify_plan_code_reports_changes_and_fails_only_when_asked']
rows = []
d = fresh(); r = subprocess.run(args, cwd=d, capture_output=True, text=True); base = r.stdout.strip().splitlines()[-1]; print('unmutated', base, flush=True)
assert r.returncode == 0
for name, (f, old, new) in M.items():
    d = fresh(); t = (d / f).read_text(encoding='utf-8'); assert t.count(old) == 1, (name, t.count(old))
    (d / f).write_text(t.replace(old, new, 1), encoding='utf-8')
    r = subprocess.run(args, cwd=d, capture_output=True, text=True)
    rows.append({'mutant': name, 'file': f, 'result': 'killed' if r.returncode else 'survived', 'summary': r.stdout.strip().splitlines()[-1]})
    print(json.dumps(rows[-1]), flush=True)
shutil.rmtree(tmp)
out = src / 'audit' / 'cycle5' / 'guard_mutants.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'unmutated': base, 'tests': args[5:], 'mutants': rows}, indent=2), encoding='utf-8')
