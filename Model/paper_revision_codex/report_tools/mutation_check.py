"""Mutation check: inject one known defect at a time into a scratch copy and record which tests fail.

The project files are copied to <scratch> first (default audit_runs/mutation_src, git-ignored),
so a running preregistered experiment never imports a mutated module. A mutant is "killed"
when pytest exits non-zero. The unmutated copy must pass both suites, otherwise kills mean nothing.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OLD = ['test_v6_protocol.py', 'test_v6_smoke.py', 'test_revision.py', 'test_codex_audit.py']
NEW = ['test_cycle3_consistency.py']
MUTANTS = [
    ('M01 NDCG ideal DCG over reachable truth only', 'policy_evaluation.py',
     "idcg = float(self.discounts[:min(k, len(self.truth[uid]))].sum())",
     "idcg = float(self.discounts[:min(k, max(len(relevant), 1))].sum())"),
    ('M02 Recall denominator = reachable truth', 'policy_evaluation.py',
     "'recall': hits / len(self.truth[uid])", "'recall': hits / max(len(relevant), 1)"),
    ('M03 cold users dropped from denominator', 'policy_evaluation.py',
     "for uid in self.users:\n            ranked", "for uid in [u for u in self.users if u in rankings]:\n            ranked"),
    ('M04 novel truth uses filtered (not raw) history', 'policy_evaluation.py',
     "truth = {u: original[u] - self.raw_seen.get(u, set()) for u in users}",
     "truth = {u: original[u] - set(map(int, train.loc[train.visitorid == u, 'itemid'])) for u in users}"),
    ('M05 novel exclusion disabled', 'policy_evaluation.py',
     "return self.seen_idx.get(uid) if self.policy == 'novel' else None", "return None"),
    ('M06 Holm without monotone step-down', 'stats_tools.py',
     "running = max(running, min(1.0, (m - rank) * p[index]))", "running = min(1.0, (m - rank) * p[index])"),
    ('M07 sign-flip one-sided', 'stats_tools.py',
     "extreme += int((np.abs((signs * values).mean(axis=1)) >= observed - 1e-15).sum())",
     "extreme += int(((signs * values).mean(axis=1) >= observed - 1e-15).sum())"),
    ('M08 paired comparison aligned by position', 'stats_tools.py',
     "other = per_user[model].reindex(base.index)", "other = per_user[model].set_axis(base.index[:len(per_user[model])])"),
    ('M09 selection grid: gamma boosts unavailable items', 'rolling_protocol.py',
     "fused = fused * np.where(available, config['gamma'], 1.)", "fused = fused * np.where(available, 1., config['gamma'])"),
    ('M10 selection grid: RRF weights swapped', 'rolling_protocol.py',
     "(w / (60 + mr) + (1 - w) / (60 + gr))", "((1 - w) / (60 + mr) + w / (60 + gr))"),
    ('M11 selection grid: MF not min-max normalised', 'rolling_protocol.py',
     "cached[uid] = (minmax(mf), minmax(g)", "cached[uid] = (mf, minmax(g)"),
    ('M12 split puts cutoff-timestamp events in train', 'data_protocol.py',
     "train = ordered.loc[ordered.timestamp < cutoff].copy()", "train = ordered.loc[ordered.timestamp <= cutoff].copy()"),
    ('M13 available = first (not last) value', 'data_protocol.py',
     '["itemid", "property"], keep="last")', '["itemid", "property"], keep="first")'),
    ('M14 properties after the cutoff used', 'data_protocol.py',
     "past_props = properties.loc[(properties.timestamp < cutoff) &", "past_props = properties.loc[(properties.timestamp < 10**15) &"),
    ('M15 MF selection takes the minimum', 'rolling_protocol.py',
     "part = frame.loc[frame.family == family].sort_values('repeat_score', ascending=False, kind='stable')",
     "part = frame.loc[frame.family == family].sort_values('repeat_score', ascending=True, kind='stable')"),
    ('M16 best_row takes the minimum', 'rolling_protocol.py',
     "frame = pd.DataFrame(rows).sort_values('score', ascending=False, kind='stable')",
     "frame = pd.DataFrame(rows).sort_values('score', ascending=True, kind='stable')"),
    ('M17 context available flag inverted', 'policy_evaluation.py',
     "has_available.eq(1).to_numpy()", "has_available.eq(0).to_numpy()"),
]


def copy_project(scratch):
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / 'audit_tools').mkdir(parents=True)
    for path in list(HERE.glob('*.py')) + [HERE / 'pytest.ini']:
        shutil.copy2(path, scratch / path.name)
    for path in (HERE / 'audit_tools').glob('*.py'):
        shutil.copy2(path, scratch / 'audit_tools' / path.name)
    shutil.copytree(HERE / 'audit' / 'results_v6', scratch / 'audit' / 'results_v6')


def run(scratch, tests):
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    result = subprocess.run([sys.executable, '-m', 'pytest', '-x', '-q', '-p', 'no:cacheprovider', *tests],
                            cwd=scratch, capture_output=True, text=True, env=env)
    return 'killed' if result.returncode else 'survived'


def main(scratch, output):
    copy_project(scratch)
    baseline = {'old': run(scratch, OLD), 'new': run(scratch, NEW)}
    if baseline != {'old': 'survived', 'new': 'survived'}:
        raise RuntimeError(f'Unmutated copy fails its tests: {baseline}')
    rows = []
    for name, file, old, new in MUTANTS:
        path = scratch / file
        original = path.read_text(encoding='utf-8')
        if original.count(old) != 1:
            raise ValueError(f'{name}: target text found {original.count(old)} times')
        path.write_text(original.replace(old, new), encoding='utf-8')
        try:
            rows.append({'mutant': name, 'file': file, 'old_tests': run(scratch, OLD), 'new_tests': run(scratch, NEW)})
        finally:
            path.write_text(original, encoding='utf-8')
        rows[-1]['killed_by_any'] = 'killed' in (rows[-1]['old_tests'], rows[-1]['new_tests'])
        print(json.dumps(rows[-1]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'baseline_unmutated': baseline, 'old_suite': OLD, 'new_suite': NEW,
                                  'mutants': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=HERE / 'audit_runs' / 'mutation_src')
    parser.add_argument('--output', type=Path, default=HERE / 'audit_runs' / 'mutation_results.json')
    args = parser.parse_args()
    main(args.scratch.resolve(), args.output.resolve())
