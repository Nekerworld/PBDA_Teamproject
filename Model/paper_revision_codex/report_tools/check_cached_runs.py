"""Apply the resume-time cache checks of rolling_protocol (fit_mf, run_gnn_jobs) to a finished root.

Every GNN job spec in <root>/job_specs must match its job_result.json on GNN_JOB_IDENTITY, and every
cached MF fit (val_mf with the selection seed, runs/seed_* with their seed) must match selection.json.
Reads no labels and trains nothing.

Usage: python report_tools/check_cached_runs.py --root audit_runs/v7 --output audit/cycle5/cache_identity_v7.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(root, output):
    root = Path(root)
    gnn = []
    for path in sorted((root / 'job_specs').glob('*.json')):
        spec = read(path)
        result = read(Path(spec['job_dir']) / 'job_result.json')['spec']
        gnn.append({'spec': path.name, 'job': Path(spec['job_dir']).name,
                    'differs': [k for k in rp.GNN_JOB_IDENTITY if result.get(k) != spec[k]]})
    mf = []
    for fold_dir in sorted(p.parent for p in root.glob('*/selection.json')):
        selection = read(fold_dir / 'selection.json')['mf']
        fits = [(fold_dir / 'val_mf', rp.SELECTION_SEED)]
        fits += [(d, int(d.name.split('_')[1])) for d in sorted((fold_dir / 'runs').glob('seed_*'))]
        for directory, seed in fits:
            cached = read(directory / 'mf_fit.json')
            wanted = {'seed': seed, 'als': selection['als'], 'bpr': selection['bpr'], 'bpr_epochs': selection['bpr_epochs']}
            mf.append({'fit': str(directory.relative_to(root)), 'differs': [k for k in wanted if cached.get(k) != wanted[k]]})
    summary = {'root': str(root), 'gnn_jobs': len(gnn), 'gnn_mismatches': sum(bool(r['differs']) for r in gnn),
               'mf_fits': len(mf), 'mf_mismatches': sum(bool(r['differs']) for r in mf), 'gnn': gnn, 'mf': mf}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print({k: v for k, v in summary.items() if k not in ('gnn', 'mf')})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
