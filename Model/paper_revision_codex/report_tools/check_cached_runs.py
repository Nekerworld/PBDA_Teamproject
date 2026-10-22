"""Check that every cached fit in a finished root is what run_fold would request from its plan.

The requested identities are rebuilt the way run_fold builds them, from plan.json (GNN grid, fixed
settings, epoch cap, seeds), validation_gnn.json (which top configurations were extended to the
cap-extension epochs) and selection.json (refit configurations and epochs, MF configuration). Each is
compared with what the cached job_result.json / mf_fit.json says was trained, and each GNN job's data
(manifest protocol_id) with the split it must use: prepared/validation for selection, prepared for refit. The specs saved in
job_specs/ are not used: they are the same file the job copied into job_result.json.
Reads no labels and trains nothing.

Usage: python report_tools/check_cached_runs.py --root audit_runs/v7 --output audit_runs/repro/cache_identity_v7.json
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


def protocol_id(folder):
    path = Path(folder) / 'manifest.json'
    return read(path)['protocol_id'] if path.exists() else None


def requested_gnn(fold_dir, plan, selection):
    """(job folder relative to the fold, identity) pairs, in run_fold order."""
    fixed, seed0 = plan['gnn_fixed'], plan['selection_seed']
    jobs = [(Path('gnn_val') / rp.config_name('gnn', c),
             dict(config=c, fixed=fixed, seed=seed0, mode='select', epochs=plan['gnn_max_epochs']))
            for c in plan['gnn_grid']]
    for row in read(fold_dir / 'validation_gnn.json'):
        if Path(row['job_dir']).name.endswith('_cap800'):
            jobs.append((Path('gnn_val') / Path(row['job_dir']).name,
                         dict(config=row['config'], fixed=fixed, seed=seed0, mode='select',
                              epochs=rp.GNN_CAP_EXTENSION_EPOCHS)))
    for seed in plan['seeds']:
        jobs.append((Path('runs') / f'gnn_seed_{seed}', dict(config=selection['gnn']['config'], fixed=fixed, seed=seed,
                                                            mode='refit', epochs=selection['gnn']['epochs'])))
        if selection['hybrid_backbone']['name'] != selection['gnn']['name']:
            backbone = selection['hybrid_backbone']
            jobs.append((Path('runs') / f'hybrid_gnn_seed_{seed}',
                         dict(config=backbone['config'], fixed=fixed, seed=seed, mode='refit', epochs=backbone['epochs'])))
    return jobs


def main(root, output):
    root = Path(root)
    plan = read(root / 'plan.json')['plan']
    if str(rp.GNN_CAP_EXTENSION_EPOCHS) not in plan['gnn_cap_rule']:
        raise ValueError('The plan cap rule does not name the cap-extension epochs used here')
    gnn, mf = [], []
    for fold_dir in sorted(p.parent for p in root.glob('*/selection.json')):
        selection = read(fold_dir / 'selection.json')
        for job, wanted in requested_gnn(fold_dir, plan, selection):
            path = fold_dir / job / 'job_result.json'
            trained = read(path)['spec'] if path.exists() else {}
            differs = [k for k in rp.GNN_JOB_IDENTITY if trained.get(k) != wanted[k]]
            source = fold_dir / 'prepared' / ('validation' if wanted['mode'] == 'select' else '')
            if protocol_id(source) is None or protocol_id(fold_dir / job) != protocol_id(source):
                differs.append('data')
            gnn.append({'job': str((fold_dir / job).relative_to(root)), 'exists': path.exists(), 'differs': differs})
        fits = [(fold_dir / 'val_mf', plan['selection_seed'])] + [(fold_dir / 'runs' / f'seed_{s}', s) for s in plan['seeds']]
        for directory, seed in fits:
            cached = read(directory / 'mf_fit.json')
            wanted = {'seed': seed, 'als': selection['mf']['als'], 'bpr': selection['mf']['bpr'],
                      'bpr_epochs': selection['mf']['bpr_epochs']}
            mf.append({'fit': str(directory.relative_to(root)), 'differs': [k for k in wanted if cached.get(k) != wanted[k]]})
    on_disk = len(list(root.glob('*/gnn_val/*/job_result.json'))) + len(list(root.glob('*/runs/*/job_result.json')))
    summary = {'root': str(root), 'gnn_requested': len(gnn), 'gnn_results_on_disk': on_disk,
               'gnn_mismatches': sum(bool(r['differs']) for r in gnn),
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
