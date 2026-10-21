"""Re-train sealed v6 artifacts with the same seed and compare them (no labels are read).

GNN: re-runs one recorded refit job spec into a new directory.
MF:  re-fits ALS (CPU) and BPR (GPU) with the sealed configuration and seed.
Reports max |difference| of factors/embeddings and top-50 list agreement for the
fold's warm evaluation users.

--blas-threads N (default 1) raises the loaded BLAS/OpenMP pools to N for the MF re-fit, to test
whether a sealed ALS fit could have come from a multi-threaded BLAS.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'  # ALS output depends on the BLAS thread count; match rolling_protocol
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402
from data_protocol import dump_json  # noqa: E402
from experiments import clone_protocol  # noqa: E402
from policy_evaluation import EvaluationContext, factor_vectors, top_k  # noqa: E402


def top_lists(context, vectors, k=50):
    return {u: top_k(v, k) for u, v in vectors.items()}


def agreement(a, b):
    same = [len(set(a[u].tolist()) & set(b[u].tolist())) / max(len(a[u]), 1) for u in a]
    exact = [np.array_equal(a[u], b[u]) for u in a]
    return {'mean_top50_overlap': float(np.mean(same)), 'identical_lists_share': float(np.mean(exact)), 'users': len(a)}


def gnn_check(root, job, out):
    spec = next(json.loads(p.read_text(encoding='utf-8')) for p in (root / 'job_specs').glob('*.json')
                if Path(json.loads(p.read_text(encoding='utf-8'))['job_dir']).resolve() == job.resolve())
    spec = dict(spec, job_dir=str(out.resolve()))
    path = out.parent / (out.name + '_spec.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, spec)
    rp.gnn_job(path)
    context = EvaluationContext(job, 'repeat')
    first, second = rp.load_gnn(job, context), rp.load_gnn(out, context)
    users_a = pd.read_csv(job / 'gnn_user_embeddings.csv').set_index('visitorid').sort_index()
    users_b = pd.read_csv(out / 'gnn_user_embeddings.csv').set_index('visitorid').sort_index()
    return {'artifact': rp.rel(job, root), 'max_abs_user_embedding_diff': float(np.abs(users_a.to_numpy() - users_b.to_numpy()).max()),
            **agreement(top_lists(context, first), top_lists(context, second))}


def mf_check(root, fold, seed, out):
    selection = json.loads((root / fold / 'selection.json').read_text(encoding='utf-8'))
    source = root / fold / 'runs' / f'seed_{seed}'
    if not (out / 'manifest.json').exists():
        clone_protocol(root / fold / 'prepared', out)
    rp.fit_mf(out, selection['mf'], seed)
    context = EvaluationContext(source, 'repeat')
    rows = []
    for name in ('als', 'bpr'):
        a = np.load(source / f'{name}_user_factors.npy')
        b = np.load(out / f'{name}_user_factors.npy')
        users = pd.read_csv(source / 'als_user_mapping.csv').visitorid.to_numpy()
        items = pd.read_csv(source / 'als_item_mapping.csv').itemid.to_numpy()
        ia, ib = np.load(source / f'{name}_item_factors.npy'), np.load(out / f'{name}_item_factors.npy')
        la = top_lists(context, factor_vectors(context, users, a, items, ia))
        lb = top_lists(context, factor_vectors(context, users, b, items, ib))
        rows.append({'artifact': f'{fold}/runs/seed_{seed}/{name}', 'max_abs_user_factor_diff': float(np.abs(a - b).max()),
                     **agreement(la, lb)})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE / 'audit_runs' / 'v6')
    parser.add_argument('--fold', default='F3_early')
    parser.add_argument('--gnn-job', default='runs/hybrid_gnn_seed_42')
    parser.add_argument('--output', type=Path, default=HERE / 'audit_runs' / 'determinism')
    parser.add_argument('--skip-gnn', action='store_true')
    parser.add_argument('--blas-threads', type=int, default=1)
    args = parser.parse_args()
    with threadpoolctl.threadpool_limits(limits=args.blas_threads):
        print('native threads during MF re-fit:', rp.native_threads(), flush=True)
        rows = mf_check(args.root, args.fold, 42, args.output / f'{args.fold}_mf_seed_42')
    for row in rows:
        row['blas_threads'] = args.blas_threads
    if not args.skip_gnn:
        rows.append(gnn_check(args.root, args.root / args.fold / args.gnn_job, args.output / f'{args.fold}_gnn_rerun'))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'determinism.csv', index=False)
    print(frame.to_string(index=False))
