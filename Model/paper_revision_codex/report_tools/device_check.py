"""Train the same GNN and BPR configuration with the same seed on CUDA and on CPU and compare.

Uses one fold's inner-validation split only (never the outer test): embeddings/factors,
top-50 list agreement for warm validation users, and validation NDCG/Recall@50 per device.
"""
import os

for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402
from data_protocol import dump_json  # noqa: E402
from policy_evaluation import EvaluationContext, factor_vectors, top_k  # noqa: E402


def lists(vectors, k=50):
    return {u: top_k(v, k) for u, v in vectors.items()}


def compare(context, a, b, matrix_a, matrix_b):
    la, lb = lists(a), lists(b)
    return {'max_abs_diff': float(np.abs(matrix_a - matrix_b).max()),
            'mean_top50_overlap': float(np.mean([len(set(la[u]) & set(lb[u])) / max(len(la[u]), 1) for u in la])),
            'identical_lists_share': float(np.mean([np.array_equal(la[u], lb[u]) for u in la]))}


def gnn_run(validation, config, epochs, device, out):
    spec = rp.gnn_spec(out, validation, config, rp.SELECTION_SEED, 'refit', epochs)
    spec['device'] = device
    path = out.parent / f'{out.name}_spec.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(path, spec)
    start = time.perf_counter()
    rp.gnn_job(path)
    return time.perf_counter() - start


def main(prepared, reference, fold, epochs, bpr_epochs, out):
    import torch
    torch.set_num_threads(8)  # OMP_NUM_THREADS=1 (kept for ALS parity) would otherwise make CPU torch single-threaded
    validation = Path(prepared) / 'validation'
    selection = json.loads((Path(reference) / fold / 'selection.json').read_text(encoding='utf-8'))
    context = EvaluationContext(validation, 'repeat')
    rows = []
    gnn = {}
    for device in ('cuda', 'cpu'):
        job = Path(out) / f'gnn_{device}'
        seconds = gnn_run(validation, selection['gnn']['config'], epochs, device, job)
        gnn[device] = (rp.load_gnn(job, context),
                       pd.read_csv(job / 'gnn_user_embeddings.csv').set_index('visitorid').sort_index().to_numpy())
        rows.append({'model': 'GNN', 'device': device, 'epochs': epochs, 'seconds': seconds,
                     **context.metrics(context.rank_scores(gnn[device][0], rp.PRIMARY_K), rp.PRIMARY_K)})
    gnn_cmp = compare(context, gnn['cuda'][0], gnn['cpu'][0], gnn['cuda'][1], gnn['cpu'][1])
    matrix, users, items = rp.interaction_matrix(validation)
    bpr = {}
    import learning
    for device in ('cuda', 'cpu'):
        start = time.perf_counter()
        result = learning.train_weighted_bpr(matrix.copy(), selection['mf']['bpr']['factors'],
                                             selection['mf']['bpr']['regularization'], bpr_epochs, rp.SELECTION_SEED,
                                             device, True, negative_mode=rp.BPR_NEGATIVES,
                                             learning_rate=selection['mf']['bpr']['learning_rate'])
        vectors = factor_vectors(context, users, result.user_factors, items, result.item_factors)
        bpr[device] = (vectors, np.asarray(result.user_factors))
        rows.append({'model': 'BPR', 'device': device, 'epochs': bpr_epochs, 'seconds': time.perf_counter() - start,
                     **context.metrics(context.rank_scores(vectors, rp.PRIMARY_K), rp.PRIMARY_K)})
    bpr_cmp = compare(context, bpr['cuda'][0], bpr['cpu'][0], bpr['cuda'][1], bpr['cpu'][1])
    frame = pd.DataFrame(rows)
    frame['score'] = (frame.ndcg + frame.recall) / 2
    result = {'fold': fold, 'split': 'inner validation (outer test not read)', 'gnn_config': selection['gnn']['config'],
              'bpr_config': selection['mf']['bpr'], 'environment': rp.environment(),
              'runs': frame.to_dict('records'), 'gnn_cuda_vs_cpu': gnn_cmp, 'bpr_cuda_vs_cpu': bpr_cmp}
    dump_json(Path(out) / f'device_check_{fold}.json', result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=HERE / 'audit' / 'results_v6')
    parser.add_argument('--fold', default='F3_early')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bpr-epochs', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.prepared, args.reference, args.fold, args.epochs, args.bpr_epochs, args.output)
