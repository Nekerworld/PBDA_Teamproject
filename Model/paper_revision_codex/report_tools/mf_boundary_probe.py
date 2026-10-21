"""Validation-only probe beyond the v6 ALS/BPR grid boundaries (never reads the outer test).

v6 selected ALS factors=256 (grid max), alpha=10 (max), iterations=32 (min) and
BPR lr=.005 (max), reg=.003 (min), epochs=256 (max) on every fold. This probe
asks whether inner-validation score keeps improving past those edges, varying
one axis at a time around each fold's sealed ALS configuration. It does not
change any sealed v6 selection or test result.
"""
import os

# ALS output depends on the BLAS thread count; match rolling_protocol (forced, not defaulted)
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402
from policy_evaluation import EvaluationContext, POLICIES, factor_vectors, selection_score  # noqa: E402

ALS_PROBE_CHANGES = [{}, dict(factors=512), dict(alpha=20.), dict(alpha=40.), dict(iterations=16),
                     dict(iterations=8), dict(factors=512, alpha=40., iterations=16)]
BPR_PROBE = [dict(factors=128, learning_rate=lr, regularization=r) for lr in (.005, .01) for r in (.003, .0003)]
BPR_CHECKPOINTS = (64, 128, 256, 384, 512)


def probe(root, fold, out):
    val_dir = root / fold / 'prepared' / 'validation'
    contexts = {p: EvaluationContext(val_dir, p) for p in POLICIES}
    matrix, users, items = rp.interaction_matrix(val_dir)
    selection = json.loads((root / fold / 'selection_mf.json').read_text(encoding='utf-8'))
    rows = []

    def record(family, config, scores, **extra):
        rows.append({'fold': fold, 'family': family, **config, **extra,
                     'repeat_score': selection_score(scores['repeat']),
                     **{f'{p}_{m}': v for p, d in scores.items() for m, v in d.items()}})
        print(rows[-1], flush=True)

    for change in ALS_PROBE_CHANGES:
        config = {**selection['als'], **change}
        uf, itf = rp.train_als(matrix, config, rp.SELECTION_SEED)
        record('ALS', config, rp.evaluate_vectors(contexts, factor_vectors(contexts['repeat'], users, uf, items, itf)))
    for config in BPR_PROBE:
        def callback(epoch, uf, itf, config=config):
            record('BPR', config, rp.evaluate_vectors(contexts, factor_vectors(contexts['repeat'], users, uf, items, itf)),
                   epochs=epoch)
        rp.train_bpr(matrix, config, max(BPR_CHECKPOINTS), rp.SELECTION_SEED, BPR_CHECKPOINTS, callback)
    frame = pd.DataFrame(rows)
    sealed = selection['validation']
    frame['sealed_repeat_score'] = frame.family.map({f: sealed[f]['repeat_score'] for f in ('ALS', 'BPR')})
    frame['gain_over_sealed'] = frame.repeat_score - frame.sealed_repeat_score
    frame.to_csv(out / f'{fold}_mf_boundary_probe.csv', index=False)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE / 'audit_runs' / 'v6')
    parser.add_argument('--folds', nargs='+', default=[f['name'] for f in rp.FOLDS])
    parser.add_argument('--output', type=Path, default=HERE / 'audit_runs' / 'boundary_probe')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [probe(args.root, fold, args.output) for fold in args.folds]
    pd.concat(frames).to_csv(args.output / 'mf_boundary_probe.csv', index=False)
