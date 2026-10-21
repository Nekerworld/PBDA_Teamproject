"""Does ranking ALS/BPR on min-maxed float32 scores (test path, FactorScores) change the lists that the
raw factor scores (selection path, evaluate_vectors) give? Inner validation only; the outer test is not read.

Usage: python report_tools/check_score_precision.py --mf-dir audit_runs/v7/F4_unseen/val_mf --output <json>
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402  (first: it forces single-threaded BLAS before numpy loads)
import numpy as np  # noqa: E402
from policy_evaluation import POLICIES, EvaluationContext, FactorScores  # noqa: E402


def main(mf_dir, output):
    rows = []
    for policy in POLICIES:
        context = EvaluationContext(mf_dir, policy)
        als, bpr = rp.load_factors(mf_dir, context)
        scaled = FactorScores(context, {u: als[u] for u in context.active}, {u: bpr[u] for u in context.active})
        for model, raw, low in (('ALS', als, scaled.als), ('BPR', bpr, scaled.bpr)):
            a = context.rank_scores({u: raw[u] for u in context.active}, rp.PRIMARY_K)
            b = context.rank_scores(low, rp.PRIMARY_K)
            ma, mb = context.metrics(a, rp.PRIMARY_K), context.metrics(b, rp.PRIMARY_K)
            rows.append({'policy': policy, 'model': model, 'raw_dtype': str(next(iter(raw.values())).dtype),
                         'users': len(a), 'identical_lists_share': float(np.mean([list(a[u]) == list(b[u]) for u in a])),
                         'ndcg_difference': mb['ndcg'] - ma['ndcg'], 'recall_difference': mb['recall'] - ma['recall']})
    result = {'mf_dir': Path(mf_dir).as_posix(), 'rows': rows}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mf-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.mf_dir, args.output)
