"""D2 (development) ceilings of re-ranking the a01 candidate pool, by module, and where history truth sits.

ceiling = NDCG@50 when every truth item of the pool is ranked first (EvaluationContext denominator), averaged over all
evaluation users (users outside the pool count zero).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from policy_evaluation import EvaluationContext  # noqa: E402

ROOT = HERE / 'audit_runs' / 'v9dev'


def main(policy='repeat', attempt='a01'):
    c = EvaluationContext(ROOT / 'mf_d2', policy)
    frame = pd.read_pickle(ROOT / 'attempts' / attempt / f'd2_{policy}.pkl')
    frame['y'] = [i in c.truth_idx[u] for u, i in zip(frame.uid, frame.idx)]
    discounts = 1 / np.log2(np.arange(2, 52))
    out = {}
    for name, mask in {'all': np.ones(len(frame), bool), 'mf': frame.in_mf, 'graph': frame.in_graph,
                       'context': frame.in_context, 'mf+graph': frame.in_mf | frame.in_graph,
                       'history': frame.h_seen == 1}.items():
        hits = frame.loc[mask].groupby('uid').y.sum()
        values = [discounts[:min(int(hits.get(u, 0)), 50)].sum() / discounts[:min(50, len(c.truth[u]))].sum()
                  for u in c.users]
        out[f'ceiling_{name}'] = float(np.mean(values))
        out[f'recall_{name}'] = float(np.mean([hits.get(u, 0) / len(c.truth[u]) for u in c.users]))
    seen = frame.loc[frame.h_seen == 1]
    out['history_items_per_user'] = float(seen.groupby('uid').size().mean())
    truth = seen.loc[seen.y]
    out['history_truth_recent_pos'] = np.expm1(truth.h_recent_pos).round().value_counts().sort_index().head(10).to_dict()
    out['history_truth_views'] = truth.h_views.clip(upper=10).value_counts().sort_index().to_dict()
    out['history_truth_carted_before'] = float((truth.h_carts > 0).mean())
    out['history_rows_carted_before'] = float((seen.h_carts > 0).mean())
    out['history_truth_same_category_as_last'] = float(truth.uc_last.mean())
    out['history_rows_same_category_as_last'] = float(seen.uc_last.mean())
    out = {k: (v if not isinstance(v, dict) else {str(a): int(b) for a, b in v.items()}) for k, v in out.items()}
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / f'pool_ceiling_{attempt}_{policy}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
