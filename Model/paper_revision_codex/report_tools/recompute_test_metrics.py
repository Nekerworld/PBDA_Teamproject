"""Recompute sealed-test metrics from saved recommendation lists and the raw events only.

Truth, the raw training history (for the novel policy) and the evaluation population are
re-derived from the original events.csv with the tie-safe cutoffs of audit_tools.leakage_check,
not read from the prepared split. Every (policy, seed, model, K) row of test_metrics.csv is
compared, and every novel list is checked against the raw training history and for duplicates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from audit_tools.leakage_check import tie_safe_cutoff  # noqa: E402

POSITIVE = ('addtocart', 'transaction')


def raw_partition(raw, end_fraction, test_size):
    events = pd.read_csv(Path(raw) / 'events.csv')
    events['order'] = np.arange(len(events))
    if end_fraction < 1:
        events = events.loc[events.timestamp < tie_safe_cutoff(events, 1 - end_fraction)]
    cutoff = tie_safe_cutoff(events, test_size)
    return events.loc[events.timestamp < cutoff], events.loc[events.timestamp >= cutoff], cutoff


def metrics(ranked, truth, k):
    hits = np.array([item in truth for item in ranked[:k]], dtype=float)
    idcg = (1 / np.log2(np.arange(2, min(k, len(truth)) + 2))).sum()
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    return dcg / idcg, hits.sum() / len(truth), hits.sum() / k


def main(root, fold, raw, output):
    plan = json.loads((Path(root) / 'plan.json').read_text(encoding='utf-8'))['plan']
    spec = next(f for f in plan['folds'] if f['name'] == fold)
    train, test, cutoff = raw_partition(raw, spec['end_fraction'], spec['test_size'])
    fold_dir = Path(root) / fold
    refits = json.loads((fold_dir / 'refits.json').read_text(encoding='utf-8'))
    stored = pd.read_csv(fold_dir / 'test_metrics.csv')
    raw_users, raw_items = set(train.visitorid), set(train.itemid)
    positives = test.loc[test.event.isin(POSITIVE)]
    eligible = positives.loc[positives.visitorid.isin(raw_users) & positives.itemid.isin(raw_items)]
    users = sorted(set(eligible.visitorid))
    original = {u: set(g.itemid) for u, g in positives.loc[positives.visitorid.isin(users)].groupby('visitorid')}
    seen = {u: set(g.itemid) for u, g in train.loc[train.visitorid.isin(users)].groupby('visitorid')}
    truths = {'repeat': original,
              'novel': {u: t - seen.get(u, set()) for u, t in original.items() if t - seen.get(u, set())}}
    rows, violations = [], {'novel_seen_items': 0, 'duplicates': 0, 'listed_items': 0}
    for seed, dirs in refits.items():
        seed_dir = Path(root) / dirs['mf']
        for policy, truth in truths.items():
            for model in sorted(stored.model.unique()):
                path = seed_dir / f'recommendations_{policy}_{model.replace("+", "_")}.csv'
                frame = pd.read_csv(path).sort_values(['visitorid', 'rank'])
                lists = {u: g.itemid.tolist() for u, g in frame.groupby('visitorid')}
                if set(lists) - set(truth):
                    raise ValueError(f'{path.name}: list for a user outside the {policy} population')
                for u, items in lists.items():
                    violations['listed_items'] += len(items)
                    violations['duplicates'] += len(items) - len(set(items))
                    if policy == 'novel':
                        violations['novel_seen_items'] += len(set(items) & seen.get(u, set()))
                for k in sorted(stored.k.unique()):
                    values = np.array([metrics(lists.get(u, []), truth[u], k) for u in sorted(truth)])
                    reference = stored.loc[(stored.policy == policy) & (stored.seed == int(seed)) &
                                           (stored.model == model) & (stored.k == k)].iloc[0]
                    rows.append({'policy': policy, 'seed': int(seed), 'model': model, 'k': int(k),
                                 'users_raw': len(truth), 'users_stored': int(reference.users),
                                 'ndcg_diff': abs(values[:, 0].mean() - reference.ndcg),
                                 'recall_diff': abs(values[:, 1].mean() - reference.recall),
                                 'precision_diff': abs(values[:, 2].mean() - reference.precision)})
    frame = pd.DataFrame(rows)
    result = {'fold': fold, 'raw_test_cutoff': int(cutoff), 'compared_rows': len(frame),
              'population_matches': bool((frame.users_raw == frame.users_stored).all()),
              'max_abs_diff': {m: float(frame[f'{m}_diff'].max()) for m in ('ndcg', 'recall', 'precision')},
              **violations}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fold', required=True)
    parser.add_argument('--raw', type=Path, default=HERE.parent.parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.fold, args.raw, args.output)
