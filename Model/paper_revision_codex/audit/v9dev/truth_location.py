"""Where the D2 policy truth lies relative to training-only candidate sources (development analysis, no model).

Sources, each read from the D2 training events only:
  history   items the user touched in training
  covisit   neighbours of the user's last 5 items in the item-item co-visit graph (same user, events within 1 hour)
  category  most popular items of the user's 3 most recent categories
  popular   most popular items of the last training week
Coverage = mean over evaluation users of |truth in source| / |truth| (EvaluationContext denominator).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from policy_evaluation import EvaluationContext  # noqa: E402

D2 = HERE / 'audit_runs' / 'v9dev' / 'dev' / 'prepared'
HOUR = 3_600_000


def covisit_graph(train, index, window=HOUR):
    frame = train.assign(idx=index.get_indexer(train.itemid)).sort_values(['visitorid', 'timestamp'], kind='stable')
    graph = defaultdict(Counter)
    for _, g in frame.groupby('visitorid', sort=False):
        ts, idx = g.timestamp.to_numpy(), g.idx.to_numpy()
        for i in range(len(idx)):
            j = i + 1
            while j < len(idx) and ts[j] - ts[i] <= window:
                if idx[j] != idx[i]:
                    graph[idx[i]][idx[j]] += 1
                    graph[idx[j]][idx[i]] += 1
                j += 1
    return graph


def main(sizes=(50, 200, 1000)):
    out = {}
    for policy in ('repeat', 'novel'):
        c = EvaluationContext(D2, policy)
        train = c.train
        if policy == 'repeat':
            graph = covisit_graph(train, c.index)
            category = pd.read_csv(D2 / 'feature_train_inliers.csv').set_index('itemid').categoryid.reindex(c.ids).fillna(-1).to_numpy()
            counts = np.bincount(c.index.get_indexer(train.itemid), minlength=len(c.ids)).astype(float)
            week = train.timestamp >= train.timestamp.max() - 7 * 86_400_000
            recent_pop = np.bincount(c.index.get_indexer(train.itemid[week]), minlength=len(c.ids)).astype(float)
            by_category = defaultdict(list)
            for i in np.argsort(-counts, kind='stable'):
                by_category[category[i]].append(i)
            ordered = train.assign(idx=c.index.get_indexer(train.itemid)).sort_values(['visitorid', 'timestamp'], ascending=[True, False], kind='stable')
            recent = {int(u): list(dict.fromkeys(g.idx.tolist())) for u, g in ordered.groupby('visitorid', sort=False)}
        rows = defaultdict(list)
        for u in c.users:
            truth = c.truth_idx[u]
            denominator = len(c.truth[u])
            seen = set(c.seen_idx[u].tolist()) if policy == 'novel' and u in c.seen_idx else set()
            items = recent.get(u, [])
            rows['reachable'].append(len(truth) / denominator)
            rows['history'].append(len(truth & set(items)) / denominator)
            for n in sizes:
                neighbours = Counter()
                for item in items[:5]:
                    neighbours.update(dict(graph[item].most_common(n)))
                covisit = set([i for i, _ in neighbours.most_common() if i not in seen][:n])
                rows[f'covisit_{n}'].append(len(truth & covisit) / denominator)
                cats = list(dict.fromkeys(category[i] for i in items))[:3]
                cat = {i for c_ in cats for i in by_category[c_][:n // max(len(cats), 1)] if i not in seen}
                rows[f'category_{n}'].append(len(truth & cat) / denominator)
                pop = [i for i in np.argsort(-recent_pop, kind='stable')[:n + len(seen)] if i not in seen][:n]
                rows[f'popular_{n}'].append(len(truth & set(pop)) / denominator)
                union = covisit | cat | (set(items) if policy == 'repeat' else set())
                rows[f'union_{n}'].append(len(truth & union) / denominator)
                rows[f'union_{n}_size'].append(len(union))
        out[policy] = {'users': len(c.users), **{k: float(np.mean(v)) for k, v in rows.items()}}
        print(policy, json.dumps(out[policy], indent=1), flush=True)
    (Path(__file__).parent / 'truth_location.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
