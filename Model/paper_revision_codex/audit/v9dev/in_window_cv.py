"""Analysis only: how well a ranker trained inside D2 (user-grouped 5-fold CV on D2 labels) ranks D2.

Compared with the D1-trained rankers of the ladder, this separates "the features cannot rank D2 better" from "a ranker
trained on D1 transfers badly to D2". Nothing here selects a model or a setting for the ladder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr  # noqa: E402
import hybrid_v9 as v9  # noqa: E402
from policy_evaluation import EvaluationContext  # noqa: E402

ROOT = HERE / 'audit_runs' / 'v9dev'
PARAMS = dict(learning_rate=.1, max_iter=100, max_leaf_nodes=31)


def main(policy='repeat', attempt='a02', folds=5):
    c = EvaluationContext(ROOT / 'mf_d2', policy)
    frame = pd.read_pickle(ROOT / 'attempts' / attempt / f'd2_{policy}.pkl')
    y = ltr.labels(c, frame)
    users = np.array(sorted(frame.uid.unique()))
    assignment = dict(zip(users, np.random.default_rng(42).permutation(len(users)) % folds))
    fold_of = frame.uid.map(assignment).to_numpy()
    sampled = v9.sample_negatives(frame, y, 100)
    out = {}
    for name, modules in {'mf': ('mf',), 'mf+graph': ('mf', 'graph'), 'all': v9.MODULES,
                          'context': ('context',)}.items():
        rows, columns = v9.module_rows(frame, modules), v9.module_features(modules)
        rankings = {}
        for f in range(folds):
            train = (fold_of != f) & sampled & rows
            model = v9.fit(frame.loc[train].reset_index(drop=True), y[train], columns, PARAMS)
            rankings.update(v9.rank(model, frame.loc[(fold_of == f) & rows], columns))
        out[name] = c.metrics(rankings)['ndcg']
        print(policy, name, round(out[name], 5), flush=True)
    single = {}
    for column in ('h_weight', 'h_days', 'h_recent_pos', 'als', 'gnn', 'cv', 'cg', 'uc_item_share', 's_events_last'):
        sign = 1 if column not in ('h_days', 'h_recent_pos') else -1
        score = sign * frame[column].to_numpy()
        ranked = {}
        for uid, part in frame.assign(s=score).groupby('uid', sort=False):
            idx, s = part.idx.to_numpy(), part.s.to_numpy()
            ranked[int(uid)] = idx[np.lexsort((idx, -s))][:50]
        single[column] = c.metrics(ranked)['ndcg']
    out['single_feature_rankers'] = single
    print(json.dumps(single, indent=1))
    (Path(__file__).parent / f'in_window_cv_{attempt}_{policy}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
