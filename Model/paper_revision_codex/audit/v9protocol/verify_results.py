"""Checks on the packaged v9 record: sealed ladder model files and the features they split on, Holm within each
family, the ladder table against the test summary, stage order, and that the users of the confirmation sample are
disjoint from the v8 test sample and the v9 development sample.

Usage: python audit/v9protocol/verify_results.py audit/results_v9 [confirmation raw folder, other raw folders ...]"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import hybrid_v9 as v9  # noqa: E402

root, others = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
fold = root / 'TB2_full'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((fold / 'selection.json').read_text(encoding='utf-8'))
plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']

mismatch, files = 0, 0
for policy, record in selection['ladder'].items():
    for name, entry in record['models'].items():
        files += 1
        path = fold / 'ladder' / entry['model']
        mismatch += sha(path) != entry['sha256']
        booster = pickle.loads(path.read_bytes()).booster_
        assert booster.num_feature() == len(entry['features'])
        split = dict(zip(entry['features'], booster.feature_importance('split')))  # columns in the fitted order
        used = sorted(f for f, n in split.items() if n > 0)
        groups = {g: sorted(set(used) & set(cols)) for g, cols in v9.FEATURE_GROUPS.items()}
        print(f'{policy} {name}: modules {entry["modules"]}, features {len(entry["features"])}, used in splits '
              f'{len(used)}, by module { {g: len(c) for g, c in groups.items()} }, params {entry["params"]}')
        outside = [f for f in used if not any(f in v9.FEATURE_GROUPS[m] for m in entry['modules'])]
        assert not outside, f'{name} splits on features of modules it does not have: {outside}'
print(f'sealed model files {files}, sha256 mismatches {mismatch}')


def holm(p):
    adjusted, running = np.empty(len(p)), 0.
    for rank, j in enumerate(np.argsort(p, kind='stable')):
        running = max(running, min(1., (len(p) - rank) * p[j]))
        adjusted[j] = running
    return adjusted


tests = pd.read_csv(fold / 'report' / 'paired_tests.csv')
for (policy, family), group in tests.groupby(['policy', 'family']):
    diff = np.abs(holm(group.p_value.to_numpy()) - group.p_holm.to_numpy()).max()
    print(f'{policy} {family}: comparisons {len(group)}, significant {int(group.significant_holm.sum())}, '
          f'max |Holm recomputed - reported| {diff:.2e}')

summary = pd.read_csv(fold / 'report' / 'test_summary.csv')
ladder = pd.read_csv(fold / 'report' / 'ladder.csv').set_index('policy')
for policy, row in ladder.iterrows():
    ndcg = summary.loc[(summary.policy == policy) & (summary.k == 50)].set_index('model').ndcg_mean.to_dict()
    again = v9.ladder_summary(ndcg, plan['baselines'])
    same = all(np.isclose(again[k], row[k]) for k in again if k != 'pass') and again['pass'] == bool(row['pass'])
    print(f'{policy} ladder recomputed from test_summary matches ladder.csv: {same}; '
          f'L2/L1 {again["L2/L1"]:.4f}, L3/L2 {again["L3/L2"]:.4f}, pass {again["pass"]}')

stages = [json.loads(line) for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
access = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
when = {}
for s in stages:  # first time each stage was reached; a resumed run logs the later stages again
    if s['stage'] in ('ladder_trained', 'selection_sealed', 'test_evaluated', 'completed'):
        when.setdefault(s['stage'], s['utc'])
print('stages (first)', when, 'log entries', len(stages))
print('test openings', len(access), [a['utc'] for a in access],
      'selection sha matches access log', all(a['selection_sha256'] == sha(fold / 'selection.json') for a in access))
order = when['ladder_trained'] <= when['selection_sealed'] < access[0]['utc'] < when['test_evaluated'] <= when['completed']
print('order ladder_trained <= selection_sealed < test opened < test_evaluated <= completed:', order)

if others:
    mine, *rest = others
    users = set(pd.read_csv(mine / 'events.csv', usecols=['visitorid']).visitorid)
    for other in rest:
        theirs = set(pd.read_csv(other / 'events.csv', usecols=['visitorid']).visitorid)
        print(f'users shared by {mine} and {other}: {len(users & theirs)} (of {len(users)} and {len(theirs)})')
