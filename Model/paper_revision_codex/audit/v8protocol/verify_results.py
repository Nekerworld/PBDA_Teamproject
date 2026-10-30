"""Checks on the packaged v8 record: sealed model files, Holm within each family, stage order, history-feature use.

Usage: python audit/v8protocol/verify_results.py audit/results_v8"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
fold = root / 'TB1_full'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((fold / 'selection.json').read_text(encoding='utf-8'))

mismatch, files = 0, 0
for policy, record in selection['rerankers'].items():
    for name, model in record['models'].items():
        files += 1
        mismatch += sha(fold / 'rerankers' / f'{policy}_{name}.pkl') != model['sha256']
print(f'sealed model files {files}, sha256 mismatches {mismatch}')

for policy, record in selection['rerankers'].items():
    for name, entry in record['models'].items():
        model = pickle.loads((fold / 'rerankers' / f'{policy}_{name}.pkl').read_bytes())
        used = {entry['features'][i] for trees in model._predictors for tree in trees
                for i, leaf in zip(tree.nodes['feature_idx'], tree.nodes['is_leaf']) if not leaf}
        print(f'{policy} {name}: features {len(entry["features"])}, used in splits {len(used)}, '
              f'history used {sorted(f for f in used if f.startswith("h_"))}')


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

stages = [json.loads(line) for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
access = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
when = {s['stage']: s['utc'] for s in stages if s['stage'] in ('rerankers_trained', 'selection_sealed', 'test_evaluated', 'completed')}
print('stages', when)
print('test openings', len(access), [a['utc'] for a in access],
      'selection sha matches access log', all(a['selection_sha256'] == sha(fold / 'selection.json') for a in access))
order = when['rerankers_trained'] <= when['selection_sealed'] < access[0]['utc'] < when['test_evaluated'] <= when['completed']
print('order rerankers_trained <= selection_sealed < test opened < test_evaluated <= completed:', order)
