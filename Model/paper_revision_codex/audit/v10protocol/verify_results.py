"""Checks on the packaged v10 record, per fold: sealed re-ranker and ladder model files (hash, feature count, ladder
splits only on its own modules), Holm within each family, the ladder table against the test summary (reported names
mapped back to the hybrid_v9 names), stage order, and one test opening per fold with the sealed selection.

Usage: python audit/v10protocol/verify_results.py audit/results_v10"""
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

root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']
names = plan['ladder']['reported_names']
original = {reported: name for name, reported in names.items()}
stages = [json.loads(line) for line in (root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
access = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
print('policies', plan['policies'], 'acknowledged records', sorted(plan['acknowledged_test_records']))


def holm(p):
    adjusted, running = np.empty(len(p)), 0.
    for rank, j in enumerate(np.argsort(p, kind='stable')):
        running = max(running, min(1., (len(p) - rank) * p[j]))
        adjusted[j] = running
    return adjusted


ok = True
for fold_spec in plan['folds']:
    name = fold_spec['name']
    fold = root / name
    selection = json.loads((fold / 'selection.json').read_text(encoding='utf-8'))
    mismatch, files = 0, 0
    for folder, key in (('rerankers', 'rerankers'), ('ladder', 'ladder')):
        for policy, record in selection[key].items():
            for model, entry in record['models'].items():
                files += 1
                path = fold / folder / entry['model']
                mismatch += sha(path) != entry['sha256']
                fitted = pickle.loads(path.read_bytes())
                if key == 'ladder':
                    booster = fitted.booster_
                    assert booster.num_feature() == len(entry['features'])
                    split = dict(zip(entry['features'], booster.feature_importance('split')))
                    used = sorted(f for f, n in split.items() if n > 0)
                    outside = [f for f in used if not any(f in v9.FEATURE_GROUPS[m] for m in entry['modules'])]
                    assert not outside, f'{name} {model} splits on features of modules it does not have: {outside}'
                    print(f'{name} {policy} ladder {names.get(model, model)}: modules {entry["modules"]}, used in '
                          f'splits {len(used)} of {len(entry["features"])}, params {entry["params"]}')
                else:
                    print(f'{name} {policy} reranker {model}: features {len(entry["features"])}, params {entry["params"]}')
    print(f'{name}: sealed model files {files}, sha256 mismatches {mismatch}')
    ok &= mismatch == 0

    tests = pd.read_csv(fold / 'report' / 'paired_tests.csv')
    for (policy, family), group in tests.groupby(['policy', 'family']):
        diff = np.abs(holm(group.p_value.to_numpy()) - group.p_holm.to_numpy()).max()
        print(f'{name} {policy} {family}: comparisons {len(group)}, significant {int(group.significant_holm.sum())}, '
              f'max |Holm recomputed - reported| {diff:.2e}')
        ok &= diff < 1e-12

    summary = pd.read_csv(fold / 'report' / 'test_summary.csv')
    ladder = pd.read_csv(fold / 'report' / 'ladder.csv').set_index('policy')
    for policy, row in ladder.iterrows():
        ndcg = summary.loc[(summary.policy == policy) & (summary.k == 50)].set_index('model').ndcg_mean.to_dict()
        ndcg = {original.get(m, m): v for m, v in ndcg.items() if m not in names}
        again = v9.ladder_summary(ndcg, plan['baselines'])
        same = all(np.isclose(again[k], row[k]) for k in again if k != 'pass') and again['pass'] == bool(row['pass'])
        print(f'{name} {policy} ladder recomputed from test_summary matches ladder.csv: {same}; '
              f'L1/L0 {again["L1/L0"]:.4f}, L2/L1 {again["L2/L1"]:.4f}, L3/L2 {again["L3/L2"]:.4f}, pass {again["pass"]}')
        ok &= same

    when = {}
    for s in stages:  # first time each stage was reached for this fold; a resumed run logs later stages again
        if s.get('fold') == name and s['stage'] in ('ladder_trained', 'rerankers_trained', 'selection_sealed',
                                                     'test_evaluated'):
            when.setdefault(s['stage'], s['utc'])
    opened = [a for a in access if a['fold'] == name]
    same_selection = all(a['selection_sha256'] == sha(fold / 'selection.json') for a in opened)
    order = (max(when['ladder_trained'], when['rerankers_trained']) <= when['selection_sealed'] < opened[0]['utc']
             < when['test_evaluated'])
    print(f'{name}: stages (first) {when}; openings {len(opened)} {[a["utc"] for a in opened]}; selection sha matches '
          f'{same_selection}; order trained <= sealed < opened < evaluated: {order}')
    ok &= len(opened) == 1 and same_selection and order
print('completed logged:', any(s['stage'] == 'completed' for s in stages))
print('ALL CHECKS PASS' if ok else 'SOME CHECK FAILED')
