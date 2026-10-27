"""v8 feasibility: data size and evaluation-population counts of candidate Taobao/Diginetica folds (no model is
trained or scored), as v7_feasibility.md did for F4.

Every candidate uses the v7 split ratios (test 0.25, validation 0.2, event 3-core, no isolation forest) over the whole
adapted period. Taobao user shares 1/2/5/10% are nested subsets of the 10% conversion (same seeded hash, lower
threshold). Usage: python audit/v8prep/feasibility.py <runs dir> <output.json>
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import dataset_adapters as da  # noqa: E402
from data_protocol import UnifiedPreprocessor  # noqa: E402
from policy_evaluation import POLICIES, EvaluationContext  # noqa: E402

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
SPLIT = dict(test_size=.25, validation_size=.2, use_if=False, core_mode='events', min_user_events=3,
             min_item_events=3, random_state=42)


def taobao_subset(percent):
    source, target = runs / 'taobao_p10' / 'raw', runs / f'taobao_p{percent}' / 'raw'
    if percent == 10 or (target / 'events.csv').exists():
        return source if percent == 10 else target
    target.mkdir(parents=True)
    events = pd.read_csv(source / 'events.csv')
    events.loc[da._keep_user(events.visitorid.to_numpy(), percent, 42)].to_csv(target / 'events.csv', index=False)
    for name in ('item_properties_part1.csv', 'item_properties_part2.csv', 'category_tree.csv'):
        shutil.copy2(source / name, target / name)
    return target


def describe(directory):
    train = pd.read_csv(directory / 'events_train_clean.csv', usecols=['visitorid', 'itemid', 'event'])
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    row = {'cutoff_utc': manifest['cutoff_utc'], 'train_events': len(train), 'train_users': int(train.visitorid.nunique()),
           'catalog_items': int(train.itemid.nunique()), 'train_purchases': int((train.event == 'transaction').sum())}
    for policy in POLICIES:
        population = EvaluationContext(directory, policy).population()
        row[f'{policy}_evaluation_users'] = population['evaluation_users']
        row[f'{policy}_warm_users'] = population['warm_users']
    return row


configs = {f'taobao_p{p}': (lambda p=p: taobao_subset(p)) for p in (1, 2, 5, 10)}
configs |= {f'diginetica_{m}': (lambda m=m: runs / f'diginetica_{m}' / 'raw') for m in ('session', 'drop')}
result = {'split': SPLIT, 'candidates': {}}
for name, raw in configs.items():
    start = time.perf_counter()
    prepared = runs / name / 'prepared'
    if not (prepared / 'manifest.json').exists():
        UnifiedPreprocessor(data_dir=raw(), processed_dir=prepared, **SPLIT).run()
    adapted = runs / ('taobao_p10' if name.startswith('taobao') else name) / 'raw' / 'adapter.json'
    result['candidates'][name] = {'adapter': json.loads(adapted.read_text(encoding='utf-8'))['source'],
                                  'test': describe(prepared), 'validation': describe(prepared / 'validation'),
                                  'seconds': round(time.perf_counter() - start, 1)}
    print(name, json.dumps(result['candidates'][name]), flush=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
