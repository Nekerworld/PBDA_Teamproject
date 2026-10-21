"""Independent split/leakage audit against the raw RetailRocket files.

The checks re-derive every boundary from raw CSVs with a separate implementation
instead of trusting manifest.json. Each check returns pass/fail with counts.
Usage: python audit_tools/leakage_check.py --processed <dir> --raw ../../data [--gnn]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POSITIVE = ('addtocart', 'transaction')
KEY = ['timestamp', 'visitorid', 'itemid', 'event']


def raw_events(raw):
    frame = pd.read_csv(Path(raw) / 'events.csv')
    frame['order'] = np.arange(len(frame))
    return frame


def tie_safe_cutoff(frame, test_size):
    ordered = frame.sort_values(['timestamp', 'order'], kind='stable')
    index = min(max(int(len(ordered) * (1 - test_size)), 1), len(ordered) - 1)
    return int(ordered.timestamp.iloc[index])


def core_fixpoint(frame, min_user, min_item):
    current = frame
    while True:
        users = current.visitorid.value_counts()
        items = current.itemid.value_counts()
        keep = current.visitorid.map(users).ge(min_user) & current.itemid.map(items).ge(min_item)
        if keep.all():
            return current
        current = current.loc[keep]


def as_keys(frame):
    return set(map(tuple, frame[KEY].astype({'timestamp': 'int64', 'visitorid': 'int64', 'itemid': 'int64'})
                   .astype(str).to_numpy()))


def latest_available(raw, items, cutoff):
    parts = [pd.read_csv(Path(raw) / f'item_properties_part{i}.csv', dtype={'property': str, 'value': str})
             for i in (1, 2)]
    props = pd.concat(parts, ignore_index=True)
    props = props.loc[(props.property == 'available') & (props.timestamp < cutoff) & props.itemid.isin(items)]
    props = props.sort_values('timestamp', kind='stable').drop_duplicates('itemid', keep='last')
    return props.set_index('itemid').value.astype(float)


def audit_partition(name, directory, train_raw, test_raw, cutoff, raw, config, checks, parent_cutoff=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))

    def record(check, ok, **detail):
        checks.append({'partition': name, 'check': check, 'pass': bool(ok), **detail})

    record('cutoff_matches_independent_recompute', manifest['cutoff_timestamp'] == cutoff,
           manifest=manifest['cutoff_timestamp'], recomputed=cutoff)
    train = pd.read_csv(directory / 'events_train_clean.csv')
    test_original = pd.read_csv(directory / 'events_test_original.csv')
    record('train_strictly_before_cutoff', train.timestamp.max() < cutoff,
           max_train=int(train.timestamp.max()))
    record('test_original_at_or_after_cutoff', test_original.timestamp.min() >= cutoff,
           min_test=int(test_original.timestamp.min()))
    if parent_cutoff is not None:
        record('validation_window_inside_outer_train',
               test_original.timestamp.max() < parent_cutoff and train.timestamp.max() < parent_cutoff,
               max_val_label=int(test_original.timestamp.max()), outer_cutoff=parent_cutoff)
    expected_train = core_fixpoint(train_raw, config['min_user_events'], config['min_item_events'])
    record('train_equals_independent_core_fixpoint', as_keys(expected_train) == as_keys(train),
           expected=len(expected_train), actual=len(train))
    record('core_thresholds_hold', train.visitorid.value_counts().min() >= config['min_user_events']
           and train.itemid.value_counts().min() >= config['min_item_events'])
    raw_users, raw_items = set(train_raw.visitorid), set(train_raw.itemid)
    reference = test_raw.loc[test_raw.visitorid.isin(raw_users) & test_raw.itemid.isin(raw_items)
                             & test_raw.event.isin(POSITIVE)]
    eval_users = sorted(set(reference.visitorid))
    saved_users = sorted(pd.read_csv(directory / 'evaluation_users.csv').visitorid)
    record('evaluation_users_match_definition', eval_users == saved_users,
           expected=len(eval_users), saved=len(saved_users))
    truth = test_raw.loc[test_raw.visitorid.isin(eval_users) & test_raw.event.isin(POSITIVE)]
    saved_truth = test_original.loc[test_original.event.isin(POSITIVE)]
    record('truth_equals_raw_test_positives', as_keys(truth) == as_keys(saved_truth),
           expected=len(truth), saved=len(saved_truth))
    future_keys = as_keys(test_raw)
    record('no_test_event_in_train_file', not (as_keys(train) & future_keys))
    props = pd.read_csv(directory / 'item_properties_train_clean.csv', dtype={'property': str, 'value': str})
    record('properties_strictly_before_cutoff', props.timestamp.max() < cutoff,
           max_property_timestamp=int(props.timestamp.max()))
    features = pd.read_csv(directory / 'feature_train_inliers.csv').set_index('itemid')
    expected_available = latest_available(raw, set(features.index), cutoff).reindex(features.index).fillna(0)
    record('available_is_last_value_before_cutoff',
           np.allclose(expected_available.to_numpy(), features.has_available.to_numpy()),
           items=len(features), available_items=int(features.has_available.eq(1).sum()))
    catalog = set(pd.read_csv(directory / 'candidate_items.csv').itemid)
    record('catalog_equals_train_items', catalog == set(train.itemid), catalog=len(catalog))
    return train, manifest


def audit_artifacts(name, directory, train, checks):
    directory = Path(directory)
    protocol = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['protocol_id']
    for stage in ('als', 'gnn'):
        path = directory / f'{stage}_artifacts.json'
        if path.exists():
            stamp = json.loads(path.read_text(encoding='utf-8'))
            checks.append({'partition': name, 'check': f'{stage}_artifact_protocol_matches',
                           'pass': stamp['protocol_id'] == protocol})
    if (directory / 'gnn_user_embeddings.csv').exists():
        users = pd.read_csv(directory / 'gnn_user_embeddings.csv', usecols=['visitorid']).visitorid
        checks.append({'partition': name, 'check': 'gnn_users_equal_train_users',
                       'pass': set(users) == set(train.visitorid)})


def audit_graph(name, directory, train, future_pairs, checks):
    from model import GNNEmbeddingGenerator
    generator = GNNEmbeddingGenerator(processed_dir=directory, graph_mode='bipartite', device='cpu')
    graph = generator._prepare_graph_structures(generator._load_inputs())
    users, items = np.asarray(graph['user_ids']), np.asarray(graph['item_ids'])
    offset = graph['offsets']['item']
    src, dst = graph['edge_sources'], graph['edge_targets']
    user_side = src < offset
    edge_pairs = set(zip(users[src[user_side]].tolist(), items[dst[user_side] - offset].tolist()))
    train_pairs = set(zip(train.visitorid.tolist(), train.itemid.tolist()))
    leaked = {pair for pair in edge_pairs - train_pairs}
    future_only = (future_pairs - train_pairs) & edge_pairs
    checks.append({'partition': name, 'check': 'gnn_edges_equal_train_pairs',
                   'pass': edge_pairs == train_pairs and not leaked,
                   'edges': len(edge_pairs), 'train_pairs': len(train_pairs), 'non_train_edges': len(leaked)})
    checks.append({'partition': name, 'check': 'no_future_only_pair_in_gnn_graph',
                   'pass': not future_only, 'future_only_pairs_in_graph': len(future_only)})


def main(processed, raw, gnn=False, output=None, end_fraction=None):
    """raw is the fold's own raw folder (fold_raw for rolling folds). Pass end_fraction to check a
    rolling fold against the full raw data instead; preprocessing manifests do not record it."""
    processed = Path(processed)
    outer = json.loads((processed / 'manifest.json').read_text(encoding='utf-8'))
    config = outer['config']
    events = raw_events(raw)
    end = end_fraction if end_fraction is not None else config.get('end_fraction', 1.0)
    if end < 1.0:
        end_cutoff = tie_safe_cutoff(events, 1 - end)
        events = events.loc[events.timestamp < end_cutoff]
    checks = []
    cutoff = tie_safe_cutoff(events, config['test_size'])
    outer_train_raw = events.loc[events.timestamp < cutoff]
    outer_test_raw = events.loc[events.timestamp >= cutoff]
    val_cutoff = tie_safe_cutoff(outer_train_raw, config['validation_size'])
    inner_train_raw = outer_train_raw.loc[outer_train_raw.timestamp < val_cutoff]
    inner_test_raw = outer_train_raw.loc[outer_train_raw.timestamp >= val_cutoff]
    outer_train, _ = audit_partition('outer', processed, outer_train_raw, outer_test_raw, cutoff, raw, config, checks)
    inner_train, _ = audit_partition('inner', processed / 'validation', inner_train_raw, inner_test_raw,
                                     val_cutoff, raw, config, checks, parent_cutoff=cutoff)
    checks.append({'partition': 'inner', 'check': 'validation_labels_never_in_inner_train',
                   'pass': not (as_keys(inner_train) & as_keys(inner_test_raw))})
    if gnn:
        audit_graph('outer', processed, outer_train,
                    set(zip(outer_test_raw.visitorid, outer_test_raw.itemid)), checks)
        audit_graph('inner', processed / 'validation', inner_train,
                    set(zip(inner_test_raw.visitorid, inner_test_raw.itemid)), checks)
    frame = pd.DataFrame(checks)
    print(frame.to_string(index=False))
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({'processed': str(processed), 'checks': checks,
                                            'all_pass': bool(frame['pass'].all())}, indent=2, default=str),
                                encoding='utf-8')
    return bool(frame['pass'].all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--gnn', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--end-fraction', type=float,
                        help='fold end fraction when --raw is the full raw folder rather than the fold_raw copy')
    args = parser.parse_args()
    sys.exit(0 if main(args.processed, args.raw, args.gnn, args.output, args.end_fraction) else 1)
