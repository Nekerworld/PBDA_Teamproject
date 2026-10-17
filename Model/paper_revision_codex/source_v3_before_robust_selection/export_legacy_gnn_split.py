"""Re-create the submitted paper's LightGCN train/validation/test split (no training).

Uses the submitted code (past_code.py) with its fixed seed:
- GNNPreprocessor: random 80/20 item split, Isolation Forest on train items,
  test = test-item events + each user's last positive train event.
- GNNEmbeddingGenerator: random 80/20 split of unique train (user, item) pairs
  into train/validation. Validation pairs also stay in the propagation graph.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from past_code import GNNEmbeddingGenerator, GNNPreprocessor

ROOT = Path(__file__).resolve().parents[2]


def describe(name, frame):
    row = {'split': name, 'rows': len(frame), 'users': frame.visitorid.nunique(),
           'items': frame.itemid.nunique()}
    if 'event' in frame:
        row.update({kind: int((frame.event == kind).sum()) for kind in ('view', 'addtocart', 'transaction')})
    if 'timestamp' in frame:
        row['min_time'] = pd.to_datetime(frame.timestamp.min(), unit='ms')
        row['max_time'] = pd.to_datetime(frame.timestamp.max(), unit='ms')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'legacy_paper_lightgcn')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data_dir, out = args.data_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # GNNPreprocessor saves figures to the relative path data/visualization.
    os.chdir(out)

    GNNPreprocessor(data_dir=data_dir, processed_dir=out, random_state=args.seed).run()

    generator = GNNEmbeddingGenerator(processed_dir=out, data_dir=data_dir, seed=args.seed, device='cpu')
    graph = generator._prepare_graph_structures(generator._load_inputs())
    pairs = graph['interactions']
    train_idx, val_idx = train_test_split(np.arange(len(pairs)), test_size=0.2, random_state=args.seed)
    users, items = np.asarray(graph['user_ids']), np.asarray(graph['item_ids'])
    split_pairs = {
        'gnn_train_pairs': pd.DataFrame({'visitorid': users[pairs[train_idx, 0]], 'itemid': items[pairs[train_idx, 1]]}),
        'gnn_validation_pairs': pd.DataFrame({'visitorid': users[pairs[val_idx, 0]], 'itemid': items[pairs[val_idx, 1]]}),
    }
    for name, frame in split_pairs.items():
        frame.to_csv(out / f'{name}.csv', index=False)

    summary = [describe('events_train_clean', pd.read_csv(out / 'events_train_clean.csv')),
               *(describe(name, frame) for name, frame in split_pairs.items()),
               describe('events_test', pd.read_csv(out / 'events_test.csv')),
               describe('events_train_outliers', pd.read_csv(out / 'events_train_outliers.csv'))]
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'split_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
