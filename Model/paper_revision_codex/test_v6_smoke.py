"""End-to-end smoke run of rolling_protocol on synthetic data with tiny grids."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rolling_protocol as rp


def make_raw(folder, users=40, items=30, events=6000, seed=0):
    """Random sessions with drifting item preferences, so later windows contain novel items."""
    folder.mkdir()
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(1_000, 1_000_000, events))
    user = rng.integers(1, users + 1, events)
    drift = (np.arange(events) * items // events)
    item = (drift + rng.integers(0, 8, events)) % items + 1
    kind = rng.choice(['view', 'addtocart', 'transaction'], events, p=[.6, .3, .1])
    pd.DataFrame({'timestamp': t, 'visitorid': user, 'event': kind, 'itemid': item,
                  'transactionid': np.nan}).to_csv(folder / 'events.csv', index=False)
    props = pd.DataFrame([(1, i, p, v) for i in range(1, items + 1)
                          for p, v in [('categoryid', '1'), ('available', str(i % 2)), ('color', 'blue')]],
                         columns=['timestamp', 'itemid', 'property', 'value'])
    props.iloc[:items].to_csv(folder / 'item_properties_part1.csv', index=False)
    props.iloc[items:].to_csv(folder / 'item_properties_part2.csv', index=False)
    pd.DataFrame({'categoryid': [1], 'parentid': [np.nan]}).to_csv(folder / 'category_tree.csv', index=False)


def tiny_settings(monkeypatch):
    monkeypatch.setattr(rp, 'SEEDS', [42, 43])
    monkeypatch.setattr(rp, 'ALS_GRID', [dict(factors=4, regularization=.02, alpha=2., iterations=3)])
    monkeypatch.setattr(rp, 'BPR_GRID', [dict(factors=4, learning_rate=1e-2, regularization=.03)])
    monkeypatch.setattr(rp, 'BPR_EPOCH_CHECKPOINTS', (1, 2))
    monkeypatch.setattr(rp, 'GNN_GRID', [dict(layers=1, embedding_dim=8, learning_rate=1e-2),
                                         dict(layers=2, embedding_dim=8, learning_rate=1e-2)])
    monkeypatch.setattr(rp, 'GNN_FIXED', {**rp.GNN_FIXED, 'batch_size': 64, 'n_negatives': 2,
                                          'validation_interval': 2, 'min_epochs': 1, 'early_stopping_patience': 4})
    monkeypatch.setattr(rp, 'GNN_MAX_EPOCHS', 4)
    monkeypatch.setattr(rp, 'GNN_CAP_EXTENSION_EPOCHS', 6)
    monkeypatch.setattr(rp, 'MF_ALPHAS', [0., .5, 1.])
    monkeypatch.setattr(rp, 'HYBRID_BACKBONES', 2)
    monkeypatch.setattr(rp, 'HYBRID_GRID', [dict(alpha=.5, strategy=s, candidate_k=c, fusion='minmax', omega=w, gamma=1.)
                                            for c, s in ((3, 'blend'), (3, 'balanced_union'), ('all', 'blend'))
                                            for w in (.3, .7)])
    monkeypatch.setattr(rp, 'POSTHOC_CANDIDATE_KS', [3, 'all'])
    monkeypatch.setattr(rp, 'git_state', lambda: {'commit': 'test', 'branch': 'test', 'dirty_files': []})


def test_rolling_protocol_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw)
    tiny_settings(monkeypatch)
    monkeypatch.setattr(rp, 'FOLDS', rp.FOLDS[:2])
    root = tmp_path / 'run'
    rp.run(root, raw)
    for fold in rp.FOLDS:
        fold_dir = root / fold['name']
        selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
        assert selection['outer_test_used_for_selection'] is False
        assert selection['policies']['novel']['hybrid']['backbone'] == selection['hybrid_backbone']['name']
        metrics = pd.read_csv(fold_dir / 'test_metrics.csv')
        assert set(metrics.model) >= {'Hybrid', 'GNN', 'ALS', 'BPR', 'ALS+BPR', 'Popularity', 'Recent', 'RecentPop'}
        assert set(metrics.policy) == {'repeat', 'novel'} and set(metrics.seed) == {42, 43}
        assert json.loads((fold_dir / 'leakage_check.json').read_text(encoding='utf-8'))['all_pass']
    assert (root / rp.FOLDS[0]['name'] / 'report' / 'posthoc_candidate_depth.csv').exists()
    assert (root / rp.FOLDS[0]['name'] / 'report' / 'if_ablation.csv').exists()
    log = [json.loads(l) for l in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [e['purpose'] for e in log].count('sealed_test') == len(rp.FOLDS)
    assert (root / 'report_all_folds_summary.csv').exists()
    rp.run(root, raw)  # resuming is idempotent: no new sealed-test access
    again = (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(again) == len(log)
    monkeypatch.setattr(rp, 'MF_ALPHAS', [0., 1.])
    with pytest.raises(RuntimeError):
        rp.run(root, raw)


def test_v7_fold_runs_without_posthoc_and_records_grid_edges(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    make_raw(raw, events=12000)
    for name in ('PROTOCOL', 'FOLDS', 'POSTHOC', 'PLAN_EXTRA'):
        monkeypatch.setattr(rp, name, rp.V7[name])
    tiny_settings(monkeypatch)
    root = tmp_path / 'run'
    rp.run(root, raw)
    fold_dir = root / 'F4_unseen'
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']
    assert plan['protocol'] == 'rolling-origin-v7' and 'grid_edge_rule' in plan
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    assert set(selection['grid_edges']) == {'ALS', 'BPR', 'GNN', 'Hybrid_backbone'}
    assert selection['grid_edges']['GNN'] == {'layers': selection['gnn']['config']['layers'] == 1 and 'min' or 'max'}
    assert not (fold_dir / 'report' / 'posthoc_candidate_depth.csv').exists()
    assert not (fold_dir / 'report' / 'if_ablation.csv').exists()
    log = [json.loads(l) for l in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [e['purpose'] for e in log] == ['sealed_test']
