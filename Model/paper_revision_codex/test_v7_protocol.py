"""Protocol v7: widened grids on an unopened window, without changing the sealed v6 plan."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp

V6_PLAN = HERE / 'audit' / 'results_v6' / 'plan.json'


def plan_hash():
    return hashlib.sha256(json.dumps(rp.plan_payload(), sort_keys=True).encode()).hexdigest()


@pytest.fixture
def v7(monkeypatch):
    for name, value in rp.V7.items():
        monkeypatch.setattr(rp, name, value)


def test_default_plan_is_the_sealed_v6_plan():
    stored = json.loads(V6_PLAN.read_text(encoding='utf-8'))
    assert plan_hash() == stored['plan_sha256']


def test_v7_plan_differs_and_disables_posthoc(v7):
    stored = json.loads(V6_PLAN.read_text(encoding='utf-8'))
    plan = rp.plan_payload()
    assert plan_hash() != stored['plan_sha256']
    assert plan['protocol'] == 'rolling-origin-v7' and plan['posthoc_test_analyses'] == 'none'
    assert rp.POSTHOC is False and [f['name'] for f in plan['folds']] == ['F4_unseen']


def test_v7_grids_extend_past_every_edge_v6_selected_and_keep_v6_selections(v7):
    stored = json.loads(V6_PLAN.read_text(encoding='utf-8'))['plan']
    grids = {'ALS': (stored['als_grid'], rp.ALS_GRID), 'BPR': (stored['bpr_grid'], rp.BPR_GRID),
             'GNN': (stored['gnn_grid'], rp.GNN_GRID)}
    hit = 0
    for fold in ('F1_hist', 'F2_mid', 'F3_early'):
        selection = json.loads((HERE / 'audit' / 'results_v6' / fold / 'selection.json').read_text(encoding='utf-8'))
        chosen = {'ALS': selection['mf']['als'], 'BPR': selection['mf']['bpr'], 'GNN': selection['gnn']['config']}
        for family, config in chosen.items():
            old_grid, new_grid = grids[family]
            for key, side in rp.grid_edges(config, old_grid).items():
                new = {c[key] for c in new_grid}
                old = {c[key] for c in old_grid}
                assert (min(new) < min(old)) if side == 'min' else (max(new) > max(old)), (fold, family, key)
                hit += 1
        assert selection['mf']['bpr_epochs'] == max(stored['bpr_epoch_checkpoints'])
    assert hit >= 9
    assert max(rp.BPR_EPOCH_CHECKPOINTS) > max(stored['bpr_epoch_checkpoints'])
    assert len({c['factors'] for c in rp.BPR_GRID}) > 1
    for fold in ('F1_hist', 'F2_mid', 'F3_early'):
        selection = json.loads((HERE / 'audit' / 'results_v6' / fold / 'selection.json').read_text(encoding='utf-8'))
        assert selection['mf']['als'] in rp.ALS_GRID
        assert selection['mf']['bpr'] in rp.BPR_GRID
        assert selection['gnn']['config'] in rp.GNN_GRID
        assert selection['hybrid_backbone']['config'] in rp.GNN_GRID


def test_v7_window_ends_before_any_v6_evaluation_window(v7):
    """Every v6 fold evaluates (validation or test) only after end * (1 - test) * (1 - validation)."""
    stored = json.loads(V6_PLAN.read_text(encoding='utf-8'))['plan']
    earliest = min(f['end_fraction'] * (1 - f['test_size']) * (1 - f['validation_size']) for f in stored['folds'])
    assert all(f['end_fraction'] <= earliest + 1e-12 for f in rp.FOLDS)


RAW_EVENTS = HERE.parent.parent / 'data' / 'events.csv'


@pytest.mark.skipif(not RAW_EVENTS.exists(), reason='raw RetailRocket events.csv not available')
def test_v7_window_ends_at_or_before_v6_inner_validation_timestamps(v7):
    """Fractions are taken on event counts after cleaning, so check the actual timestamps too."""
    import numpy as np
    import pandas as pd
    events = pd.read_csv(RAW_EVENTS, usecols=['timestamp'])
    ts = events.timestamp.to_numpy()
    order = np.lexsort((np.arange(len(ts)), ts))
    for fold in rp.FOLDS:
        end = int(ts[order[int(len(ts) * fold['end_fraction'])]])
        for v6_fold in ('F1_hist', 'F2_mid', 'F3_early'):
            checks = json.loads((HERE / 'audit' / 'results_v6' / v6_fold / 'leakage_check.json')
                                .read_text(encoding='utf-8'))['checks']
            inner = next(c['manifest'] for c in checks
                         if c['partition'] == 'inner' and c['check'] == 'cutoff_matches_independent_recompute')
            assert end <= inner, (fold['name'], v6_fold)


def test_grid_edges():
    grid = [dict(a=a, b=b) for a in (1, 2, 3) for b in (.1, .2)]
    assert rp.grid_edges(dict(a=2, b=.1), grid) == {'b': 'min'}
    assert rp.grid_edges(dict(a=3, b=.2), grid) == {'a': 'max', 'b': 'max'}
    assert rp.grid_edges(dict(a=2, c=5), grid + [dict(a=2, c=5)]) == {}


def test_use_protocol_v6_is_a_no_op():
    before = plan_hash()
    rp.use_protocol('v6')
    assert plan_hash() == before
