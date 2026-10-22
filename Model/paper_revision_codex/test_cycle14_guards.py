"""The plan hash covers the Hybrid grid values, not only the grid size."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp


def plan_hash():
    return hashlib.sha256(json.dumps(rp.plan_payload(), sort_keys=True).encode()).hexdigest()


def test_same_size_hybrid_grid_with_other_values_changes_the_plan(monkeypatch):
    sealed = json.loads((HERE / 'audit' / 'results_v6' / 'plan.json').read_text(encoding='utf-8'))['plan_sha256']
    assert plan_hash() == sealed
    changed = [dict(c, gamma=3. if c['gamma'] == 1.5 else c['gamma']) for c in rp.HYBRID_GRID]
    monkeypatch.setattr(rp, 'HYBRID_GRID', changed)
    assert len(changed) == rp.plan_payload()['hybrid_grid_size'] and plan_hash() != sealed
    assert rp.plan_payload()['hybrid_grid_axes']['gamma'] == [1., 1.1, 1.25, 3.]


def test_hybrid_grid_is_the_one_its_plan_axes_describe():
    for version in ('v6', 'v7'):
        plan = json.loads((HERE / 'audit' / f'results_{version}' / 'plan.json').read_text(encoding='utf-8'))['plan']
        axes = plan['hybrid_grid_axes']
        rebuilt = [dict(alpha=a, strategy=s, candidate_k=c, fusion=f, omega=w, gamma=g)
                   for a in axes['alpha'] for c in plan['candidate_ks']
                   for s in axes['strategy'] if not (c == 'all' and s != 'blend')
                   for f in axes['fusion'] for w in axes['omega'] for g in axes['gamma']]
        assert rebuilt == rp.HYBRID_GRID, version
