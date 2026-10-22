"""The plan hash changes with any change of the Hybrid grid: candidate depths, strategy pairing or order."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp

SEALED = {v: json.loads((HERE / 'audit' / f'results_{v}' / 'plan.json').read_text(encoding='utf-8'))['plan_sha256']
          for v in ('v6', 'v7')}


def plan_hash():
    return hashlib.sha256(json.dumps(rp.plan_payload(), sort_keys=True).encode()).hexdigest()


def other_pairing(grid):
    """20000 for blend only and 'all' for both strategies: same size, different configurations."""
    out = [c for c in grid if c['candidate_k'] not in (20000, 'all')]
    out += [c for c in grid if c['candidate_k'] == 20000 and c['strategy'] == 'blend']
    out += [c for c in grid if c['candidate_k'] == 'all']
    out += [dict(c, candidate_k='all') for c in grid if c['candidate_k'] == 20000 and c['strategy'] == 'balanced_union']
    return out


@pytest.mark.parametrize('change', [
    lambda g: [dict(c, candidate_k=300 if c['candidate_k'] == 500 else c['candidate_k']) for c in g],
    other_pairing,
    lambda g: sorted(g, key=lambda c: str(c['candidate_k'])),
], ids=['depth values', 'strategy pairing', 'order only'])
@pytest.mark.parametrize('version', ['v6', 'v7'])
def test_any_other_hybrid_grid_changes_the_sealed_plan_hash(monkeypatch, version, change):
    for name in rp.V7:
        monkeypatch.setattr(rp, name, getattr(rp, name))
    rp.use_protocol(version)
    assert plan_hash() == SEALED[version]
    changed = change(rp.HYBRID_GRID)
    assert len(changed) == len(rp.HYBRID_GRID) and changed != rp.HYBRID_GRID
    monkeypatch.setattr(rp, 'HYBRID_GRID', changed)
    assert plan_hash() != SEALED[version]
    assert rp.plan_payload()['hybrid_grid_sha256'] == rp.hybrid_grid_sha256(changed)
