import pytest
from robust_hybrid import robust_select,configuration_grid


def test_selection_uses_three_seed_mean_not_favorable_single_seed():
    rows=[]
    for seed,ndcg,recall in [(42,.15,.24),(43,.10,.17),(44,.10,.17)]:
        rows.append(dict(name='unstable',seed=seed,alpha=.5,strategy='blend',fusion='rrf',candidate_k=5000,omega=.3,gamma=1.,ndcg=ndcg,recall=recall))
    for seed in [42,43,44]:
        rows.append(dict(name='stable',seed=seed,alpha=.5,strategy='blend',fusion='rrf',candidate_k=5000,omega=.3,gamma=1.,ndcg=.13,recall=.21))
    selected,_=robust_select(rows,dict(ndcg=.12,recall=.20))
    assert selected['name']=='stable'
    assert selected['joint_lift']==pytest.approx(.05)


def test_selection_rejects_missing_validation_seed():
    row=dict(name='x',seed=42,alpha=.5,strategy='blend',fusion='rrf',candidate_k=5000,omega=.3,gamma=1.,ndcg=.13,recall=.21)
    with pytest.raises(ValueError,match='coverage'):robust_select([row],dict(ndcg=.12,recall=.20))


def test_grid_contains_gnn_dominant_settings_and_neutral_inventory():
    grid=configuration_grid()
    assert len(grid)==576
    assert any(c['omega']==.1 and c['gamma']==1. and c['candidate_k']==20000 for c in grid)
    assert all(0<c['omega']<1 for c in grid)
