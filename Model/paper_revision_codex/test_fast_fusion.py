import numpy as np
import pytest
from fast_fusion import fast_top_indices,cached_grid
from recommendation_scoring import top_indices
from test_hybrid_improvement import toy_cache
from improve_hybrid import config_ranking


@pytest.mark.parametrize('size,k',[(0,3),(4,2),(50,50),(20000,50)])
def test_partial_topk_matches_full_sort_with_boundary_ties(size,k):
    rng=np.random.default_rng(5);scores=rng.integers(0,20,size=size).astype(float)
    ids=rng.permutation(np.arange(size)*13+99)
    assert np.array_equal(fast_top_indices(scores,ids,k),top_indices(scores,ids,k))


def test_cached_grid_matches_original_with_mf_and_gnn_ties():
    cache=toy_cache()
    cache.gnn={u:np.array([.5,.5,.1,.1]) for u in cache.active}
    cache.available=np.array([True,False,True,False])
    grid=[dict(alpha=.5,strategy=s,candidate_k=4,fusion=f,omega=w,gamma=g)
        for s in ['blend','balanced_union'] for f in ['minmax','rrf'] for w in [.1,.5,.9] for g in [1.,1.5]]
    actual=cached_grid(cache,grid,42,'toy')
    for cfg,row in zip(grid,actual):
        expected=cache.metrics(config_ranking(cache,cfg))
        assert row['ndcg']==pytest.approx(expected['ndcg'],abs=1e-14)
        assert row['recall']==pytest.approx(expected['recall'],abs=1e-14)
