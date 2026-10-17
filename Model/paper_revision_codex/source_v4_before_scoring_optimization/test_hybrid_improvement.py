"""Checks that the improvement cannot win through labels, cohorts or lost BPR."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from recommendation_scoring import hybrid_scores,top_indices
from ranking_evaluation import ranking_metrics
from improve_hybrid import ScoreCache,joint_score,select_row,candidate_indices
from learning import MFMixedNegativeSampler


def toy_cache():
    cache=ScoreCache.__new__(ScoreCache)
    cache.users=[1,2,3];cache.active=[1,2]
    cache.ids=np.array([10,20,30,40])
    cache.truth={1:{20,40},2:{10},3:{99}}
    cache.positive_indices={1:{1,3},2:{0}}
    cache.als={1:np.array([1.,.2,.1,0]),2:np.array([.1,.8,.3,.2])}
    cache.bpr={1:np.array([0.,1.,.1,.4]),2:np.array([1.,.1,.3,.2])}
    return cache


def test_joint_selection_cannot_trade_away_recall():
    reference={'ndcg':.1,'recall':.2}
    rows=[dict(ndcg=.13,recall=.18),dict(ndcg=.11,recall=.22)]
    assert select_row(rows,reference)['recall']==.22
    assert joint_score(rows[0],reference)<0


def test_saved_settings_are_reselected_against_final_reference():
    rows=[dict(ndcg=.120,recall=.190),dict(ndcg=.115,recall=.205)]
    first=select_row(rows,{'ndcg':.11,'recall':.18})
    final=select_row(rows,{'ndcg':.12,'recall':.20})
    assert first['ndcg']==.120
    assert final['ndcg']==.115


def test_fast_metrics_match_authoritative_metrics_and_count_missing():
    cache=toy_cache();ranking={1:np.array([1,3]),2:np.array([2,0])}
    records=[(u,int(cache.ids[i]),r) for u,items in ranking.items() for r,i in enumerate(items,1)]
    frame=pd.DataFrame(records,columns=['visitorid','itemid','rank'])
    expected,details=ranking_metrics(frame,cache.truth,2)
    actual=cache.metrics(ranking,2)
    assert actual['ndcg']==pytest.approx(expected['ndcg'])
    assert actual['recall']==pytest.approx(expected['recall'])
    assert cache.details(ranking,2).ndcg.mean()==pytest.approx(expected['ndcg'])
    assert len(details)==3


def test_mf_endpoint_preserves_ensemble_not_als():
    cache=toy_cache();uid=1
    mf=cache.mf(uid,.5)
    final=hybrid_scores(mf,np.arange(4.),np.ones(4,dtype=bool),1.,1.)
    assert np.array_equal(top_indices(final,cache.ids,4),top_indices(mf,cache.ids,4))
    assert not np.array_equal(top_indices(final,cache.ids,4),top_indices(cache.als[uid],cache.ids,4))


def test_candidates_do_not_depend_on_test_truth_and_have_no_duplicates():
    cache=toy_cache()
    a=candidate_indices(cache,1,.5,3,'balanced_union')
    cache.truth={1:{99999}}
    b=candidate_indices(cache,1,.5,3,'balanced_union')
    assert np.array_equal(a,b)
    assert len(a)==len(set(a))==3


def test_balanced_union_keeps_each_retriever_quota_after_filling():
    cache=toy_cache()
    cache.bpr[1]=np.array([0.,0.,0.,1.])
    union=candidate_indices(cache,1,1.,2,'balanced_union')
    blend=candidate_indices(cache,1,1.,2,'blend')
    assert set(union)=={0,3}
    assert set(blend)=={0,1}
    # With overlap, keep the distinct BPR contribution rather than collapsing
    # back to the blended retrieval list after filling remaining slots.
    cache.ids=np.arange(6)
    cache.als[1]=np.array([6,5,4,3,2,1.])
    cache.bpr[1]=np.array([6,1,2,3,4,5.])
    union=candidate_indices(cache,1,1.,4,'balanced_union')
    assert 5 in union and len(union)==4


def test_mixed_negatives_align_ids_exclude_history_and_skip_saturated(tmp_path):
    # Graph user/item orders deliberately differ from saved MF order.
    pd.DataFrame({'visitorid':[200,100]}).to_csv(tmp_path/'als_user_mapping.csv',index=False)
    pd.DataFrame({'itemid':[30,10,20]}).to_csv(tmp_path/'als_item_mapping.csv',index=False)
    for model in ['als','bpr']:
        np.save(tmp_path/f'{model}_user_factors.npy',np.array([[.1,.2],[.3,.4]]))
        np.save(tmp_path/f'{model}_item_factors.npy',np.array([[.2,.1],[.4,.2],[.1,.3]]))
    pairs=np.array([[0,0],[1,0],[1,1],[1,2]])
    sampler=MFMixedNegativeSampler(pairs,3,tmp_path,[100,200],[10,20,30])
    draws=sampler.sample(np.array([0]*100+[1]*100),np.random.default_rng(4))
    assert set(draws[:100]) <= {1,2}
    assert np.all(draws[100:]==-1)
