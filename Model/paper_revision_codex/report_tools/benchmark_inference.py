"""Batch inference benchmark from saved models, excluding model loading.

Measures actual full-catalog MF retrieval and candidate-only graph scoring for
Hybrid. These are single-process batch measurements, not online latency.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recommendation_scoring import FactorScorer,GNNScorer,minmax,top_indices,hybrid_scores
from improve_hybrid import ScoreCache,candidate_indices
from data_protocol import require_artifacts
from ranking_evaluation import ranking_metrics


def main(root):
    root=Path(root);selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    config=selection['hybrid']['config'];rows=[];differences=[]
    for seed in [42,43,44]:
        directory=root/'runs'/f'seed_{seed}'
        factor=FactorScorer(directory,selection['mf_alpha'])
        plain=GNNScorer(directory/'gnn',factor.item_ids)
        graph=GNNScorer(directory/'hybrid',factor.item_ids)
        users=pd.read_csv(directory/'evaluation_users.csv').visitorid.astype(int).tolist()
        users=[u for u in users if u in factor.users]
        available=pd.read_csv(directory/'feature_train_inliers.csv').set_index('itemid').reindex(factor.item_ids).has_available.eq(1).to_numpy()
        als_meta=require_artifacts(directory,'als')
        train=als_meta['training_seconds']
        gmeta=require_artifacts(directory/'gnn','gnn');hmeta=require_artifacts(directory/'hybrid','gnn')
        training={'ALS':train['ALS'],'BPR':train['BPR'],'ALS+BPR':train['ALS+BPR'],
            'GNN':gmeta['training_seconds']+(train['ALS+BPR'] if gmeta['negative_sampling'].startswith('mf_mixed') else 0),
            'Hybrid':train['ALS+BPR']+hmeta['training_seconds']}
        for name in ['ALS','BPR','ALS+BPR','GNN','Hybrid']:
            durations=[];last={}
            for repeat in range(3):
                start=time.perf_counter()
                for uid in users:
                    row=factor.users[uid]
                    if name=='ALS':scores=minmax(factor.als_i@factor.als_u[row]).astype(np.float32);ids=factor.item_ids
                    elif name=='BPR':scores=minmax(factor.bpr_i@factor.bpr_u[row]).astype(np.float32);ids=factor.item_ids
                    elif name=='ALS+BPR':
                        a=factor.als_i@factor.als_u[row];b=factor.bpr_i@factor.bpr_u[row]
                        scores=selection['mf_alpha']*minmax(a).astype(np.float32)+(1-selection['mf_alpha'])*minmax(b).astype(np.float32)
                        ids=factor.item_ids
                    elif name=='GNN':scores=plain.scores(uid);ids=factor.item_ids
                    else:
                        a=factor.als_i@factor.als_u[row];b=factor.bpr_i@factor.bpr_u[row]
                        cache=ScoreCache.__new__(ScoreCache);cache.ids=factor.item_ids
                        cache.als={uid:minmax(a).astype(np.float32)};cache.bpr={uid:minmax(b).astype(np.float32)}
                        candidate=candidate_indices(cache,uid,config['alpha'],config['candidate_k'],config['strategy'])
                        mf=cache.mf(uid,config['alpha'])
                        scores=hybrid_scores(mf[candidate],graph.scores(uid,candidate),available[candidate],config['omega'],config['gamma'],config['fusion'])
                        ids=factor.item_ids[candidate]
                    last[uid]=ids[top_indices(scores,ids,50)].tolist()
                durations.append(time.perf_counter()-start)
            expected=pd.read_csv(directory/f'recommendations_{name.replace("+","_")}.csv').groupby('visitorid').itemid.agg(list).to_dict()
            identical=sum(last[u]==expected[u] for u in users)
            truth_frame=pd.read_csv(directory/'events_test_original.csv')
            truth={int(u):set(group.itemid) for u,group in truth_frame.loc[truth_frame.event.isin(['addtocart','transaction'])].groupby('visitorid')}
            frame=pd.DataFrame([(u,item,rank) for u,items in last.items() for rank,item in enumerate(items,1)],columns=['visitorid','itemid','rank'])
            measured,_=ranking_metrics(frame,truth,50)
            actual=pd.read_csv(root/'experiment_metrics.csv')
            saved=actual.loc[(actual.seed==seed)&(actual.model==name)&(actual.k==50)].iloc[0]
            for u in users:
                if last[u]!=expected[u]:
                    difference={'seed':seed,'model':name,'visitorid':u,
                        'same_set':set(last[u])==set(expected[u]),'saved':expected[u],'measured':last[u]}
                    if name=='Hybrid':
                        a,b,_=factor.scores(u)
                        check=ScoreCache.__new__(ScoreCache);check.ids=factor.item_ids
                        check.als={u:minmax(a).astype(np.float32)};check.bpr={u:minmax(b).astype(np.float32)}
                        candidate=candidate_indices(check,u,config['alpha'],config['candidate_k'],config['strategy'])
                        full=graph.scores(u)[candidate];local=graph.scores(u,candidate)
                        scores=hybrid_scores(check.mf(u,config['alpha'])[candidate],full,available[candidate],config['omega'],config['gamma'],config['fusion'])
                        reproduced=factor.item_ids[candidate][top_indices(scores,factor.item_ids[candidate],50)].tolist()
                        difference['max_abs_gnn_rounding_error']=float(np.abs(full-local).max())
                        difference['matches_saved_using_full_gnn']=reproduced==expected[u]
                        if reproduced!=expected[u]:raise ValueError('Benchmark mismatch is not explained by candidate GEMV rounding')
                    else:raise ValueError('Unexpected non-Hybrid benchmark mismatch')
                    differences.append(difference)
            rows.append({'seed':seed,'model':name,'active_users':len(users),
                'training_seconds':training[name],'inference_batch_seconds':float(np.median(durations)),
                'amortized_user_ms':float(np.median(durations))*1000/len(users),
                'repetitions':3,'exact_matching_lists':identical,'total_lists':len(users),
                'ndcg50_measured':measured['ndcg'],'recall50_measured':measured['recall'],
                'ndcg_delta_vs_saved':float(measured['ndcg']-saved.ndcg),'recall_delta_vs_saved':float(measured['recall']-saved.recall),
                'note':'Model loading excluded; single-process CPU batch; not online latency. MF mining dependency included in GNN training when used.'})
            print(seed,name,rows[-1]['inference_batch_seconds'],'matching',identical,'/',len(users),flush=True)
    pd.DataFrame(rows).to_csv(root/'runtime_benchmark.csv',index=False)
    (root/'runtime_rounding_differences.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args();main(a.experiment)
