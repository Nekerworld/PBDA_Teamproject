"""Measured fixed-setting component controls and label-based diagnostic ceilings."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from improve_hybrid import ScoreCache,config_ranking,candidate_indices


def main(root):
    root=Path(root);selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    selected=selection['hybrid'];config=selected['config']
    validation=Path(selected['trial']['directory'])
    if selected['checkpoint']=='hybrid':validation=validation/'hybrid_checkpoint'
    sources=[('validation',42,root/'validation_mf',validation)]+[
        ('test',seed,root/'runs'/f'seed_{seed}',root/'runs'/f'seed_{seed}'/'hybrid') for seed in [42,43,44]]
    rows=[];oracles=[];complementarity=[]
    for role,seed,mf,gnn in sources:
        cache=ScoreCache(mf,gnn)
        rows.append({'role':role,'seed':seed,'model':'GNN_full_same_weights','k':50,
            **cache.metrics(cache.ranking(cache.gnn))})
        prepared={u:candidate_indices(cache,u,config['alpha'],config['candidate_k'],config['strategy']) for u in cache.active}
        configurations=[('Hybrid',dict(config)),('Hybrid_no_inventory',dict(config,gamma=1.)),
            ('MF_only_same_inventory',dict(config,omega=1.)),('GNN_candidates_same_inventory',dict(config,omega=0.)),
            ('MF_only_no_inventory',dict(config,omega=1.,gamma=1.)),('GNN_candidates_no_inventory',dict(config,omega=0.,gamma=1.))]
        for name,cfg in configurations:
            rankings=config_ranking(cache,cfg,prepared)
            total=sum(len(items) for items in rankings.values())
            available=sum(int(cache.available[items].sum()) for items in rankings.values())
            rows.append({'role':role,'seed':seed,'model':name,'k':50,**cache.metrics(rankings),
                'top50_available_share':available/total if total else 0.,
                'recommended_items':total,'available_recommended_items':available})
        for uid in cache.users:
            relevant=cache.truth[uid]
            candidate=prepared.get(uid,[])
            found=len(set(cache.ids[candidate])&relevant)
            idcg=sum(1/np.log2(i+2) for i in range(min(50,len(relevant))))
            dcg=sum(1/np.log2(i+2) for i in range(min(50,found)))
            oracles.append({'role':role,'seed':seed,'visitorid':uid,'candidate_k':config['candidate_k'],
                'candidate_recall':found/len(relevant),'oracle_ndcg50':dcg/idcg,
                'note':'Diagnostic label-based upper bound; never an inference model or selection input from test.'})
        if role=='test':
            predictions={}
            for label in ['ALS_BPR','GNN','Hybrid']:
                frame=pd.read_csv(mf/f'recommendations_{label}.csv')
                predictions[label]=frame.groupby('visitorid').itemid.agg(lambda x:set(x)).to_dict()
            for uid in cache.users:
                relevant=cache.truth[uid]
                m=predictions['ALS_BPR'].get(uid,set())&relevant
                g=predictions['GNN'].get(uid,set())&relevant
                h=predictions['Hybrid'].get(uid,set())&relevant
                complementarity.append({'seed':seed,'visitorid':uid,'mf_only_user':int(bool(m) and not g),
                    'gnn_only_user':int(bool(g) and not m),'both_user':int(bool(m) and bool(g)),
                    'neither_user':int(not m and not g),'mf_only_hits':len(m-g),'gnn_only_hits':len(g-m),
                    'hybrid_recovers_mf_only':len(h&(m-g)),'hybrid_recovers_gnn_only':len(h&(g-m)),
                    'hybrid_new_hits_beyond_both_top50':len(h-(m|g))})
        del cache
    pd.DataFrame(rows).to_csv(root/'component_ablation.csv',index=False)
    actual=pd.read_csv(root/'experiment_metrics.csv')
    for row in rows:
        if row['role']=='test' and row['model']=='Hybrid':
            expected=actual.loc[(actual.seed==row['seed'])&(actual.model=='Hybrid')&(actual.k==50)].iloc[0]
            assert np.allclose([row['ndcg'],row['recall']],[expected.ndcg,expected.recall],rtol=1e-10,atol=1e-12)
    pd.DataFrame(oracles).to_csv(root/'candidate_oracle_users.csv',index=False)
    pd.DataFrame(oracles).groupby(['role','seed'])[['candidate_recall','oracle_ndcg50']].mean().to_csv(root/'candidate_oracle.csv')
    pd.DataFrame(complementarity).to_csv(root/'complementarity_users.csv',index=False)
    pd.DataFrame(complementarity).drop(columns='visitorid').groupby('seed').sum().to_csv(root/'complementarity_summary.csv')
    print('Fixed-setting component controls and diagnostic candidate ceilings saved.',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args();main(a.experiment)
