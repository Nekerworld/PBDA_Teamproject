"""Real-data 80:20 Hybrid experiment. Selection reads inner validation only.

The outer test is evaluated only after selection.json has been sealed. Earlier
project test results were already inspected; this is a corrected re-evaluation,
not a claim that the historical test was never observed by the researchers.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from data_protocol import require_protocol,require_artifacts,positive_sets,dump_json,sha256
from experiments import clone_protocol,write_dataset_statistics_tables,paired_bootstrap
from model import ALSRecommender,GNNEmbeddingGenerator
from recommendation_scoring import FactorScorer,GNNScorer,minmax,top_indices,hybrid_scores

KS=(10,20,30,40,50)
MF_KWARGS=dict(factors=128,regularization=.02,iterations=256,alpha=2.,top_k=50,
    bpr_epochs=128,device='cuda',als_threads=8,time_decay=True,half_life_days=7.,
    negative_mode='popularity',ensemble_alpha=.5)


def link_mf(source,dest):
    meta=require_artifacts(source,'als')
    for name in [*meta['files'],'als_artifacts.json']:
        target=Path(dest)/name
        if not target.exists(): os.link(Path(source)/name,target)


def progress(root,stage,**extra):
    payload={'stage':stage,'utc':pd.Timestamp.now(tz='UTC').isoformat(),'process_id':os.getpid(),**extra}
    dump_json(root/'progress.json',payload)
    print(json.dumps(payload,ensure_ascii=False),flush=True)


class ScoreCache:
    def __init__(self,directory,gnn_directory=None):
        self.directory=Path(directory)
        protocol=require_protocol(self.directory)
        if protocol['role'] not in ('validation','test'):
            raise ValueError('Unknown evaluation role')
        self.truth=positive_sets(pd.read_csv(self.directory/'events_test_original.csv'))
        self.users=sorted(self.truth)
        self.factor=FactorScorer(directory,.5)
        self.ids=self.factor.item_ids
        self.active=[uid for uid in self.users if uid in self.factor.users]
        self.als={};self.bpr={};self.gnn={}
        for uid in self.active:
            a,b,_=self.factor.scores(uid)
            self.als[uid]=minmax(a).astype(np.float32)
            self.bpr[uid]=minmax(b).astype(np.float32)
        features=pd.read_csv(self.directory/'feature_train_inliers.csv').set_index('itemid')
        self.available=features.reindex(self.ids).has_available.eq(1).to_numpy()
        self.positive_indices={uid:set(np.flatnonzero(np.isin(self.ids,list(self.truth[uid])))) for uid in self.active}
        if gnn_directory is not None:self.load_gnn(gnn_directory)

    def load_gnn(self,directory):
        require_artifacts(directory,'gnn')
        scorer=GNNScorer(directory,self.ids)
        self.gnn={uid:scorer.scores(uid).astype(np.float32) for uid in self.active}

    def mf(self,uid,alpha):
        return alpha*self.als[uid]+(1-alpha)*self.bpr[uid]

    def ranking(self,scores,k=50):
        return {uid:top_indices(values,self.ids,k) for uid,values in scores.items()}

    def details(self,rankings,k=50):
        rows=[]
        discounts=1/np.log2(np.arange(k)+2)
        for uid in self.users:
            ranked=np.asarray(rankings.get(uid,[]),dtype=np.int64)[:k]
            relevant=self.truth[uid]
            hit=np.isin(self.ids[ranked],list(relevant))
            hits=int(hit.sum())
            idcg=float(discounts[:min(k,len(relevant))].sum())
            rows.append({'visitorid':uid,'ndcg':float((hit*discounts[:len(hit)]).sum()/idcg),
                'recall':hits/len(relevant),'precision':hits/k,'hit_rate':float(hits>0),
                'recommendations':len(ranked)})
        return pd.DataFrame(rows)

    def metrics(self,rankings,k=50):
        # Hot search path: all missing cohort members remain zeros.
        ndcg=recall=0.
        discounts=1/np.log2(np.arange(k)+2)
        for uid,ranked in rankings.items():
            relevant=self.positive_indices[uid]
            hit=np.fromiter((int(i) in relevant for i in ranked[:k]),dtype=bool)
            ndcg+=float((hit*discounts[:len(hit)]).sum()/discounts[:min(k,len(self.truth[uid]))].sum())
            recall+=int(hit.sum())/len(self.truth[uid])
        return {'ndcg':ndcg/len(self.users),'recall':recall/len(self.users)}


def joint_score(metrics,reference):
    return min(metrics[name]/max(reference[name],1e-12)-1 for name in ('ndcg','recall'))


def select_row(rows,reference):
    for row in rows:
        row['joint_lift']=joint_score(row,reference)
        row['average_lift']=np.mean([row[n]/max(reference[n],1e-12)-1 for n in ('ndcg','recall')])
    return max(rows,key=lambda r:(r['joint_lift'],r['average_lift'],-r.get('candidate_k',0),
        -abs(r.get('gamma',1)-1),-r.get('omega',0)))


def baseline_search(cache):
    rows=[]
    for alpha in np.round(np.arange(0,1.01,.1),1):
        ranking=cache.ranking({u:cache.mf(u,alpha) for u in cache.active})
        rows.append({'alpha':float(alpha),**cache.metrics(ranking)})
    reference={n:max(r[n] for r in rows) for n in ('ndcg','recall')}
    best=select_row(rows,reference)
    return best,rows


def candidate_indices(cache,uid,alpha,count,strategy):
    if strategy=='blend': return top_indices(cache.mf(uid,alpha),cache.ids,count)
    half=(count+1)//2
    a=top_indices(cache.als[uid],cache.ids,half)
    b=top_indices(cache.bpr[uid],cache.ids,count//2)
    union=np.union1d(a,b)
    if len(union)<count:
        fill=top_indices(cache.mf(uid,alpha),cache.ids,count)
        # Fill only missing slots: adding the whole blended Top-C and then
        # taking its Top-C again silently collapses this strategy into blend.
        missing=np.setdiff1d(fill,union,assume_unique=True)
        missing=missing[top_indices(cache.mf(uid,alpha)[missing],cache.ids[missing],min(count-len(union),len(missing)))]
        union=np.r_[union,missing]
    order=top_indices(cache.mf(uid,alpha)[union],cache.ids[union],min(count,len(union)))
    return union[order]


def config_ranking(cache,config,prepared=None,legacy_als=False):
    rankings={}
    for uid in cache.active:
        idx=prepared[uid] if prepared is not None else candidate_indices(cache,uid,config['alpha'],config['candidate_k'],config['strategy'])
        mf=cache.als[uid] if legacy_als else cache.mf(uid,config['alpha'])
        scores=hybrid_scores(mf[idx],cache.gnn[uid][idx],cache.available[idx],
            config['omega'],config['gamma'],config['fusion'])
        rankings[uid]=idx[top_indices(scores,cache.ids[idx],50)]
    return rankings


def fusion_search(cache,alpha,reference,coarse=False):
    rows=[]
    alphas=sorted(set([round(float(alpha),2),.25,.5,.75]))
    depths=[500,1000,2000] if coarse else [500,1000,2000,5000]
    weights=[.2,.5,.8] if coarse else [.1,.3,.5,.7,.9]
    for strategy in ['blend','balanced_union']:
        for mix in alphas:
            for count in depths:
                prepared={u:candidate_indices(cache,u,mix,count,strategy) for u in cache.active}
                for fusion in ['minmax','rrf']:
                    for omega in weights:
                        cfg=dict(alpha=mix,candidate_k=count,strategy=strategy,fusion=fusion,omega=omega,gamma=1.)
                        rows.append({**cfg,**cache.metrics(config_ranking(cache,cfg,prepared))})
    first=select_row(rows,reference)
    finalists=sorted(rows,key=lambda r:r['joint_lift'],reverse=True)[:8]
    for item in finalists:
        prepared={u:candidate_indices(cache,u,item['alpha'],item['candidate_k'],item['strategy']) for u in cache.active}
        for omega in sorted(set([item['omega'],max(.1,round(item['omega']-.1,2)),min(.9,round(item['omega']+.1,2))])):
            for gamma in [1.,1.1,1.25,1.5,2.]:
                cfg={k:item[k] for k in ('alpha','candidate_k','strategy','fusion')}
                cfg.update(omega=omega,gamma=gamma)
                rows.append({**cfg,**cache.metrics(config_ranking(cache,cfg,prepared))})
    best=select_row(rows,reference)
    return best,rows


class CheckpointObserver:
    def __init__(self,cache,alpha,directory):
        self.cache=cache;self.alpha=alpha;self.directory=Path(directory);self.best=-np.inf
        self.prepared={u:candidate_indices(cache,u,alpha,2000,'blend') for u in cache.active}
        mf=cache.metrics(cache.ranking({u:cache.mf(u,alpha) for u in cache.active}))
        self.mf_reference=mf

    def __call__(self,model,graph,epoch):
        with torch.no_grad():vectors=model(0.).detach().cpu().numpy()
        item=vectors[graph['offsets']['item']:graph['offsets']['item']+graph['num_items']]
        self.cache.gnn={u:item@vectors[graph['user_map'][u]] for u in self.cache.active}
        base=self.cache.metrics(self.cache.ranking(self.cache.gnn))
        reference={n:max(base[n],self.mf_reference[n]) for n in ('ndcg','recall')}
        rows=[]
        for omega in [.2,.5,.8]:
            cfg=dict(alpha=self.alpha,candidate_k=2000,strategy='blend',fusion='minmax',omega=omega,gamma=1.)
            rows.append({**cfg,**self.cache.metrics(config_ranking(self.cache,cfg,self.prepared))})
        best=select_row(rows,reference)
        if best['joint_lift']>self.best:
            self.best=best['joint_lift']
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k!='adjacency'},self.directory/'hybrid_best_state.pt')
            dump_json(self.directory/'hybrid_checkpoint.json',{'epoch':epoch,**best,'reference':reference})
        print('VALIDATION',epoch,'GNN',base,'Hybrid',best['ndcg'],best['recall'],'joint',best['joint_lift'],flush=True)
        return {'validation_original_recall':base['recall'],'hybrid_joint_lift':best['joint_lift'],
                'hybrid_ndcg':best['ndcg'],'hybrid_recall':best['recall']}


def gnn_kwargs(seed,variant,epochs=200):
    return dict(seed=seed,device='cuda',embedding_dim=64,layers=variant['layers'],epochs=epochs,
        batch_size=8192,learning_rate=.001,reg=.0001,early_stopping_patience=30,min_epochs=5,
        graph_mode='bipartite',loss_mode='infonce',n_negatives=32,temperature=.2,
        edge_dropout=.1,propagation='unweighted',time_decay=True,half_life_days=7.,
        negative_mode=variant['negative_mode'])


def train_trial(root,source,mf_dir,cache,alpha,name,variant,seed=42):
    directory=root/'trials'/name
    if not directory.exists():clone_protocol(source,directory)
    link_mf(mf_dir,directory)
    if (directory/'trial_complete.json').exists():return json.loads((directory/'trial_complete.json').read_text())
    start=time.perf_counter()
    generator=GNNEmbeddingGenerator(processed_dir=directory,**gnn_kwargs(seed,variant))
    graph=generator._prepare_graph_structures(generator._load_inputs())
    observer=CheckpointObserver(cache,alpha,directory)
    generator._checkpoint_observer=observer
    model,chosen,history=generator._fit_graph(graph,generator.epochs,torch,cache.truth)
    generator._publish(model,graph,history,chosen,start,{'mode':'inner_validation_only'})
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items() if k!='adjacency'},directory/'baseline_best_state.pt')
    alternative=directory/'hybrid_checkpoint'
    if not alternative.exists():clone_protocol(source,alternative)
    link_mf(mf_dir,alternative)
    model.load_state_dict(torch.load(directory/'hybrid_best_state.pt',map_location='cpu',weights_only=True),strict=False)
    selected=json.loads((directory/'hybrid_checkpoint.json').read_text())
    generator.processed_dir=alternative
    generator._publish(model,graph,history,selected['epoch'],start,{'mode':'inner_hybrid_joint_selection'})
    result={'variant':variant,'baseline_epochs':chosen,'hybrid_epochs':selected['epoch'],
        'seconds':time.perf_counter()-start,'directory':str(directory)}
    dump_json(directory/'trial_complete.json',result)
    del model,graph;torch.cuda.empty_cache()
    return result


def final_rankings(cache,alpha,config):
    models={'ALS':cache.ranking(cache.als),'BPR':cache.ranking(cache.bpr),
            'ALS+BPR':cache.ranking({u:cache.mf(u,alpha) for u in cache.active}),
            'GNN':cache.ranking(cache.gnn),'Hybrid':config_ranking(cache,config)}
    return models


def link_tree(source,dest):
    source,dest=Path(source),Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    for child in source.iterdir():
        target=dest/child.name
        if child.is_dir():link_tree(child,target)
        elif not target.exists():os.link(child,target)


def run(root,source,max_hours=24,reuse_root=None,reuse_validation_metrics=False):
    root=Path(root);source=Path(source)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    started=time.time()
    frozen={'test_size':.2,'use_if':False,'core':3,'seeds':[42,43,44],
        'objective':'min relative lift of original NDCG@50 and Recall@50 against strongest baselines',
        'source':str(source),'source_manifest_hash':sha256(source/'manifest.json'),
        'max_hours':max_hours,'historical_test_seen':True,
        'reuse_root':str(reuse_root) if reuse_root is not None else None,
        'reuse_validation_metrics':bool(reuse_validation_metrics),
        'search_extension':'global joint re-selection and boundary candidates 10000/20000',
        'code':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    plan=root/'experiment_plan.json'
    if plan.exists():
        previous=json.loads(plan.read_text())
        if previous!=frozen:raise ValueError('Plan/source changed; choose a new result directory')
    else:dump_json(plan,frozen)
    if reuse_validation_metrics:
        if reuse_root is None:raise ValueError('Metric reuse requires an explicit parent experiment')
        parent=json.loads((Path(reuse_root)/'experiment_plan.json').read_text(encoding='utf-8'))
        # Only the orchestration/search driver changes. The numerical training,
        # data protocol and scorer implementations must remain identical.
        for name in ['model.py','learning.py','data_protocol.py','recommendation_scoring.py','ranking_evaluation.py','als_debug.py']:
            if parent['code'][name]!=frozen['code'][name]:raise ValueError('Numerical source changed: '+name)
        if parent['source_manifest_hash']!=frozen['source_manifest_hash']:raise ValueError('Data source changed')
    proto=require_protocol(source)
    if proto['use_if'] or abs(proto['config']['test_size']-.2)>1e-9:
        raise ValueError('Requires no-IF common 80:20 prepared source')
    datasets=root/'datasets'/'no_if_events'
    if not datasets.exists():clone_protocol(source,datasets)
    write_dataset_statistics_tables(root/'datasets',root,'no_if_events')
    val_source=datasets/'validation'
    mf_dir=root/'validation_mf'
    if not mf_dir.exists():clone_protocol(val_source,mf_dir)
    if reuse_root is not None and not (mf_dir/'als_artifacts.json').exists():
        link_mf(Path(reuse_root)/'validation_mf',mf_dir)
    progress(root,'validation_mf')
    if not (mf_dir/'als_artifacts.json').exists():ALSRecommender(mf_dir,random_state=42,**MF_KWARGS).run()
    cache=ScoreCache(mf_dir)
    alpha,alpha_rows=baseline_search(cache)
    pd.DataFrame(alpha_rows).to_csv(root/'validation_mf_alpha.csv',index=False)
    variants=[{'layers':2,'negative_mode':'uniform'},{'layers':3,'negative_mode':'uniform'},
              {'layers':2,'negative_mode':'mf_mixed'},{'layers':3,'negative_mode':'mf_mixed'}]
    selections=[];baseline_candidates=[]
    for index,variant in enumerate(variants):
        name=f'l{variant["layers"]}_{variant["negative_mode"]}'
        if index and time.time()-started>max(0,max_hours-8)*3600:break
        if reuse_root is not None and not (root/'trials'/name).exists() and (Path(reuse_root)/'trials'/name/'trial_complete.json').exists():
            link_tree(Path(reuse_root)/'trials'/name,root/'trials'/name)
        progress(root,'validation_gnn',variant=name,index=index+1)
        trial=train_trial(root,val_source,mf_dir,cache,alpha['alpha'],name,variant)
        for checkpoint in ['baseline','hybrid']:
            directory=Path(trial['directory']) if checkpoint=='baseline' else Path(trial['directory'])/'hybrid_checkpoint'
            cache.load_gnn(directory)
            gm=cache.metrics(cache.ranking(cache.gnn))
            baseline_candidates.append({'name':name,'checkpoint':checkpoint,'metrics':gm,'trial':trial})
            reference={n:max(max(a[n] for a in alpha_rows),gm[n],max(c['metrics'][n] for c in baseline_candidates)) for n in ('ndcg','recall')}
            progress(root,'fusion_search',variant=name,checkpoint=checkpoint)
            parent_csv=Path(reuse_root)/f'validation_fusion_{name}_{checkpoint}.csv' if reuse_root is not None else None
            if reuse_validation_metrics and parent_csv.is_file():
                rows=pd.read_csv(parent_csv).to_dict('records')
                best=select_row(rows,reference)
            else:
                best,rows=fusion_search(cache,alpha['alpha'],reference)
            pd.DataFrame(rows).to_csv(root/f'validation_fusion_{name}_{checkpoint}.csv',index=False)
            selections.append({'name':name,'checkpoint':checkpoint,'trial':trial,'config':best})
        dump_json(root/'validation_candidates.json',{'hybrid':selections,'gnn':baseline_candidates})
    reference={n:max(max(a[n] for a in alpha_rows),max(c['metrics'][n] for c in baseline_candidates)) for n in ('ndcg','recall')}
    # Compare all saved settings under one final reference rather than keeping
    # a local winner chosen before later baselines were available.
    for entry in selections:
        path=root/f'validation_fusion_{entry["name"]}_{entry["checkpoint"]}.csv'
        rows=pd.read_csv(path).to_dict('records')
        entry['config']=select_row(rows,reference)
        pd.DataFrame(rows).to_csv(path,index=False)
    boundary=sorted(selections,key=lambda x:joint_score(x['config'],reference),reverse=True)[:3]
    for entry in boundary:
        directory=Path(entry['trial']['directory'])
        if entry['checkpoint']=='hybrid':directory=directory/'hybrid_checkpoint'
        cache.load_gnn(directory)
        base=entry['config']
        path=root/f'validation_fusion_{entry["name"]}_{entry["checkpoint"]}.csv'
        rows=pd.read_csv(path).to_dict('records')
        progress(root,'candidate_boundary_extension',variant=entry['name'],checkpoint=entry['checkpoint'])
        for depth in [10000,20000]:
            prepared={u:candidate_indices(cache,u,base['alpha'],depth,base['strategy']) for u in cache.active}
            for omega in sorted(set([base['omega'],max(.1,round(base['omega']-.1,2)),min(.9,round(base['omega']+.1,2))])):
                for gamma in [1.,1.1,1.25,1.5,2.]:
                    cfg={k:base[k] for k in ['alpha','strategy','fusion']}
                    cfg.update(candidate_k=depth,omega=omega,gamma=gamma)
                    rows.append({**cfg,**cache.metrics(config_ranking(cache,cfg,prepared))})
        entry['config']=select_row(rows,reference)
        pd.DataFrame(rows).to_csv(path,index=False)
    winner=max(selections,key=lambda x:(joint_score(x['config'],reference),x['config']['ndcg']+x['config']['recall']))
    gnn_winner=max(baseline_candidates,key=lambda x:(joint_score(x['metrics'],reference),x['metrics']['ndcg']+x['metrics']['recall']))
    selection={'mf_alpha':alpha['alpha'],'hybrid':winner,'gnn':gnn_winner,
        'validation_reference':reference,'validation_joint_lift':joint_score(winner['config'],reference),
        'locked_utc':pd.Timestamp.now(tz='UTC').isoformat(),'outer_test_used_for_selection':False}
    dump_json(root/'selection.json',selection)
    selection_hash=sha256(root/'selection.json')
    del cache;torch.cuda.empty_cache()
    metric_rows=[];detail_rows=[];timings=[]
    for seed in [42,43,44]:
        progress(root,'final_refit',seed=seed,selection_hash=selection_hash)
        out=root/'runs'/f'seed_{seed}'
        if not out.exists():clone_protocol(datasets,out)
        if reuse_root is not None and not (out/'als_artifacts.json').exists() and (Path(reuse_root)/'runs'/f'seed_{seed}'/'als_artifacts.json').exists():
            link_mf(Path(reuse_root)/'runs'/f'seed_{seed}',out)
        t=time.perf_counter()
        if not (out/'als_artifacts.json').exists():ALSRecommender(out,random_state=seed,**MF_KWARGS).run()
        cache=ScoreCache(out)
        models={};gnn_seconds={}
        for label,entry in [('GNN',gnn_winner),('Hybrid',winner)]:
            tr=entry['trial'];epochs=tr['baseline_epochs'] if entry['checkpoint']=='baseline' else tr['hybrid_epochs']
            directory=out/label.lower()
            if reuse_root is not None and not directory.exists():
                previous=Path(reuse_root)/'selection.json'
                if previous.exists():
                    previous=json.loads(previous.read_text(encoding='utf-8'))
                    old=previous['gnn' if label=='GNN' else 'hybrid']
                    old_tr=old['trial'];old_epochs=old_tr['baseline_epochs'] if old['checkpoint']=='baseline' else old_tr['hybrid_epochs']
                    old_directory=Path(reuse_root)/'runs'/f'seed_{seed}'/label.lower()
                    if old_tr['variant']==tr['variant'] and old_epochs==epochs and (old_directory/'gnn_artifacts.json').exists():
                        link_tree(old_directory,directory)
            if not directory.exists():clone_protocol(datasets,directory)
            link_mf(out,directory)
            if not (directory/'gnn_artifacts.json').exists():
                kwargs=gnn_kwargs(seed,tr['variant'],epochs)
                kwargs['early_stopping_patience']=None
                generator=GNNEmbeddingGenerator(processed_dir=directory,**kwargs)
                generator.run()
                del generator;torch.cuda.empty_cache()
            cache.load_gnn(directory)
            gnn_seconds[label]=require_artifacts(directory,'gnn')['training_seconds']
            if label=='GNN':models['GNN']=cache.ranking(cache.gnn)
            else:models['Hybrid']=config_ranking(cache,winner['config'])
        models.update({'ALS':cache.ranking(cache.als),'BPR':cache.ranking(cache.bpr),
            'ALS+BPR':cache.ranking({u:cache.mf(u,alpha['alpha']) for u in cache.active})})
        # An explicit implementation ablation, using exactly the same candidate set.
        models['Hybrid_ALS_only']=config_ranking(cache,winner['config'],legacy_als=True)
        for label,rankings in models.items():
            records=[(u,int(cache.ids[i]),rank) for u,items in rankings.items() for rank,i in enumerate(items,1)]
            pd.DataFrame(records,columns=['visitorid','itemid','rank']).to_csv(out/f'recommendations_{label.replace("+","_")}.csv',index=False)
            for k in KS:
                details=cache.details(rankings,k)
                details.to_csv(out/f'metrics_{label.replace("+","_")}_k{k}.csv',index=False)
                metric_rows.append({'seed':seed,'model':label,'k':k,'users':len(details),
                    'missing_users':int((details.recommendations==0).sum()),
                    **{n:float(details[n].mean()) for n in ('ndcg','recall','precision','hit_rate')}})
                if k==50:
                    details['model']=label;details['seed']=seed;detail_rows.append(details)
        timings.append({'seed':seed,'seconds_total':time.perf_counter()-t,**gnn_seconds})
        dump_json(out/'complete.json',{'selection_hash':selection_hash,'seed':seed,
            'files':{p.name:sha256(p) for p in out.glob('metrics_*.csv')}})
        pd.DataFrame(metric_rows).to_csv(root/'experiment_metrics.csv',index=False)
        pd.DataFrame(timings).to_csv(root/'timing.csv',index=False)
        del cache;torch.cuda.empty_cache()
    metrics=pd.DataFrame(metric_rows)
    summary=metrics.groupby(['model','k'])[['ndcg','recall','precision','hit_rate']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns]
    summary.to_csv(root/'model_summary.csv',index=False)
    detail=pd.concat(detail_rows)
    paired=[]
    for model in ['ALS','BPR','ALS+BPR','GNN']:
        for metric in ['ndcg','recall']:
            a=detail.loc[detail.model=='Hybrid'].groupby('visitorid')[metric].mean().sort_index()
            b=detail.loc[detail.model==model].groupby('visitorid')[metric].mean().reindex(a.index)
            mean,lo,hi=paired_bootstrap(a,b)
            paired.append({'comparison':'Hybrid - '+model,'metric':metric,'difference':mean,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(paired).to_csv(root/'paired_differences.csv',index=False)
    primary=summary.loc[summary.k==50].set_index('model')
    competitors=['ALS','BPR','ALS+BPR','GNN']
    success=all(primary.loc['Hybrid',n+'_mean']>primary.loc[competitors,n+'_mean'].max() for n in ['ndcg','recall'])
    dump_json(root/'result.json',{'success':bool(success),'criterion':'3-seed mean original NDCG@50 AND Recall@50 above every baseline',
        'selection_hash':selection_hash,'seconds_total':time.time()-started,
        'primary':primary.reset_index().to_dict('records'),'paired':paired})
    progress(root,'completed',success=bool(success),seconds=time.time()-started)
    print(primary.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'results/hybrid_joint_8020_v3')
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--max-hours',type=float,default=24.)
    parser.add_argument('--reuse-root',type=Path,help='Reuse immutable matching model artifacts; rescore validation under this source version')
    parser.add_argument('--reuse-validation-metrics',action='store_true',help='Reuse parent validation CSVs only when numerical modules and data hashes match')
    args=parser.parse_args()
    run(args.output_dir,args.source_dir,args.max_hours,args.reuse_root,args.reuse_validation_metrics)
