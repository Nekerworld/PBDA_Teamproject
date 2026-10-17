"""Three-seed validation selection after the disclosed unsuccessful v3 trial.

The prior outer test was observed. No outer labels enter this new selector;
this is an adaptive research revision, not a pristine once-only holdout claim.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_protocol import require_protocol,require_artifacts,dump_json,sha256
from experiments import clone_protocol,write_dataset_statistics_tables,paired_bootstrap
from model import ALSRecommender,GNNEmbeddingGenerator
from improve_hybrid import (ScoreCache,MF_KWARGS,KS,link_mf,link_tree,gnn_kwargs,
    progress,candidate_indices,config_ranking,joint_score)


def robust_select(rows,reference):
    """Each setting must have all three validation seeds, no test observations."""
    frame=pd.DataFrame(rows)
    keys=['name','alpha','strategy','fusion','candidate_k','omega','gamma']
    grouped=frame.groupby(keys,dropna=False).agg(ndcg=('ndcg','mean'),recall=('recall','mean'),
        seed_count=('seed','nunique'),ndcg_std=('ndcg','std'),recall_std=('recall','std')).reset_index()
    if not grouped.seed_count.eq(3).all():raise ValueError('Incomplete validation seed coverage')
    grouped['joint_lift']=grouped.apply(lambda r:joint_score(r,reference),axis=1)
    grouped['average_lift']=((grouped.ndcg/reference['ndcg']-1)+(grouped.recall/reference['recall']-1))/2
    grouped['gamma_distance']=(grouped.gamma-1).abs()
    ordered=grouped.sort_values(['joint_lift','average_lift','candidate_k','gamma_distance','omega'],
        ascending=[False,False,True,True,True],kind='stable')
    return ordered.iloc[0].to_dict(),grouped


def configuration_grid():
    # Fixed before executing the additional validation seeds, recorded in plan.
    return [dict(alpha=a,strategy=s,candidate_k=c,fusion=f,omega=w,gamma=g)
        for a in [.25,.5] for s in ['blend','balanced_union'] for c in [5000,10000,20000]
        for f in ['minmax','rrf'] for w in [.1,.2,.3,.4,.5,.7]
        for g in [1.,1.1,1.25,1.5]]


def run(root,parent,max_hours):
    root,parent=Path(root),Path(parent);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);start=time.time()
    pplan=json.loads((parent/'experiment_plan.json').read_text(encoding='utf-8'))
    old=json.loads((parent/'selection.json').read_text(encoding='utf-8'))
    trials=json.loads((parent/'validation_candidates.json').read_text(encoding='utf-8'))['hybrid']
    specifications=[('l2_uniform','baseline'),('l3_uniform','baseline'),('l2_mf_mixed','hybrid')]
    entries={name:next(t for t in trials if t['name']==name and t['checkpoint']==checkpoint)
        for name,checkpoint in specifications}
    grid=configuration_grid()
    frozen={'test_size':.2,'use_if':False,'core':3,'seeds':[42,43,44],
        'validation_seeds':[42,43,44],'historical_test_seen':True,'prior_test_result_observed':str(parent/'result.json'),
        'prior_test_success':False,'parent':str(parent),'max_hours':max_hours,
        'source':pplan['source'],'source_manifest_hash':pplan['source_manifest_hash'],
        'method':'three-seed mean validation min relative NDCG/Recall lift; fixed v3 standalone GNN comparator',
        'configurations_per_backbone':len(grid),'grid':grid,'backbones':specifications,
        'code':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    for name in ['model.py','learning.py','data_protocol.py','recommendation_scoring.py','ranking_evaluation.py','als_debug.py']:
        if frozen['code'][name]!=pplan['code'][name]:raise ValueError('Numerical training source changed: '+name)
    plan=root/'experiment_plan.json'
    if plan.exists():
        if json.loads(plan.read_text(encoding='utf-8'))!=frozen:raise ValueError('Changed plan; choose new directory')
    else:dump_json(plan,frozen)
    source=Path(pplan['source']);require_protocol(source)
    datasets=root/'datasets/no_if_events'
    if not datasets.exists():clone_protocol(source,datasets)
    write_dataset_statistics_tables(root/'datasets',root,'no_if_events')
    alpha_rows=[];base_rows=[];all_rows=[]
    for seed in [42,43,44]:
        mf=root/f'validation_seed_{seed}'/'mf'
        if not mf.exists():clone_protocol(datasets/'validation',mf)
        if seed==42 and not (mf/'als_artifacts.json').exists():link_mf(parent/'validation_mf',mf)
        progress(root,'robust_validation_mf',seed=seed)
        if not (mf/'als_artifacts.json').exists():ALSRecommender(mf,random_state=seed,**MF_KWARGS).run()
        if seed==42 and not (root/'validation_mf').exists():link_tree(mf,root/'validation_mf')
        cache=ScoreCache(mf)
        for alpha in np.round(np.arange(0,1.01,.1),1):
            met=cache.metrics(cache.ranking({u:cache.mf(u,float(alpha)) for u in cache.active}))
            alpha_rows.append({'seed':seed,'alpha':float(alpha),**met})
        for name,checkpoint in specifications:
            if time.time()-start>max_hours*3600:raise RuntimeError('Time allocation exhausted; existing artifacts preserved')
            entry=entries[name];trial=entry['trial'];epochs=trial['baseline_epochs'] if checkpoint=='baseline' else trial['hybrid_epochs']
            directory=root/f'validation_seed_{seed}'/name
            if seed==42 and not directory.exists():
                prior=Path(trial['directory'])
                if checkpoint=='hybrid':prior=prior/'hybrid_checkpoint'
                link_tree(prior,directory)
            if not directory.exists():clone_protocol(datasets/'validation',directory)
            link_mf(mf,directory)
            progress(root,'robust_validation_gnn',seed=seed,variant=name,epochs=epochs)
            if not (directory/'gnn_artifacts.json').exists():
                kw=gnn_kwargs(seed,trial['variant'],epochs);kw['early_stopping_patience']=None
                generator=GNNEmbeddingGenerator(processed_dir=directory,**kw);generator.run()
                del generator;torch.cuda.empty_cache()
            cache.load_gnn(directory)
            base_rows.append({'seed':seed,'name':name,**cache.metrics(cache.ranking(cache.gnn))})
            saved=root/f'robust_validation_{name}_s{seed}.csv'
            if saved.exists():rows=pd.read_csv(saved).to_dict('records')
            else:
                rows=[];prepared=None;last=None
                for n,cfg in enumerate(grid):
                    key=(cfg['alpha'],cfg['strategy'],cfg['candidate_k'])
                    if key!=last:
                        prepared={u:candidate_indices(cache,u,cfg['alpha'],cfg['candidate_k'],cfg['strategy']) for u in cache.active};last=key
                        progress(root,'robust_validation_fusion',seed=seed,variant=name,configuration=n,total=len(grid))
                    rows.append({'seed':seed,'name':name,**cfg,**cache.metrics(config_ranking(cache,cfg,prepared))})
                pd.DataFrame(rows).to_csv(saved,index=False)
            all_rows.extend(rows)
            # Match the schema consumed by diagnostics/reporting with seed42.
            if seed==42:
                tr={**trial,'directory':str(directory)}
                entries[name]={**entry,'checkpoint':'baseline','trial':{**tr,'baseline_epochs':epochs,'hybrid_epochs':epochs}}
                target=root/'trials'/name
                if not target.exists():link_tree(directory,target)
        del cache;torch.cuda.empty_cache()
        pd.DataFrame(alpha_rows).to_csv(root/'validation_mf_alpha_per_seed.csv',index=False)
        pd.DataFrame(base_rows).to_csv(root/'validation_gnn_per_seed.csv',index=False)
    alpha_mean=pd.DataFrame(alpha_rows).groupby('alpha')[['ndcg','recall']].mean().reset_index()
    alpha_ref={n:float(alpha_mean[n].max()) for n in ['ndcg','recall']}
    alpha_mean['joint_lift']=alpha_mean.apply(lambda r:joint_score(r,alpha_ref),axis=1)
    mf_alpha=float(alpha_mean.sort_values(['joint_lift','alpha'],ascending=[False,True]).iloc[0].alpha)
    alpha_mean.to_csv(root/'validation_mf_alpha.csv',index=False)
    gnn_mean=pd.DataFrame(base_rows).groupby('name')[['ndcg','recall']].mean()
    reference={n:max(alpha_ref[n],float(gnn_mean[n].max())) for n in ['ndcg','recall']}
    best,aggregate=robust_select(all_rows,reference)
    aggregate.to_csv(root/'robust_validation_aggregate.csv',index=False)
    winner=entries[best['name']]
    cfg={key:best[key] for key in ['alpha','strategy','fusion','candidate_k','omega','gamma']}
    cfg['candidate_k']=int(cfg['candidate_k']);cfg.update({n:float(best[n]) for n in ['ndcg','recall','joint_lift','average_lift']})
    winner={**winner,'config':cfg}
    # Existing strongest standalone architecture/checkpoint stays unchanged.
    gnn_winner=entries['l3_uniform']
    gnn_winner={**gnn_winner,'metrics':gnn_mean.loc['l3_uniform'].to_dict()}
    selection={'mf_alpha':mf_alpha,'hybrid':winner,'gnn':gnn_winner,
        'validation_reference':reference,'validation_joint_lift':float(best['joint_lift']),
        'validation_seeds':[42,43,44],'locked_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'outer_test_used_for_selection':False,'prior_test_observed':True}
    dump_json(root/'selection.json',selection);selection_hash=sha256(root/'selection.json')
    # Existing plot schema: keep mean rows, clearly label their three-seed scope.
    for name,entry in entries.items():
        rows=aggregate.loc[aggregate.name==name].copy();rows.to_csv(root/f'validation_fusion_{name}_baseline.csv',index=False)
    dump_json(root/'validation_candidates.json',{'hybrid':[winner],'gnn':[gnn_winner],
        'scope':'three-seed validation means; original 42-selected epochs fixed'})
    metric_rows=[];detail_rows=[];timings=[]
    for seed in [42,43,44]:
        progress(root,'final_refit',seed=seed,selection_hash=selection_hash)
        out=root/'runs'/f'seed_{seed}'
        if not out.exists():clone_protocol(datasets,out)
        if not (out/'als_artifacts.json').exists():link_mf(parent/'runs'/f'seed_{seed}',out)
        t=time.perf_counter();cache=ScoreCache(out);models={};seconds={}
        for label,entry in [('GNN',gnn_winner),('Hybrid',winner)]:
            trial=entry['trial'];epochs=trial['baseline_epochs'];directory=out/label.lower()
            prior_entry=old['gnn' if label=='GNN' else 'hybrid'];pt=prior_entry['trial']
            pe=pt['baseline_epochs'] if prior_entry['checkpoint']=='baseline' else pt['hybrid_epochs']
            prior_dir=parent/'runs'/f'seed_{seed}'/label.lower()
            if not directory.exists() and pt['variant']==trial['variant'] and pe==epochs:link_tree(prior_dir,directory)
            # If the Hybrid uses the same GNN architecture/epoch as the baseline,
            # reuse that identical numerical model rather than training twice.
            if label=='Hybrid' and not directory.exists() and trial['variant']==gnn_winner['trial']['variant'] and epochs==gnn_winner['trial']['baseline_epochs']:
                link_tree(out/'gnn',directory)
            if not directory.exists():clone_protocol(datasets,directory)
            link_mf(out,directory)
            if not (directory/'gnn_artifacts.json').exists():
                kw=gnn_kwargs(seed,trial['variant'],epochs);kw['early_stopping_patience']=None
                generator=GNNEmbeddingGenerator(processed_dir=directory,**kw);generator.run()
                del generator;torch.cuda.empty_cache()
            cache.load_gnn(directory);seconds[label]=require_artifacts(directory,'gnn')['training_seconds']
            models[label]=cache.ranking(cache.gnn) if label=='GNN' else config_ranking(cache,cfg)
        models.update({'ALS':cache.ranking(cache.als),'BPR':cache.ranking(cache.bpr),
            'ALS+BPR':cache.ranking({u:cache.mf(u,mf_alpha) for u in cache.active}),
            'Hybrid_ALS_only':config_ranking(cache,cfg,legacy_als=True)})
        for label,rankings in models.items():
            records=[(u,int(cache.ids[i]),rank) for u,items in rankings.items() for rank,i in enumerate(items,1)]
            pd.DataFrame(records,columns=['visitorid','itemid','rank']).to_csv(out/f'recommendations_{label.replace("+","_")}.csv',index=False)
            for k in KS:
                details=cache.details(rankings,k);details.to_csv(out/f'metrics_{label.replace("+","_")}_k{k}.csv',index=False)
                metric_rows.append({'seed':seed,'model':label,'k':k,'users':len(details),
                    'missing_users':int((details.recommendations==0).sum()),
                    **{n:float(details[n].mean()) for n in ['ndcg','recall','precision','hit_rate']}})
                if k==50:details['model']=label;details['seed']=seed;detail_rows.append(details)
        timings.append({'seed':seed,'seconds_total':time.perf_counter()-t,**seconds})
        dump_json(out/'complete.json',{'selection_hash':selection_hash,'seed':seed,
            'files':{p.name:sha256(p) for p in out.glob('metrics_*.csv')}})
        pd.DataFrame(metric_rows).to_csv(root/'experiment_metrics.csv',index=False)
        pd.DataFrame(timings).to_csv(root/'timing.csv',index=False)
        del cache;torch.cuda.empty_cache()
    metrics=pd.DataFrame(metric_rows)
    summary=metrics.groupby(['model','k'])[['ndcg','recall','precision','hit_rate']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns];summary.to_csv(root/'model_summary.csv',index=False)
    detail=pd.concat(detail_rows);paired=[]
    for model in ['ALS','BPR','ALS+BPR','GNN']:
        for metric in ['ndcg','recall']:
            a=detail.loc[detail.model=='Hybrid'].groupby('visitorid')[metric].mean().sort_index()
            b=detail.loc[detail.model==model].groupby('visitorid')[metric].mean().reindex(a.index)
            mean,lo,hi=paired_bootstrap(a,b)
            paired.append({'comparison':'Hybrid - '+model,'metric':metric,'difference':mean,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(paired).to_csv(root/'paired_differences.csv',index=False)
    primary=summary.loc[summary.k==50].set_index('model')
    success=all(primary.loc['Hybrid',n+'_mean']>primary.loc[['ALS','BPR','ALS+BPR','GNN'],n+'_mean'].max() for n in ['ndcg','recall'])
    dump_json(root/'result.json',{'success':bool(success),'criterion':'3-seed mean original NDCG@50 AND Recall@50 above every baseline',
        'selection_hash':selection_hash,'seconds_total':time.time()-start,
        'primary':primary.reset_index().to_dict('records'),'paired':paired})
    progress(root,'completed',success=bool(success),seconds=time.time()-start);print(primary.to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--max-hours',type=float,default=9)
    a=p.parse_args();run(a.output_dir,a.parent,a.max_hours)
