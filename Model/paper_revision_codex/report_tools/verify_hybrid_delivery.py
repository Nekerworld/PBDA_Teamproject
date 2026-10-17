"""Independent replay of final ranking CSVs and delivery integrity checks."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(path,truth,k):
    frame=pd.read_csv(path)
    if frame.duplicated(['visitorid','itemid']).any():raise ValueError('Duplicate recommendations')
    lists=frame.sort_values(['visitorid','rank']).groupby('visitorid').itemid.agg(list).to_dict()
    rows=[]
    for user,relevant in sorted(truth.items()):
        items=lists.get(user,[])[:k]
        dcg=sum(1/math.log2(i+2) for i,item in enumerate(items) if item in relevant)
        idcg=sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
        rows.append((dcg/idcg,len(set(items)&relevant)/len(relevant)))
    return np.asarray(rows).mean(axis=0)


def main(root,output):
    root=Path(root);checks=[]
    selection=root/'selection.json'
    plan=json.loads((root/'experiment_plan.json').read_text(encoding='utf-8'))
    assert plan['test_size']==.2 and plan['use_if'] is False
    assert plan['seeds']==[42,43,44]
    selected=json.loads(selection.read_text(encoding='utf-8'))
    assert selected['outer_test_used_for_selection'] is False
    assert 0<selected['hybrid']['config']['omega']<1
    checks.append('80:20 no-IF plan and nondegenerate MF+GNN weights')
    baseline_parent=None
    if plan.get('validation_seeds'):
        baseline_parent=Path(plan['parent'])
        if not baseline_parent.is_absolute():baseline_parent=root.parents[1]/baseline_parent
    metrics=pd.read_csv(root/'experiment_metrics.csv')
    details=[];cohort=None;training_protocol=[]
    for seed in [42,43,44]:
        directory=root/'runs'/f'seed_{seed}'
        complete=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
        assert complete['selection_hash']==digest(selection)
        if baseline_parent is not None:
            for model in ['ALS','BPR','GNN']:
                filename=f'recommendations_{model}.csv'
                assert digest(directory/filename)==digest(baseline_parent/'runs'/f'seed_{seed}'/filename),f'Existing baseline changed: {seed}/{model}'
        for name,expected in complete['files'].items():assert digest(directory/name)==expected,name
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['config']['test_size']==.2 and manifest['use_if'] is False
        for label,key in [('GNN','gnn'),('Hybrid','hybrid')]:
            entry=selected[key];trial=entry['trial']
            epochs=trial['baseline_epochs'] if entry['checkpoint']=='baseline' else trial['hybrid_epochs']
            artifact=directory/key/'gnn_artifacts.json';history_path=directory/key/'gnn_training_history.json'
            meta=json.loads(artifact.read_text(encoding='utf-8'))
            history=json.loads(history_path.read_text(encoding='utf-8'))
            assert meta['chosen_epochs']==history['chosen_epochs']==epochs
            assert history['mode']=='fixed_epochs'
            assert meta['protocol_id']==manifest['protocol_id']
            reused_from=None
            if baseline_parent is not None:
                for previous_label in ['gnn','hybrid']:
                    previous=baseline_parent/'runs'/f'seed_{seed}'/previous_label/'gnn_artifacts.json'
                    if previous.exists() and digest(previous)==digest(artifact):
                        try:reused_from=str(previous.relative_to(root.parents[1]))
                        except ValueError:reused_from=str(previous)
                        break
            training_protocol.append({'seed':seed,'model':label,'epochs':epochs,'refit_mode':history['mode'],
                'artifact':str(artifact.relative_to(root)),'artifact_sha256':digest(artifact),
                'reused_from':reused_from,
                'history_sha256':digest(history_path),'producer_static_selection_metric':meta.get('selection_metric'),
                'effective_epoch_selection':'Initial seed42 inner validation, fixed before these refits',
                'effective_configuration_selection':'Three-seed inner validation mean joint NDCG/Recall' if plan.get('validation_seeds') else 'Initial inner validation joint NDCG/Recall'})
        test=pd.read_csv(directory/'events_test_original.csv')
        test=test.loc[test.event.isin(['addtocart','transaction'])]
        truth=test.groupby('visitorid').itemid.agg(lambda x:set(x)).to_dict()
        users=tuple(sorted(truth))
        if cohort is None:cohort=users
        assert cohort==users
        catalog=set(pd.read_csv(directory/'candidate_items.csv').itemid)
        for model in ['ALS','BPR','ALS+BPR','GNN','Hybrid','Hybrid_ALS_only']:
            path=directory/f'recommendations_{model.replace("+","_")}.csv'
            records=pd.read_csv(path)
            assert set(records.itemid)<=catalog
            assert set(records.visitorid)<=set(users)
            for k in [10,20,30,40,50]:
                expected=metrics.loc[(metrics.seed==seed)&(metrics.model==model)&(metrics.k==k)]
                assert len(expected)==1
                actual=replay(path,truth,k)
                assert np.allclose(actual,expected[['ndcg','recall']].to_numpy()[0],rtol=1e-10,atol=1e-12),(seed,model,k)
        checks.append(f'Seed {seed}: completion hashes, catalog, cohort and all 30 metric pairs replayed')
    summary=metrics.groupby(['model','k'])[['ndcg','recall']].mean()
    primary=summary.xs(50,level='k')
    baselines=['ALS','BPR','ALS+BPR','GNN']
    success=all(primary.loc['Hybrid',m]>primary.loc[baselines,m].max() for m in ['ndcg','recall'])
    result=json.loads((root/'result.json').read_text(encoding='utf-8'))
    assert result['success']==bool(success)
    checks.append('Success flag independently recomputed from all three seeds')
    if baseline_parent is not None:checks.append('All nine ALS/BPR/standalone-GNN recommendation files exactly match the prior baselines')
    protocol_path=root/'effective_training_protocol.json'
    protocol_path.write_text(json.dumps({'selection_hash':digest(selection),'models':training_protocol,
        'note':'The reusable producer writes a static selection_metric string even for fixed-epoch refits. Actual selection is the sealed selection.json and fixed_epochs history; original producer metadata are preserved.'},indent=2),encoding='utf-8')
    checks.append('Six model records match the sealed fixed epochs; effective protocol records matching artifact reuse and inherited static metadata')
    report={'verified':True,'users':len(cohort),'metric_pairs_replayed':90,'success':bool(success),'checks':checks,
        'effective_training_protocol_hash':digest(protocol_path),
        'experiment':str(root.resolve()),'selection_hash':digest(selection),'result_hash':digest(root/'result.json')}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.experiment,a.output)
