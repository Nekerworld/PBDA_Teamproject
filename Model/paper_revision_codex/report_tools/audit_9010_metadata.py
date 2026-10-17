"""Read-only provenance audit of the historical 90:10 directory."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd


def main(root,output):
    root=Path(root);directory=root/'datasets/if_events'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    stats=pd.read_csv(directory/'dataset_statistics.csv').set_index('stage')
    train=int(stats.loc['train_before_filter','events']);test=int(stats.loc['test_before_filter','events'])
    ratio=train/(train+test);configured=manifest['config']['test_size']
    runs=[]
    for seed in [42,43,44]:
        run=root/'runs'/f'if_events__main__s{seed}'
        complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
        actual=hashlib.sha256((run/'evaluation_artifacts.json').read_bytes()).hexdigest()
        runs.append({'seed':seed,'complete_present':True,'evaluation_hash_matches':actual==complete['evaluation_hash'],
            'gamma':complete['gamma'],'candidate_k':complete['candidate_k']})
    summary=pd.read_csv(root/'model_comparison_9010.csv')
    payload={'source':str(root),'train_events':train,'test_events':test,'actual_train_ratio':ratio,
        'manifest_config_test_size':configured,'manifest_actual_prefilter_train_ratio':manifest['actual_prefilter_train_ratio'],
        'manifest_config_matches_counts':abs((1-configured)-ratio)<1e-5,
        'train_purchases_before_filter':int(stats.loc['train_before_filter','transaction_events']),
        'train_purchases_after_filter':int(stats.loc['train_final','transaction_events']),
        'evaluation_users':manifest['evaluation_users'],'runs':runs,
        'summary':summary.to_dict('records'),
        'interpretation':'Historical counts describe 90:10; a nested manifest config retains 0.2. Completion hashes do not fix incorrect metadata.'}
    Path(output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='summary'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
