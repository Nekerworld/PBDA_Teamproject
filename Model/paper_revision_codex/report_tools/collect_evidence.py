"""Collect immutable factual evidence for the professor briefing, without training."""
import argparse,csv,hashlib,json
from pathlib import Path
import pandas as pd


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(project,output):
    project=Path(project);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=project/'data/research_revision_paper/datasets'
    result={'conditions':{},'splits':{},'event_overlap':{},'source_hashes':{}}
    for condition in ['if_events','no_if_events']:
        directory=source/condition
        with (directory/'dataset_statistics.csv').open(encoding='utf-8-sig') as f:stats=list(csv.DictReader(f))
        result['conditions'][condition]={'manifest':read(directory/'manifest.json'),'statistics':stats}
        result['source_hashes'][str(directory/'manifest.json')]=sha(directory/'manifest.json')
    directory=source/'no_if_events'
    # Full unfiltered chronological partitions are represented in audited statistics.
    for role,p in [('outer',directory),('inner',directory/'validation')]:
        m=read(p/'manifest.json')
        with (p/'dataset_statistics.csv').open(encoding='utf-8-sig') as f:stats={r['stage']:r for r in csv.DictReader(f)}
        result['splits'][role]={'cutoff_utc':m['cutoff_utc'],'train_before_filter':stats['train_before_filter'],
            'train_final':stats['train_final'],'test_before_filter':stats['test_before_filter'],
            'evaluation_users':m['evaluation_users'],'conditional_evaluation_users':m['conditional_evaluation_users']}
        train=pd.read_csv(p/'events_train_clean.csv',usecols=['event_id','timestamp'])
        test=pd.read_csv(p/'events_test_original.csv',usecols=['event_id','timestamp'])
        result['event_overlap'][role]={'retained_train_vs_original_positive_test':len(set(train.event_id)&set(test.event_id)),
            'max_train_timestamp':int(train.timestamp.max()),'min_positive_test_timestamp':int(test.timestamp.min())}
    legacy=project/'data/legacy_paper_lightgcn'
    retained=pd.read_csv(directory/'events_train_clean.csv',usecols=['visitorid','itemid'])
    reference=set(pd.read_csv(directory/'evaluation_users.csv').visitorid)
    active=reference&set(retained.visitorid)
    catalog=set(retained.itemid)
    labels=pd.read_csv(directory/'events_test_original.csv')
    truth=labels.loc[labels.event.isin(['addtocart','transaction'])].groupby('visitorid').itemid.agg(set)
    found={u:len(items&catalog) if u in active else 0 for u,items in truth.items()}
    result['evaluation_support']={'reference_users':len(reference),'retained_user_vectors':len(active),
        'missing_user_vectors':len(reference-active),'users_with_reachable_truth':sum(n>0 for n in found.values()),
        'retained_users_without_reachable_truth':sum(u in active and n==0 for u,n in found.items()),
        'full_catalog_recall_ceiling':sum(n/len(truth[u]) for u,n in found.items())/len(reference),
        'full_catalog_recall50_ceiling':sum(min(50,n)/len(truth[u]) for u,n in found.items())/len(reference)}
    train=pd.read_csv(legacy/'gnn_train_pairs.csv')
    val=pd.read_csv(legacy/'gnn_validation_pairs.csv')
    events=pd.read_csv(legacy/'events_train_clean.csv',usecols=['visitorid','itemid','event'])
    pairs=events[['visitorid','itemid']].drop_duplicates()
    result['legacy_reconstruction']={'pair_train_rows':len(train),'pair_validation_rows':len(val),
        'train_validation_pair_overlap':len(train.merge(val,on=['visitorid','itemid'])),
        'validation_pairs_in_full_graph':len(val.merge(pairs,on=['visitorid','itemid'])),
        'training_events':len(events),'training_transactions':int(events.event.eq('transaction').sum()),
        'evidence_status':'Reconstructed from past_code; original submission-time artifacts were overwritten according to chat.'}
    result['legacy_reconstruction']['split_statistics']=pd.read_csv(legacy/'split_summary.csv').fillna('').to_dict('records')
    result['read_only_sources']={'paper':r'C:\YS\TUK\S4E2.5\버전 관리\IEEE_ACCESS_submission.pdf',
        'chat_team':r'C:\Users\chris\Downloads\KakaoTalk_20261016_2327_45_064_group.txt',
        'chat_professor':r'C:\Users\chris\Downloads\KakaoTalk_20261016_2327_57_050_group.txt'}
    for p in result['read_only_sources'].values():result['source_hashes'][p]=sha(Path(p))
    (out/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'splits':result['splits'],'event_overlap':result['event_overlap'],
                      'legacy_reconstruction':result['legacy_reconstruction']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.project,args.output)
