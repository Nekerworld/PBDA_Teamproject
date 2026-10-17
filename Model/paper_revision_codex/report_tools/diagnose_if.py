"""Train-only IF feature ablation explaining purchase signal collapse.

These explanatory fits never choose the primary preprocessing condition and do
not access outer-test outcomes. All original cached labels remain unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def main(dataset,output):
    dataset=Path(dataset);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    items=pd.read_csv(dataset/'item_anomaly_audit.csv')
    features=meta['if_features']
    params=meta['if_parameters']
    rows=[]
    groups=[]
    positive=items.transaction_count.gt(0)
    for name,mask in [('purchase_items',positive),('no_purchase_items',~positive)]:
        group=items.loc[mask]
        for feature in features:
            groups.append({'group':name,'feature':feature,'items':len(group),
                'mean':float(group[feature].mean()),'median':float(group[feature].median()),
                'q25':float(group[feature].quantile(.25)),'q75':float(group[feature].quantile(.75))})
    for name,columns in [('original_9',features),
            ('without_behavior_code',[c for c in features if c not in ['event_code_mean','event_code_std']]),
            ('behavior_code_only',['event_code_mean','event_code_std'])]:
        forest=IsolationForest(**params)
        labels=forest.fit_predict(items[columns])
        kept=labels==1
        rows.append({'condition':name,'features':';'.join(columns),'kept_items':int(kept.sum()),
            'removed_items':int((~kept).sum()),'kept_events':int(items.loc[kept,'event_count'].sum()),
            'removed_events':int(items.loc[~kept,'event_count'].sum()),
            'kept_purchases':int(items.loc[kept,'transaction_count'].sum()),
            'removed_purchases':int(items.loc[~kept,'transaction_count'].sum()),
            'kept_cart':int(items.loc[kept,'addtocart_count'].sum()),
            'agreement_with_cached_labels':float((labels==items.if_label).mean()) if name=='original_9' else None})
    pd.DataFrame(rows).to_csv(out/'if_feature_ablation.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'if_feature_groups.csv',index=False)
    payload={'training_only':True,'cutoff_utc':meta['cutoff_utc'],'parameters':params,'results':rows,
        'interpretation':'Controlled input-feature ablation, not verified bot classification. Original-label agreement must be checked before attributing differences.'}
    (out/'if_diagnosis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).drop(columns='features').to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.dataset,a.output)
