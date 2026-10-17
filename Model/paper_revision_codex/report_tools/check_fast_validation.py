"""Compare cached scoring with already completed full real-data v4 CSVs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from improve_hybrid import ScoreCache,config_ranking
from robust_hybrid import configuration_grid
from fast_fusion import cached_grid,fast_top_indices
from recommendation_scoring import hybrid_scores,top_indices


def main(root,output):
    root=Path(root);started=time.time();grid=configuration_grid()
    cache=ScoreCache(root/'validation_seed_42/mf',root/'validation_seed_42/l3_uniform')
    actual=pd.DataFrame(cached_grid(cache,grid,42,'l3_uniform'))
    expected=pd.read_csv(root/'robust_validation_l3_uniform_s42.csv')
    assert len(actual)==len(expected)==576
    errors={m:float(np.abs(actual[m]-expected[m]).max()) for m in ['ndcg','recall']}
    assert all(x<1e-14 for x in errors.values()),errors
    # Exact individual recommendation lists, in addition to aggregate metrics.
    from improve_hybrid import candidate_indices
    checks=0
    for cfg in grid[::47]:
        for uid in cache.active[::23]:
            idx=candidate_indices(cache,uid,cfg['alpha'],cfg['candidate_k'],cfg['strategy'])
            values=hybrid_scores(cache.mf(uid,cfg['alpha'])[idx],cache.gnn[uid][idx],cache.available[idx],cfg['omega'],cfg['gamma'],cfg['fusion'])
            assert np.array_equal(fast_top_indices(values,cache.ids[idx],50),top_indices(values,cache.ids[idx],50))
            checks+=1
    report={'verified':True,'real_configurations':576,'maximum_metric_error':errors,
        'exact_top50_lists_checked':checks,'users':len(cache.users),'seconds':time.time()-started,
        'scope':'All real-data validation settings of seed42 l3_uniform; no outer-test selection.'}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
