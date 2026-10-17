"""Numerically equivalent cached validation scoring, deterministic boundary ties."""
import numpy as np
from recommendation_scoring import top_indices,minmax,ranks_from_scores
from improve_hybrid import candidate_indices


def fast_top_indices(scores,ids,k):
    scores=np.asarray(scores)
    if k<1 or not np.isfinite(scores).all():raise ValueError('Positive K and finite scores are required')
    if len(scores)<=k:return top_indices(scores,ids,k)
    boundary=np.partition(scores,len(scores)-k)[len(scores)-k]
    pool=np.flatnonzero(scores>=boundary)
    # Include every boundary tie, then use the authoritative item-ID tie rule.
    return pool[top_indices(scores[pool],np.asarray(ids)[pool],k)]


def cached_grid(cache,grid,seed,name,callback=None):
    rows=[];last=None;prepared={};values={}
    for n,cfg in enumerate(grid):
        key=(cfg['alpha'],cfg['strategy'],cfg['candidate_k'])
        if key!=last:
            prepared={u:candidate_indices(cache,u,cfg['alpha'],cfg['candidate_k'],cfg['strategy']) for u in cache.active}
            values={}
            for uid,idx in prepared.items():
                mf=cache.mf(uid,cfg['alpha'])[idx];gnn=cache.gnn[uid][idx]
                values[uid]=(minmax(mf),minmax(gnn),ranks_from_scores(mf),ranks_from_scores(gnn))
            last=key
            if callback is not None:callback(n,len(grid))
        rankings={};weight=cfg['omega'];gamma=cfg['gamma']
        for uid,idx in prepared.items():
            mf,gnn,mr,gr=values[uid]
            if cfg['fusion']=='minmax':scores=weight*mf+(1-weight)*gnn
            elif cfg['fusion']=='rrf':scores=weight/(60+mr)+(1-weight)/(60+gr)
            else:raise ValueError('Unknown fusion')
            scores=scores*np.where(cache.available[idx],gamma,1.)
            rankings[uid]=idx[fast_top_indices(scores,cache.ids[idx],50)]
        rows.append({'seed':seed,'name':name,**cfg,**cache.metrics(rankings)})
    return rows
