"""Behavior-weighted pairwise learning with history-safe negative sampling.

LightGCN may propagate on unweighted or behavior-weighted user-item edges.
Pairwise loss always uses the pair weights. RetailRocket encodes a negative
number as a leading ``n`` and may store several property tokens in one value.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

BEHAVIOR_WEIGHTS = {'view': 1., 'addtocart': 6., 'transaction': 12.}


def interaction_table(events, weighted=True, time_decay=False, half_life_days=7.0):
    """Aggregate visitor-item weights. Optional exponential time decay on events."""
    frame = events[['visitorid', 'itemid', 'event']].copy()
    frame['weight'] = frame.event.map(BEHAVIOR_WEIGHTS if weighted else {k: 1. for k in BEHAVIOR_WEIGHTS})
    if frame.weight.isna().any():
        raise ValueError('Unknown event type')
    if time_decay:
        if 'timestamp' not in events.columns:
            raise ValueError('time_decay requires a timestamp column')
        if half_life_days <= 0:
            raise ValueError('half_life_days must be positive')
        stamps = events['timestamp'].to_numpy(dtype=np.float64)
        age_days = (stamps.max() - stamps) / (1000.0 * 60.0 * 60.0 * 24.0)
        frame['weight'] = frame.weight.to_numpy(dtype=np.float64) * (0.5 ** (age_days / half_life_days))
    pairs = frame.groupby(['visitorid', 'itemid'], as_index=False).weight.sum()
    if not weighted:
        pairs['weight'] = 1.  # Binary-pair ablation, not repeated-event weighting.
    return pairs


def complement_indices(draws, forbidden, n_items):
    """Map uniform draws on the complement back to item indexes."""
    forbidden = np.unique(np.asarray(forbidden, dtype=np.int64))
    draws = np.asarray(draws, dtype=np.int64)
    remaining = n_items - len(forbidden)
    if remaining <= 0:
        raise ValueError('No unobserved item remains')
    if len(draws) and (draws.min() < 0 or draws.max() >= remaining):
        raise ValueError('Complement draw is out of range')
    shifted = forbidden - np.arange(len(forbidden))
    mapped = draws + np.searchsorted(shifted, draws, side='right')
    if len(mapped) and (mapped.max() >= n_items or np.isin(mapped, forbidden).any()):
        raise RuntimeError('Complement mapping returned an observed item')
    return mapped


class NegativeSampler:
    """Negatives from each user's unobserved training items.

    ``mode='uniform'`` is rejection + complement fallback.
    ``mode='popularity'`` samples unobserved items with probability proportional
    to training popularity (with a small floor so rare items stay reachable).
    Saturated users get -1 and are skipped. Outer-test labels must never be passed.
    """
    def __init__(self, pairs, n_items, mode='uniform', item_popularity=None):
        if n_items < 1:
            raise ValueError('At least one item is required')
        if mode not in {'uniform', 'popularity'}:
            raise ValueError('Unknown negative sampling mode')
        self.n_items = int(n_items)
        self.mode = mode
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        unique = np.unique(pairs, axis=0)
        self.keys = unique[:, 0] * self.n_items + unique[:, 1]
        if len(unique) and unique[:, 0].max() > 10_000_000:
            raise ValueError('Negative sampling expects compact user indexes')
        self.counts = np.bincount(unique[:, 0]) if len(unique) else np.zeros(0, dtype=int)
        starts = np.r_[0, np.flatnonzero(np.diff(unique[:, 0])) + 1, len(unique)] if len(unique) else [0]
        self.blocked = {int(unique[a, 0]): unique[a:b, 1]
                        for a, b in zip(starts[:-1], starts[1:])} if len(unique) else {}
        if item_popularity is None:
            pop = np.bincount(unique[:, 1], minlength=self.n_items).astype(np.float64) if len(unique) else np.ones(self.n_items)
        else:
            pop = np.asarray(item_popularity, dtype=np.float64)
            if pop.shape != (self.n_items,):
                raise ValueError('item_popularity must match n_items')
        self.popularity = np.maximum(pop, 1.0)

    def sample(self, users, rng):
        users = np.asarray(users, dtype=np.int64)
        if self.mode == 'popularity':
            return self._sample_popularity(users, rng)
        negatives = rng.integers(self.n_items, size=len(users))
        counts = np.zeros(len(users), dtype=int)
        known = (users >= 0) & (users < len(self.counts))
        counts[known] = self.counts[users[known]]
        active = counts < self.n_items
        negatives[~active] = -1

        def collisions():
            values = users * self.n_items + negatives
            loc = np.searchsorted(self.keys, values)
            hit = np.zeros(len(users), dtype=bool)
            valid = active & (loc < len(self.keys))
            hit[valid] = self.keys[loc[valid]] == values[valid]
            return hit

        conflict = collisions()
        for _ in range(32):
            if not conflict.any():
                return negatives
            negatives[conflict] = rng.integers(self.n_items, size=int(conflict.sum()))
            conflict = collisions()
        for uid in np.unique(users[conflict]):
            rows = np.flatnonzero((users == uid) & conflict)
            forbidden = self.blocked.get(int(uid), np.empty(0, dtype=np.int64))
            if self.n_items - len(forbidden) <= 0:
                negatives[rows] = -1
                continue
            draws = rng.integers(self.n_items - len(forbidden), size=len(rows))
            negatives[rows] = complement_indices(draws, forbidden, self.n_items)
        return negatives

    def _sample_popularity(self, users, rng):
        users = np.asarray(users, dtype=np.int64)
        probabilities = self.popularity / self.popularity.sum()
        negatives = rng.choice(self.n_items, size=len(users), replace=True, p=probabilities)
        counts = np.zeros(len(users), dtype=int)
        known = (users >= 0) & (users < len(self.counts))
        counts[known] = self.counts[users[known]]
        active = counts < self.n_items
        negatives[~active] = -1

        def collisions():
            values = users * self.n_items + negatives
            loc = np.searchsorted(self.keys, values)
            hit = np.zeros(len(users), dtype=bool)
            valid = active & (loc < len(self.keys))
            hit[valid] = self.keys[loc[valid]] == values[valid]
            return hit

        conflict = collisions()
        for _ in range(64):
            if not conflict.any():
                return negatives
            negatives[conflict] = rng.choice(self.n_items, size=int(conflict.sum()), replace=True, p=probabilities)
            conflict = collisions()
        for uid in np.unique(users[conflict]):
            rows = np.flatnonzero((users == uid) & conflict)
            forbidden = self.blocked.get(int(uid), np.empty(0, dtype=np.int64))
            if self.n_items - len(forbidden) <= 0:
                negatives[rows] = -1
                continue
            weights = self.popularity.copy()
            weights[forbidden] = 0.0
            total = weights.sum()
            if total <= 0:
                negatives[rows] = -1
                continue
            negatives[rows] = rng.choice(self.n_items, size=len(rows), replace=True, p=weights / total)
        return negatives


def weighted_pairwise_loss(user, positive, negative, weights, regularization):
    differences = (user * (positive - negative)).sum(dim=-1)
    ranking = (torch.nn.functional.softplus(-differences) * weights).mean()
    penalty = regularization * (user.square().sum(-1) + positive.square().sum(-1) + negative.square().sum(-1)).mean()
    return ranking + penalty


def info_nce_loss(user, positive, negatives, weights, regularization, temperature=0.2):
    """Multi-negative InfoNCE. ``negatives`` has shape [batch, n_neg, dim]."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    pos = (user * positive).sum(dim=-1, keepdim=True) / temperature
    neg = (user.unsqueeze(1) * negatives).sum(dim=-1) / temperature
    logits = torch.cat([pos, neg], dim=1)
    log_prob = torch.nn.functional.log_softmax(logits, dim=1)[:, 0]
    ranking = -(log_prob * weights).mean()
    neg_reg = negatives.square().mean(dim=1).sum(dim=-1)
    penalty = regularization * (user.square().sum(-1) + positive.square().sum(-1) + neg_reg).mean()
    return ranking + penalty


class MFMixedNegativeSampler:
    """A fixed train-only hard pool plus uniform draws; never reads held-out labels.

    For each training user, 64 unobserved items are drawn and ranked by frozen
    ALS+BPR factors. The top 16 form a hard pool. Half of later draws use this
    pool, half remain uniform. IDs are explicitly aligned to graph vocabulary.
    """
    def __init__(self, pairs, n_items, directory, user_ids, item_ids, seed=42):
        from pathlib import Path
        root=Path(directory)
        self.base=NegativeSampler(pairs,n_items)
        self.n_items=n_items
        self.users=np.asarray(user_ids,dtype=np.int64)
        items=np.asarray(item_ids,dtype=np.int64)
        uids=pd.read_csv(root/'als_user_mapping.csv').visitorid.to_numpy()
        iids=pd.read_csv(root/'als_item_mapping.csv').itemid.to_numpy()
        uorder=pd.Index(uids).get_indexer(self.users)
        iorder=pd.Index(iids).get_indexer(items)
        if (uorder<0).any() or (iorder<0).any():
            raise ValueError('MF mining vocabulary differs from graph vocabulary')
        au=np.load(root/'als_user_factors.npy')[uorder]
        ai=np.load(root/'als_item_factors.npy')[iorder]
        bu=np.load(root/'bpr_user_factors.npy')[uorder]
        bi=np.load(root/'bpr_item_factors.npy')[iorder]
        self.pool=np.full((len(user_ids),16),-1,dtype=np.int32)
        rng=np.random.default_rng(seed+1701)
        for start in range(0,len(user_ids),512):
            users=np.arange(start,min(start+512,len(user_ids)))
            draws=self.base.sample(np.repeat(users,64),rng).reshape(-1,64)
            safe=np.maximum(draws,0)
            als=np.einsum('bd,bnd->bn',au[users],ai[safe])
            bpr=np.einsum('bd,bnd->bn',bu[users],bi[safe])
            def norm(x):
                return (x-x.min(axis=1,keepdims=True))/np.maximum(np.ptp(x,axis=1,keepdims=True),1e-12)
            scores=.5*norm(als)+.5*norm(bpr)
            scores[draws<0]=-np.inf
            chosen=np.argsort(-scores,axis=1,kind='stable')[:,:16]
            self.pool[users]=np.take_along_axis(draws,chosen,axis=1)

    def sample(self,users,rng):
        users=np.asarray(users,dtype=np.int64)
        uniform=self.base.sample(users,rng)
        hard=self.pool[users,rng.integers(16,size=len(users))]
        choose=(rng.random(len(users))<.5)&(hard>=0)
        return np.where(choose,hard,uniform)


def sample_negatives_matrix(sampler, users, rng, n_neg):
    """Draw ``n_neg`` negatives per row. Saturated users get -1 rows."""
    users = np.asarray(users, dtype=np.int64)
    if n_neg < 1:
        raise ValueError('n_neg must be positive')
    matrix = np.empty((len(users), n_neg), dtype=np.int64)
    for column in range(n_neg):
        matrix[:, column] = sampler.sample(users, rng)
    return matrix


@dataclass
class FactorResult:
    user_factors: np.ndarray
    item_factors: np.ndarray
    history: list


def train_weighted_bpr(matrix, factors=128, regularization=.03, epochs=128,
                       seed=42, device='cpu', weighted=True, batch_size=65536,
                       negative_mode='uniform'):
    """Explicit weighted BPR; CSR data are actual pair-loss weights."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    coo = matrix.tocoo()
    pairs = np.column_stack([coo.row, coo.col]).astype(np.int64)
    popularity = np.bincount(coo.col, minlength=matrix.shape[1]).astype(np.float64)
    sampler = NegativeSampler(pairs, matrix.shape[1], mode=negative_mode, item_popularity=popularity)
    raw = coo.data.astype(np.float32)
    weights = raw / raw.mean() if weighted else np.ones(len(raw), dtype=np.float32)
    users = torch.nn.Embedding(matrix.shape[0], factors, device=device)
    items = torch.nn.Embedding(matrix.shape[1], factors, device=device)
    torch.nn.init.normal_(users.weight, std=.05)
    torch.nn.init.normal_(items.weight, std=.05)
    optimizer = torch.optim.Adam([users.weight, items.weight], lr=.001)
    history = []
    for epoch in range(1, epochs + 1):
        negative = sampler.sample(pairs[:, 0], rng)
        order = rng.permutation(np.flatnonzero(negative >= 0))
        if not len(order):
            raise ValueError('No unobserved items remain for BPR negative sampling')
        total = 0.
        count = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            u = torch.as_tensor(pairs[idx, 0], device=device)
            p = torch.as_tensor(pairs[idx, 1], device=device)
            n = torch.as_tensor(negative[idx], device=device)
            w = torch.as_tensor(weights[idx], device=device)
            optimizer.zero_grad()
            loss = weighted_pairwise_loss(users(u), items(p), items(n), w, regularization)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite BPR loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        history.append({'epoch': epoch, 'loss': total / count, 'pairs': count,
                        'saturated_pairs_skipped': int((negative < 0).sum()),
                        'negative_mode': negative_mode})
        if epoch == 1 or epoch % 25 == 0:
            print(f'Weighted BPR epoch {epoch}/{epochs}: {total / count:.6f}', flush=True)
    return FactorResult(users.weight.detach().cpu().numpy(), items.weight.detach().cpu().numpy(), history)


def numeric_token(values):
    """Parse a full token. RetailRocket's leading n means a negative number."""
    text = pd.Series(values, copy=False).astype(str).str.strip()
    negative = text.str.match(r'^n\d+(?:\.\d+)?$')
    parsed = text.where(~negative, '-' + text.str.slice(1))
    return pd.to_numeric(parsed, errors='coerce')


def encode_properties(frame, mode='values', metadata_policy='latest', bins=10, min_frequency=2):
    """Train-period property nodes."""
    frame = frame.sort_values(['timestamp', 'itemid', 'property'], kind='stable')
    if metadata_policy == 'latest':
        frame = frame.drop_duplicates(['itemid', 'property'], keep='last').copy()
    elif metadata_policy == 'history':
        frame = frame.copy()
    else:
        raise ValueError('Unknown metadata policy')
    frame['property'] = frame.property.astype(str)
    frame['value'] = frame.value.astype(str)
    if mode == 'names':
        frame['node_key'] = frame.property
        frame['token'] = frame.property
        return frame
    if mode != 'values':
        raise ValueError('Unknown property mode')
    if bins < 2 or min_frequency < 1:
        raise ValueError('Invalid property encoding settings')
    parts = frame.assign(token=frame.value.str.split()).explode('token', ignore_index=True)
    parts['token'] = parts.token.fillna('').astype(str).str.strip()
    parts = parts.loc[parts.token != ''].copy()
    parts['number'] = numeric_token(parts.token)
    special = parts.property.isin(['categoryid', 'available'])
    categorical = ~special & parts.number.isna()
    numeric = ~special & parts.number.notna()
    parts['label'] = pd.Series(pd.NA, index=parts.index, dtype='object')
    parts.loc[special, 'label'] = parts.loc[special, 'token']
    if categorical.any():
        cat = parts.loc[categorical, ['property', 'token', 'itemid']].copy()
        freq = cat.groupby(['property', 'token'], sort=False).itemid.nunique().rename('freq')
        cat = cat.join(freq, on=['property', 'token'])
        parts.loc[categorical, 'label'] = np.where(cat.freq.to_numpy() < min_frequency, '<rare>', cat.token.to_numpy())
    for _, group in parts.loc[numeric].groupby('property', sort=True):
        values = group.number.to_numpy(dtype=float)
        if np.unique(values).size > bins:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                labels = np.array(['bin:0'] * len(group), dtype=object)
            else:
                labels = np.array(['bin:' + str(int(i)) for i in np.searchsorted(edges[1:-1], values, side='right')], dtype=object)
        else:
            labels = np.array([format(float(v), '.6g') for v in values], dtype=object)
        parts.loc[group.index, 'label'] = labels
    if parts.label.isna().any():
        raise ValueError('Property encoding left an unlabeled token')
    parts['node_key'] = [str((prop, label)) for prop, label in zip(parts.property, parts.label)]
    return parts


def item_cooccurrence_edges(pairs, item_map, min_count=2, max_neighbors=20):
    """Undirected item-item edges from shared users. Indexes are local item ids."""
    from collections import defaultdict
    users = defaultdict(list)
    for visitor, item in zip(pairs.visitorid.to_numpy(), pairs.itemid.to_numpy()):
        if item not in item_map:
            continue
        users[int(visitor)].append(int(item_map[item]))
    counts = defaultdict(int)
    for items in users.values():
        unique = sorted(set(items))
        if len(unique) < 2:
            continue
        # Cap complete-subgraph cost for heavy users.
        if len(unique) > max_neighbors + 1:
            unique = unique[:max_neighbors + 1]
        for i, left in enumerate(unique):
            for right in unique[i + 1:]:
                counts[(left, right)] += 1
    sources, targets, weights = [], [], []
    for (left, right), count in counts.items():
        if count < min_count:
            continue
        sources.extend([left, right])
        targets.extend([right, left])
        weights.extend([float(count), float(count)])
    if not sources:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.column_stack([sources, targets]).astype(np.int64), np.asarray(weights, dtype=np.float32)
