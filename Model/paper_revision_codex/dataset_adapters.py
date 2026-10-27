"""Raw Taobao UserBehavior and Diginetica (CIKM Cup 2016) files in the RetailRocket layout the pipeline reads.

Output folder: ``events.csv`` (timestamp ms, visitorid, event in view/addtocart/transaction, itemid, transactionid),
``item_properties_part1.csv``/``part2.csv`` (only ``categoryid``, stamped at time 0 because it is static catalogue
metadata) and ``category_tree.csv`` (flat). Neither source has a stock flag, so every item gets ``has_available`` 0 and
the stock signal of the Hybrid is constant for every model. ``adapter.json`` records the source hashes, the mapping
and every dropped row count. No row is read or dropped because of its label or its position in a future test window.

Taobao: no header, columns user, item, category, behaviour (pv/cart/fav/buy), Unix seconds. pv -> view, cart ->
addtocart, buy -> transaction; fav has no counterpart in the shared view/cart/purchase schema and is dropped. Rows
outside the published period (2017-11-25 to 2017-12-03, UTC+8) are dropped. ``user_percent`` keeps a fixed share
of users chosen by a seeded hash of the user id, so the subset does not depend on file order or on behaviour.

Diginetica: ``train-item-views.csv`` and ``train-purchases.csv`` (``;``-separated). ``eventdate`` is the calendar date
of the event itself (it changes inside sessions that pass midnight) and ``timeframe`` is ms since the session start,
whose time of day is not given, so only the date is absolute: the event time is the event date (UTC midnight). Every
split cutoff then falls on a day boundary, and two sessions of one user on the same day cannot be put in the wrong
order across a cutoff; the order inside a day is not used. Signed-in users keep their ``userId``;
anonymous sessions are either one pseudo-user each (``anonymous='session'``, ids from ``SESSION_OFFSET``; they do not
return, so they add training interactions and are evaluated only when the session itself spans the cutoff) or dropped
(``anonymous='drop'``).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAOBAO_EVENTS = {'pv': 'view', 'cart': 'addtocart', 'buy': 'transaction'}
TAOBAO_PERIOD = (1511539200, 1512316800)  # 2017-11-25 00:00 to 2017-12-04 00:00, UTC+8
SESSION_OFFSET = 10 ** 10


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _keep_user(users, percent, seed):
    """Seeded hash of the user id: the same users are kept whatever the file order."""
    if percent is None:
        return np.ones(len(users), dtype=bool)
    with np.errstate(over='ignore'):  # SplitMix64; pandas' hash_array ignores its key for integer arrays
        x = np.asarray(users, dtype=np.int64).astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(10_000)) < np.uint64(round(percent * 100))


def _write(out, events, categories, info):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    events = events.sort_values(['timestamp', 'visitorid', 'itemid'], kind='stable').reset_index(drop=True)
    events[['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']].to_csv(out / 'events.csv', index=False)
    properties = pd.DataFrame({'timestamp': 0, 'itemid': categories.itemid.to_numpy(),
                               'property': 'categoryid', 'value': categories.categoryid.astype(str).to_numpy()})
    half = len(properties) // 2
    properties.iloc[:half].to_csv(out / 'item_properties_part1.csv', index=False)
    properties.iloc[half:].to_csv(out / 'item_properties_part2.csv', index=False)
    pd.DataFrame({'categoryid': sorted(categories.categoryid.unique()), 'parentid': np.nan}).to_csv(
        out / 'category_tree.csv', index=False)
    info = {**info, 'events': int(len(events)), 'event_counts': events.event.value_counts().to_dict(),
            'users': int(events.visitorid.nunique()), 'items': int(events.itemid.nunique()),
            'first_timestamp': int(events.timestamp.min()), 'last_timestamp': int(events.timestamp.max())}
    (out / 'adapter.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    return info


def taobao_to_raw(source, out, user_percent=None, seed=42, chunksize=5_000_000):
    source = Path(source)
    kept, counts = [], {'rows': 0, 'fav_dropped': 0, 'outside_period': 0, 'unknown_behaviour': 0, 'users_not_sampled': 0}
    categories = []
    for chunk in pd.read_csv(source, header=None, names=['user', 'item', 'category', 'behaviour', 'ts'],
                             dtype={'user': np.int64, 'item': np.int64, 'category': np.int64, 'behaviour': str,
                                    'ts': np.int64}, chunksize=chunksize):
        counts['rows'] += len(chunk)
        inside = chunk.ts.between(TAOBAO_PERIOD[0], TAOBAO_PERIOD[1] - 1)
        counts['outside_period'] += int((~inside).sum())
        chunk = chunk.loc[inside]
        fav = chunk.behaviour.eq('fav')
        counts['fav_dropped'] += int(fav.sum())
        chunk = chunk.loc[~fav]
        known = chunk.behaviour.isin(TAOBAO_EVENTS)
        counts['unknown_behaviour'] += int((~known).sum())
        chunk = chunk.loc[known]
        sampled = _keep_user(chunk.user.to_numpy(), user_percent, seed)
        counts['users_not_sampled'] += int((~sampled).sum())
        chunk = chunk.loc[sampled]
        categories.append(chunk[['item', 'category']].drop_duplicates())
        kept.append(pd.DataFrame({'timestamp': chunk.ts.to_numpy() * 1000, 'visitorid': chunk.user.to_numpy(),
                                  'event': chunk.behaviour.map(TAOBAO_EVENTS).to_numpy(),
                                  'itemid': chunk.item.to_numpy(), 'transactionid': np.nan}))
    events = pd.concat(kept, ignore_index=True)
    categories = pd.concat(categories).rename(columns={'item': 'itemid', 'category': 'categoryid'})
    conflicting = int(categories.drop_duplicates().itemid.duplicated().sum())
    categories = categories.drop_duplicates('itemid', keep='first')  # Taobao items carry one category; count any other
    return _write(out, events, categories, {
        'source': 'taobao_user_behavior', 'files': {source.name: _sha256(source)}, 'mapping': TAOBAO_EVENTS,
        'period_seconds_utc8': TAOBAO_PERIOD, 'user_percent': user_percent, 'seed': seed,
        'items_with_more_than_one_category': conflicting, 'dropped': counts})


def diginetica_to_raw(source, out, anonymous='session'):
    if anonymous not in ('session', 'drop'):
        raise ValueError("anonymous must be 'session' or 'drop'")
    source = Path(source)
    views = pd.read_csv(source / 'train-item-views.csv', sep=';')
    purchases = pd.read_csv(source / 'train-purchases.csv', sep=';')
    categories = pd.read_csv(source / 'product-categories.csv', sep=';').rename(columns={'itemId': 'itemid',
                                                                                         'categoryId': 'categoryid'})

    def frame(rows, event):
        day = (pd.to_datetime(rows.eventdate, format='%Y-%m-%d', utc=True)
               - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(milliseconds=1)  # unit-free: pandas may parse to s or us
        user = pd.to_numeric(rows.userId, errors='coerce')
        signed = user.notna().to_numpy()
        visitor = np.where(signed, user.fillna(0).astype(np.int64), SESSION_OFFSET + rows.sessionId.astype(np.int64))
        return pd.DataFrame({'timestamp': day.to_numpy(),
                             'visitorid': visitor, 'event': event, 'itemid': rows.itemId.astype(np.int64).to_numpy(),
                             'transactionid': rows.ordernumber.to_numpy() if 'ordernumber' in rows else np.nan,
                             'signed_in': signed})
    events = pd.concat([frame(views, 'view'), frame(purchases, 'transaction')], ignore_index=True)
    counts = {'views': int(len(views)), 'purchases': int(len(purchases)),
              'anonymous_events': int((~events.signed_in).sum())}
    if anonymous == 'drop':
        events = events.loc[events.signed_in]
    if (events.visitorid[events.signed_in] >= SESSION_OFFSET).any():
        raise ValueError('A signed-in user id collides with the pseudo-user range')
    categories = categories.loc[categories.itemid.isin(events.itemid)].drop_duplicates('itemid', keep='first')
    files = {name: _sha256(source / name) for name in ('train-item-views.csv', 'train-purchases.csv',
                                                        'product-categories.csv')}
    return _write(out, events.drop(columns='signed_in'), categories, {
        'source': 'diginetica_cikm2016', 'files': files, 'anonymous': anonymous, 'session_offset': SESSION_OFFSET,
        'timestamp_rule': 'eventdate (UTC midnight); timeframe is relative to an unknown session start and is not used',
        'dropped': counts})
