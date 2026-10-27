"""Taobao and Diginetica files in the documented layouts become the RetailRocket layout the preprocessor reads."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dataset_adapters as da
from data_protocol import UnifiedPreprocessor, load_source_tables

DAY = 86_400


def taobao_file(path, users=60, rows=6000, seed=0):
    rng = np.random.default_rng(seed)
    start, end = da.TAOBAO_PERIOD
    ts = np.sort(rng.integers(start, end, rows))
    ts[:5] = start - 10  # before the published period
    ts[-3:] = end + 10   # after it
    behaviour = rng.choice(['pv', 'cart', 'fav', 'buy'], rows, p=[.7, .12, .08, .1])
    item = (np.arange(rows) * 30 // rows + rng.integers(0, 8, rows)) % 30 + 1
    pd.DataFrame({'u': rng.integers(1, users + 1, rows), 'i': item, 'c': item % 4 + 100, 'b': behaviour, 't': ts}
                 ).to_csv(path, header=False, index=False)
    return path


def diginetica_dir(path, sessions=400, seed=0):
    rng = np.random.default_rng(seed)
    path.mkdir()
    views, purchases = [], []
    for s in range(1, sessions + 1):
        user = int(rng.integers(1, 40)) if rng.random() < .6 else None
        day = pd.Timestamp('2016-01-01') + pd.Timedelta(days=int(s * 150 // sessions))
        frames = np.sort(rng.integers(0, 3_600_000, int(rng.integers(3, 9))))
        items = (s * 30 // sessions + rng.integers(0, 8, len(frames))) % 30 + 1
        views += [(s, user, int(i), int(f), day.strftime('%Y-%m-%d')) for i, f in zip(items, frames)]
        if rng.random() < .4:
            purchases.append((s, user, int(frames[-1]) + 1, day.strftime('%Y-%m-%d'), 1000 + s, int(items[-1])))
    pd.DataFrame(views, columns=['sessionId', 'userId', 'itemId', 'timeframe', 'eventdate']).to_csv(
        path / 'train-item-views.csv', sep=';', index=False)
    pd.DataFrame(purchases, columns=['sessionId', 'userId', 'timeframe', 'eventdate', 'ordernumber', 'itemId']).to_csv(
        path / 'train-purchases.csv', sep=';', index=False)
    pd.DataFrame({'itemId': range(1, 31), 'categoryId': [i % 5 + 1 for i in range(1, 31)]}).to_csv(
        path / 'product-categories.csv', sep=';', index=False)
    return path


def test_taobao_mapping_period_and_dropped_counts(tmp_path):
    source = taobao_file(tmp_path / 'UserBehavior.csv')
    info = da.taobao_to_raw(source, tmp_path / 'raw')
    raw = pd.read_csv(source, header=None, names=['u', 'i', 'c', 'b', 't'])
    inside = raw.loc[raw.t.between(da.TAOBAO_PERIOD[0], da.TAOBAO_PERIOD[1] - 1)]
    events = pd.read_csv(tmp_path / 'raw' / 'events.csv')
    assert info['dropped']['outside_period'] == 8 and info['dropped']['fav_dropped'] == int((inside.b == 'fav').sum())
    expected = inside.loc[inside.b != 'fav']
    assert len(events) == len(expected) == info['events']
    assert sorted(zip(events.timestamp // 1000, events.visitorid, events.itemid, events.event)) == sorted(
        zip(expected.t, expected.u, expected.i, expected.b.map(da.TAOBAO_EVENTS)))
    assert events.timestamp.is_monotonic_increasing
    assert json.loads((tmp_path / 'raw' / 'adapter.json').read_text(encoding='utf-8'))['files']


def test_taobao_user_sample_is_fixed_by_the_user_id_not_the_file_order(tmp_path):
    source = taobao_file(tmp_path / 'UserBehavior.csv', users=400)
    da.taobao_to_raw(source, tmp_path / 'a', user_percent=30, seed=7)
    shuffled = pd.read_csv(source, header=None).sample(frac=1, random_state=1)
    shuffled.to_csv(tmp_path / 'shuffled.csv', header=False, index=False)
    da.taobao_to_raw(tmp_path / 'shuffled.csv', tmp_path / 'b', user_percent=30, seed=7, chunksize=700)
    a, b = (set(pd.read_csv(tmp_path / d / 'events.csv').visitorid) for d in 'ab')
    assert a == b and 0.15 < len(a) / 400 < 0.45
    da.taobao_to_raw(source, tmp_path / 'c', user_percent=30, seed=8)
    assert set(pd.read_csv(tmp_path / 'c' / 'events.csv').visitorid) != a


def test_diginetica_times_users_and_purchases(tmp_path):
    source = diginetica_dir(tmp_path / 'dg')
    da.diginetica_to_raw(source, tmp_path / 'raw')
    events = pd.read_csv(tmp_path / 'raw' / 'events.csv')
    views = pd.read_csv(source / 'train-item-views.csv', sep=';')
    purchases = pd.read_csv(source / 'train-purchases.csv', sep=';')
    assert (events.event == 'view').sum() == len(views) and (events.event == 'transaction').sum() == len(purchases)
    row = views.iloc[0]
    expected_ts = int(pd.Timestamp(row.eventdate, tz='UTC').value // 1_000_000) + int(row.timeframe)
    expected_user = int(row.userId) if pd.notna(row.userId) else da.SESSION_OFFSET + int(row.sessionId)
    assert ((events.timestamp == expected_ts) & (events.visitorid == expected_user) & (events.itemid == row.itemId)).any()
    bought = events.loc[events.event == 'transaction']
    assert set(bought.transactionid.astype(int)) == set(purchases.ordernumber)
    da.diginetica_to_raw(source, tmp_path / 'signed', anonymous='drop')
    signed = pd.read_csv(tmp_path / 'signed' / 'events.csv')
    assert (signed.visitorid < da.SESSION_OFFSET).all()
    assert len(signed) == views.userId.notna().sum() + purchases.userId.notna().sum()


@pytest.mark.parametrize('kind', ['taobao', 'diginetica'])
def test_adapted_folders_run_through_the_preprocessor_with_no_stock_flag(tmp_path, kind):
    if kind == 'taobao':
        da.taobao_to_raw(taobao_file(tmp_path / 'UserBehavior.csv'), tmp_path / 'raw')
    else:
        da.diginetica_to_raw(diginetica_dir(tmp_path / 'dg'), tmp_path / 'raw')
    _, clean, invalid, properties, _ = load_source_tables(tmp_path / 'raw')
    assert invalid.empty and set(properties.property) == {'categoryid'}
    UnifiedPreprocessor(tmp_path / 'raw', tmp_path / 'prepared', n_estimators=20, use_if=False).run()
    flags = pd.read_csv(tmp_path / 'prepared' / 'feature_train_inliers.csv')
    assert (flags.has_available == 0).all() and (flags.categoryid > 0).all()
