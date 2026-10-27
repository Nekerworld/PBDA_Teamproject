"""Development comparison of the stacked Hybrid (hybrid_ltr.py) on RetailRocket windows that were never a test.

Windows (all inside the v7 F4 training period, which ends at the F4 test cutoff, about 27% of the events):
  D1  train ~0-17%, labels ~17-21.6%: the ranker, the ALS+BPR alpha and the v7 Hybrid fusion grid are fit here.
  D2  train 0-21.6%, labels 21.6-27% (byte-identical to the v7 F4 inner validation split): every model is scored
      here once, with settings fit on D1. D2 labels choose nothing.
Base models use the v7 F4 selected settings (chosen on D2 labels in v7, the same for every model compared), seed 42,
GNN for the fixed selected epochs. No test window is read. This is a development comparison, not a sealed test: the
confirmatory evaluation of whatever design is kept belongs to a new preregistered protocol with an unopened test.

Ranker designs (--design):
  policy_truth  trained on the D1 evaluation users' policy truth (add-to-cart/purchase), dev_results.json;
  weighted      trained on every warm D1 user with a label-period event, behaviour-weighted
                (hybrid_ltr.TrainingWindow), dev_results_weighted.json. Its CV metric is still the D1 policy truth.

Usage: python hybrid_ltr_dev.py [--root audit_runs/v8dev/rr_f4_inner] [--design policy_truth|weighted]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr  # noqa: E402
import rolling_protocol as rp  # noqa: E402
from data_protocol import UnifiedPreprocessor, dump_json, sha256  # noqa: E402
from experiments import clone_protocol  # noqa: E402
from policy_evaluation import POLICIES, EvaluationContext, FactorScores, hybrid_ranking, selection_score  # noqa: E402
from stats_tools import compare_family  # noqa: E402

V7 = HERE / 'audit_runs' / 'v7' / 'F4_unseen'
SELECTION = HERE / 'audit' / 'results_v7' / 'F4_unseen' / 'selection.json'
PLAN = HERE / 'audit' / 'results_v7' / 'plan.json'
RAW = HERE.parent.parent / 'data'
SEED = 42
VARIANTS = {  # stacked Hybrid and its feature-group ablations
    'Hybrid_v2': ltr.FEATURES,
    'v2_mf': ltr.FEATURE_GROUPS['mf'],
    'v2_mf_gnn': ltr.FEATURE_GROUPS['mf'] + ltr.FEATURE_GROUPS['gnn'],
    'v2_no_history': [f for f in ltr.FEATURES if f not in ltr.FEATURE_GROUPS['history']],
    'v2_no_gnn': [f for f in ltr.FEATURES if f not in ltr.FEATURE_GROUPS['gnn']],
    'v2_no_mf': [f for f in ltr.FEATURES if f not in ltr.FEATURE_GROUPS['mf']],
}


def prepare(root):
    cutoff = int(json.loads((V7 / 'prepared' / 'manifest.json').read_text(encoding='utf-8'))['cutoff_timestamp'])
    raw_dir, prepared = root / 'fold_raw', root / 'prepared'
    if not (raw_dir / 'events.csv').exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in ('item_properties_part1.csv', 'item_properties_part2.csv', 'category_tree.csv'):
            os.link(RAW / name, raw_dir / name)
        events = pd.read_csv(RAW / 'events.csv')
        events.loc[events.timestamp < cutoff].to_csv(raw_dir / 'events.csv', index=False)
        dump_json(raw_dir / 'fold_raw.json', {'end_cutoff_timestamp': cutoff, 'rule': 'raw events before the v7 F4 test cutoff'})
    if not (prepared / 'manifest.json').exists():
        UnifiedPreprocessor(data_dir=raw_dir, processed_dir=prepared, test_size=.2, validation_size=.2, use_if=False,
                            core_mode='events', min_user_events=3, min_item_events=3, random_state=42).run()
    # event_id is the row number in the truncated raw file, which here stops at the F4 test cutoff instead of 36%
    def content(path):
        frame = pd.read_csv(path)
        return frame.drop(columns=[c for c in ('event_id',) if c in frame]).fillna(-1)
    same = {name: content(prepared / name).equals(content(V7 / 'prepared' / 'validation' / name))
            for name in ('events_train_clean.csv', 'events_test_original.csv', 'evaluation_users.csv', 'candidate_items.csv',
                         'feature_train_inliers.csv')}
    if not all(same.values()):
        raise RuntimeError(f'D2 differs from the v7 F4 inner validation split: {same}')
    return {'d1': prepared / 'validation', 'd2': prepared}, same


def fit_bases(root, windows, selection, plan):
    out = {}
    for name, source in windows.items():
        mf_dir, gnn_dir = root / f'mf_{name}', root / f'gnn_{name}'
        if not (mf_dir / 'manifest.json').exists():
            clone_protocol(source, mf_dir)
        rp.fit_mf(mf_dir, selection['mf'], SEED)
        backbone = selection['hybrid_backbone']
        spec = rp.gnn_spec(gnn_dir, source, backbone['config'], SEED, 'refit', backbone['epochs'])
        spec['fixed'] = plan['gnn_fixed']
        spec_path = root / f'gnn_{name}_spec.json'
        spec_path.write_text(json.dumps(spec), encoding='utf-8')
        rp.gnn_job(spec_path)
        out[name] = (mf_dir, gnn_dir)
    return out


def load(policy, mf_dir, gnn_dir):
    context = EvaluationContext(mf_dir, policy)
    als, bpr = rp.load_factors(mf_dir, context)
    scores = FactorScores(context, als, bpr, rp.load_gnn(gnn_dir, context))
    als_items = ltr.aligned_item_factors(pd.read_csv(mf_dir / 'als_item_mapping.csv').itemid,
                                         np.load(mf_dir / 'als_item_factors.npy'), context)
    gnn = pd.read_csv(gnn_dir / 'gnn_item_embeddings.csv')
    gnn_items = ltr.aligned_item_factors(gnn.itemid, gnn.drop(columns='itemid').to_numpy(), context)
    return context, scores, als_items, gnn_items


def run_policy(policy, bases, design='policy_truth'):
    c1, s1, a1, g1 = load(policy, *bases['d1'])
    c2, s2, a2, g2 = load(policy, *bases['d2'])
    k = rp.PRIMARY_K
    record = {'populations': {'d1': c1.population(), 'd2': c2.population()}}
    rankings = {'ALS': c2.rank_scores(s2.als, k), 'BPR': c2.rank_scores(s2.bpr, k), 'GNN': c2.rank_scores(s2.gnn, k),
                'Popularity': c2.popularity(k), 'Recent': c2.recent(k), 'RecentPop': c2.recent(k, fill=True)}
    alphas = [{'alpha': a, 'score': selection_score(c1.metrics(c1.rank_scores({u: s1.mf(u, a) for u in c1.active}, k), k))}
              for a in rp.MF_ALPHAS]
    alpha = max(alphas, key=lambda r: r['score'])['alpha']  # first best
    rankings['ALS+BPR'] = c2.rank_scores({u: s2.mf(u, alpha) for u in c2.active}, k)
    start = time.perf_counter()
    grid = rp.hybrid_grid(c1, {u: s1.als[u] for u in c1.active}, {u: s1.bpr[u] for u in c1.active}, s1.gnn, 'backbone')
    best = rp.best_row(grid, ['alpha', 'strategy', 'candidate_k', 'fusion', 'omega', 'gamma'])
    best['candidate_k'] = best['candidate_k'] if str(best['candidate_k']) == 'all' else int(best['candidate_k'])
    rankings['Hybrid_v1'] = hybrid_ranking(s2, best, k)
    record['fit_on_d1'] = {'mf_alpha': alpha, 'hybrid_v1': best, 'hybrid_v1_grid_seconds': time.perf_counter() - start}
    h1, h2 = ltr.History(c1), ltr.History(c2)
    pool1, pool2 = ltr.candidates(c1, s1, h1), ltr.candidates(c2, s2, h2)
    f1, f2 = ltr.features(c1, s1, h1, pool1, a1, g1), ltr.features(c2, s2, h2, pool2, a2, g2)
    y1, w1, t1, cv_context = ltr.labels(c1, f1), None, f1, c1
    if design == 'weighted':
        window = ltr.TrainingWindow(bases['d1'][0], policy)
        if not np.array_equal(window.ids, c1.ids):
            raise RuntimeError('Training window catalog differs from D1')
        als, bpr = rp.load_factors(bases['d1'][0], window)
        sw = FactorScores(window, als, bpr, rp.load_gnn(bases['d1'][1], window))
        t1 = ltr.features(window, sw, h1, ltr.candidates(window, sw, h1), a1, g1)
        y1, w1 = ltr.weighted_labels(window, t1)
        cv_context = window
        record['training_window'] = {'users': len(window.active), 'users_with_policy_truth': len(window.truth),
                                     'rows': len(t1), 'positive_rows': int(y1.sum()),
                                     'positive_rows_by_weight': pd.Series(w1[y1 == 1]).value_counts().sort_index().to_dict()}
    record['pool'] = {name: {'rows': len(f), 'users': int(f.uid.nunique()),
                             'truth_in_pool': float(np.mean([len(c.truth_idx[u] & set(p.get(u, []))) / max(len(c.truth[u]), 1)
                                                             for u in c.users]))}
                      for name, f, c, p in (('d1', f1, c1, pool1), ('d2', f2, c2, pool2))}
    record['ranker'] = {}
    for name, columns in VARIANTS.items():
        params, table = ltr.select_params(cv_context, t1, y1, columns, sample_weight=w1)
        model = ltr.fit(t1, y1, columns, params, sample_weight=w1)
        rankings[name] = ltr.rank(model, f2, columns, k)
        record['ranker'][name] = {'params': params, 'cv': table.to_dict('records'), 'features': columns}
    per_user = {m: c2.details(r, k).set_index('visitorid') for m, r in rankings.items()}
    record['metrics_d2'] = {m: {'ndcg': float(f.ndcg.mean()), 'recall': float(f.recall.mean()), 'users': len(f)}
                            for m, f in per_user.items()}
    competitors = rp.BASELINES + ['Hybrid_v1']
    record['paired_d2'] = {
        'Hybrid_v2 vs baselines and Hybrid_v1 (one Holm family)': compare_family(per_user, 'Hybrid_v2', competitors),
        'Hybrid_v1 vs baselines (one Holm family)': compare_family(per_user, 'Hybrid_v1', rp.BASELINES)}
    return record


def main(root, design='policy_truth'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    plan = json.loads(PLAN.read_text(encoding='utf-8'))['plan']
    windows, same = prepare(root)
    bases = fit_bases(root, windows, selection, plan)
    result = {'design': design, 'started_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'd2_equals_v7_inner_validation': same,
              'windows': {name: json.loads((path / 'manifest.json').read_text(encoding='utf-8'))['cutoff_utc']
                          for name, path in windows.items()},
              'base_settings': {'mf': selection['mf'], 'gnn': selection['hybrid_backbone']['config'],
                                'gnn_epochs': selection['hybrid_backbone']['epochs'], 'seed': SEED},
              'pool': ltr.POOL, 'param_grid': ltr.PARAM_GRID, 'policies': {}}
    for policy in POLICIES:
        result['policies'][policy] = run_policy(policy, bases, design)
        summary = pd.DataFrame(result['policies'][policy]['metrics_d2']).T
        print(policy, '\n', summary.round(4).to_string(), flush=True)
    result['finished_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    dump_json(root / ('dev_results.json' if design == 'policy_truth' else f'dev_results_{design}.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE / 'audit_runs' / 'v8dev' / 'rr_f4_inner')
    parser.add_argument('--design', choices=('policy_truth', 'weighted'), default='policy_truth')
    args = parser.parse_args()
    main(args.root, args.design)
