"""v9 development windows and base models on the Taobao development sample (audit/v9_dev_plan.md).

Sample: Taobao UserBehavior users in the seed-42 hash band [100, 300) (dataset_adapters.taobao_to_raw(user_band=...)),
disjoint from the v8 test sample [0, 100) and the v9 confirmation sample [300, 400).
Windows (v8 fold ratios):
  full  the sample preprocessed like v8 (test 0.25, validation 0.2). Its outer test is the development holdout H.
  d2    raw events before the H cutoff, preprocessed with test 0.2: the same split as the inner validation of full
        (checked). Development evaluation window, read repeatedly.
  d1    inner validation of d2. The upper modules (re-rankers, fusion weights) are fit here.
Base models (ALS, BPR, GNN) use the v8-selected settings, seed 42, the GNN for the v8 selected epochs. They are never
re-tuned on the development windows.

Ladder (``ladder``): every model of the plan's ladder on d2 (rankers and the ALS+BPR alpha fit on d1), the ratios of
the plan's criterion, and one row per policy appended to audit/v9dev/attempts.csv.

Usage: python hybrid_v9_dev.py prepare
       python hybrid_v9_dev.py bases --window d1|d2|h
       python hybrid_v9_dev.py ladder --attempt <name> --note <text> [--design weighted|policy_truth]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr  # noqa: E402
import hybrid_v9 as v9  # noqa: E402
import rolling_protocol as rp  # noqa: E402
from data_protocol import UnifiedPreprocessor, dump_json  # noqa: E402
from experiments import clone_protocol  # noqa: E402
from policy_evaluation import POLICIES, EvaluationContext, FactorScores, selection_score  # noqa: E402
from stats_tools import compare_family  # noqa: E402

ROOT = HERE / 'audit_runs' / 'v9dev'
RAW = ROOT / 'raw_b100_300'
SELECTION = HERE / 'audit' / 'results_v8' / 'TB1_full' / 'selection.json'
PLAN = HERE / 'audit' / 'results_v8' / 'plan.json'
SEED = 42
PREPROCESS = dict(use_if=False, core_mode='events', min_user_events=3, min_item_events=3, random_state=42)


def windows(root=ROOT):
    return {'d1': root / 'dev' / 'prepared' / 'validation', 'd2': root / 'dev' / 'prepared',
            'h': root / 'full' / 'prepared'}


def prepare(root=ROOT, raw=RAW):
    full = root / 'full' / 'prepared'
    if not (full / 'manifest.json').exists():
        UnifiedPreprocessor(data_dir=raw, processed_dir=full, test_size=.25, validation_size=.2, **PREPROCESS).run()
    cutoff = int(json.loads((full / 'manifest.json').read_text(encoding='utf-8'))['cutoff_timestamp'])
    raw_dir, dev = root / 'dev' / 'fold_raw', root / 'dev' / 'prepared'
    if not (raw_dir / 'events.csv').exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in ('item_properties_part1.csv', 'item_properties_part2.csv', 'category_tree.csv'):
            os.link(raw / name, raw_dir / name)
        events = pd.read_csv(raw / 'events.csv')
        events.loc[events.timestamp < cutoff].to_csv(raw_dir / 'events.csv', index=False)
        dump_json(raw_dir / 'fold_raw.json', {'end_cutoff_timestamp': cutoff, 'rule': 'raw events before the H cutoff'})
    if not (dev / 'manifest.json').exists():
        UnifiedPreprocessor(data_dir=raw_dir, processed_dir=dev, test_size=.2, validation_size=.2, **PREPROCESS).run()

    def content(path):
        frame = pd.read_csv(path)
        return frame.drop(columns=[c for c in ('event_id',) if c in frame]).fillna(-1)
    same = {name: content(dev / name).equals(content(full / 'validation' / name))
            for name in ('events_train_clean.csv', 'events_test_original.csv', 'evaluation_users.csv',
                         'candidate_items.csv', 'feature_train_inliers.csv')}
    if not all(same.values()):
        raise RuntimeError(f'd2 differs from the inner validation split of full: {same}')
    record = {'h_cutoff_timestamp': cutoff, 'd2_equals_full_inner_validation': same,
              'cutoffs_utc': {name: json.loads((path / 'manifest.json').read_text(encoding='utf-8'))['cutoff_utc']
                              for name, path in windows(root).items()}}
    dump_json(root / 'windows.json', record)
    return record


def fit_bases(window, root=ROOT):
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    plan = json.loads(PLAN.read_text(encoding='utf-8'))['plan']
    source = windows(root)[window]
    mf_dir, gnn_dir = root / f'mf_{window}', root / f'gnn_{window}'
    if not (mf_dir / 'manifest.json').exists():
        clone_protocol(source, mf_dir)
    rp.fit_mf(mf_dir, selection['mf'], SEED)
    backbone = selection['hybrid_backbone']
    spec = rp.gnn_spec(gnn_dir, source, backbone['config'], SEED, 'refit', backbone['epochs'])
    spec['fixed'] = plan['gnn_fixed']
    spec_path = root / f'gnn_{window}_spec.json'
    spec_path.write_text(json.dumps(spec), encoding='utf-8')
    rp.gnn_job(spec_path)
    return mf_dir, gnn_dir


def bases(root=ROOT):
    return {w: (root / f'mf_{w}', root / f'gnn_{w}') for w in windows(root)}


# ----------------------------------------------------------------------------- ladder
# name -> modules the ranker pools and reads. L2 and L3 are the ladder of the plan; the rest are descriptive.
RANKERS = {'Hybrid_MF_Graph': ('mf', 'graph'), 'Hybrid': ('mf', 'graph', 'context'),
           'LTR_MF': ('mf',), 'Hybrid_no_graph': ('mf', 'context'), 'Hybrid_no_mf': ('graph', 'context')}
LADDER = {'L1': 'ALS+BPR', 'L2': 'Hybrid_MF_Graph', 'L3': 'Hybrid'}


def load(policy, mf_dir, gnn_dir):
    context = EvaluationContext(mf_dir, policy)
    als, bpr = rp.load_factors(mf_dir, context)
    scores = FactorScores(context, als, bpr, rp.load_gnn(gnn_dir, context))
    return context, scores, ltr.item_tables(context, mf_dir, gnn_dir)


def feature_table(context, scores, tables, cache, pool=v9.POOL):
    if cache.exists():
        return pd.read_pickle(cache)
    signals = v9.Signals(context)
    frame = v9.features(context, scores, signals, v9.candidates(context, scores, signals, pool), *tables)
    frame.to_pickle(cache)
    return frame


def ladder_policy(policy, train_window, eval_window, cache_dir, design, objective='binary', variant='a03'):
    pool, drop = v9.DESIGNS[variant]['pool'], v9.DESIGNS[variant]['drop']
    k = rp.PRIMARY_K
    (mf1, gnn1), (mf2, gnn2) = bases()[train_window], bases()[eval_window]
    c1, s1, t1 = load(policy, mf1, gnn1)
    c2, s2, t2 = load(policy, mf2, gnn2)
    record = {'populations': {train_window: c1.population(), eval_window: c2.population()}}
    rankings = {'ALS': c2.rank_scores(s2.als, k), 'BPR': c2.rank_scores(s2.bpr, k), 'GNN': c2.rank_scores(s2.gnn, k),
                'Popularity': c2.popularity(k), 'Recent': c2.recent(k), 'RecentPop': c2.recent(k, fill=True)}
    alphas = [{'alpha': a, 'score': selection_score(c1.metrics(c1.rank_scores({u: s1.mf(u, a) for u in c1.active}, k), k))}
              for a in rp.MF_ALPHAS]
    alpha = max(alphas, key=lambda r: r['score'])['alpha']  # first best
    rankings['ALS+BPR'] = c2.rank_scores({u: s2.mf(u, alpha) for u in c2.active}, k)
    record['mf_alpha'] = alpha
    signals2 = v9.Signals(c2)
    rankings['CoVisit'] = v9.covisit_ranking(c2, signals2, k)
    f2 = feature_table(c2, s2, t2, cache_dir / f'{eval_window}_{policy}.pkl', pool)
    if design == 'weighted':
        window = ltr.TrainingWindow(mf1, policy)
        sw = FactorScores(window, *rp.load_factors(mf1, window), rp.load_gnn(gnn1, window))
        f1 = feature_table(window, sw, t1, cache_dir / f'{train_window}_{policy}_window.pkl', pool)
        y1, w1 = ltr.weighted_labels(window, f1)
        cv_context = window
    else:
        f1 = feature_table(c1, s1, t1, cache_dir / f'{train_window}_{policy}.pkl', pool)
        y1, w1, cv_context = ltr.labels(c1, f1), None, c1
    record['pool'] = {}
    for name, frame, ctx in ((train_window, f1, cv_context), (eval_window, f2, c2)):
        record['pool'][name] = {'rows': len(frame), 'users': int(frame.uid.nunique())}
        for modules in [(m,) for m in v9.MODULES] + [('mf', 'graph'), v9.MODULES]:
            part = frame.loc[v9.module_rows(frame, modules)]
            pooled = part.groupby('uid').idx.agg(set).to_dict()
            record['pool'][name]['truth_in_pool_' + '+'.join(modules)] = float(np.mean(
                [len(ctx.truth_idx[u] & pooled.get(u, set())) / max(len(ctx.truth[u]), 1) for u in ctx.truth]))
    record['ranker'] = {}
    for name, modules in RANKERS.items():
        start = time.perf_counter()
        keep1, keep2 = v9.module_rows(f1, modules), v9.module_rows(f2, modules)
        columns = v9.module_features(modules, drop)
        train = f1.loc[keep1].reset_index(drop=True)
        weight = None if w1 is None else w1[keep1]
        params, table, sampled = v9.select_params(cv_context, train, y1[keep1], columns, sample_weight=weight,
                                                  objective=objective)
        model = v9.fit(train.loc[sampled].reset_index(drop=True), y1[keep1][sampled], columns, params, objective,
                       sample_weight=None if weight is None else weight[sampled])
        rankings[name] = v9.rank(model, f2.loc[keep2], columns, k)
        record['ranker'][name] = {'modules': list(modules), 'params': params, 'cv': table.to_dict('records'),
                                  'train_rows': int(keep1.sum()), 'train_rows_sampled': int(sampled.sum()),
                                  'seconds': time.perf_counter() - start}
        print(policy, name, params, f'{time.perf_counter() - start:.0f}s', flush=True)
    per_user = {m: c2.details(r, k).set_index('visitorid') for m, r in rankings.items()}
    record['metrics'] = {m: {'ndcg': float(f.ndcg.mean()), 'recall': float(f.recall.mean()), 'users': len(f)}
                         for m, f in per_user.items()}
    record['paired'] = {
        'Hybrid vs every other model (one Holm family)': compare_family(per_user, 'Hybrid', [m for m in per_user if m != 'Hybrid']),
        'Hybrid_MF_Graph vs ALS+BPR': compare_family(per_user, 'Hybrid_MF_Graph', ['ALS+BPR'])}
    record['ladder'] = ladder_summary(record['metrics'])
    return record


def ladder_summary(metrics):
    ndcg = {m: v['ndcg'] for m, v in metrics.items()}
    l0 = max(ndcg[m] for m in ('ALS', 'BPR', 'GNN'))
    l1, l2, l3 = (ndcg[LADDER[s]] for s in ('L1', 'L2', 'L3'))
    baselines = max(ndcg[m] for m in rp.BASELINES + ['CoVisit'])
    return {'L0_max': l0, 'L1': l1, 'L2': l2, 'L3': l3, 'L1/L0': l1 / l0, 'L2/L1': l2 / l1, 'L3/L2': l3 / l2,
            'min_step': min(l2 / l1, l3 / l2), 'best_baseline': baselines,
            'pass': bool(l1 > l0 and l2 >= 1.1 * l1 and l3 >= 1.1 * l2 and l3 > baselines)}


def log_attempt(attempt, note, design, eval_window, result, root=ROOT):
    design = f"{design}/{result.get('objective', 'binary')}"
    path = HERE / 'audit' / 'v9dev' / 'attempts.csv'
    rows = []
    for policy, record in result['policies'].items():
        rows.append({'attempt': attempt, 'utc': result['finished_utc'], 'window': eval_window, 'design': design,
                     'policy': policy, **{k: v for k, v in record['ladder'].items()},
                     'git': result['git_at_start'], 'note': note})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode='a', header=not path.exists(), index=False)


def ladder(attempt, note, design='weighted', train_window='d1', eval_window='d2', root=ROOT, features_from=None,
           objective='binary', variant='a03'):
    """``features_from``: reuse the feature tables of an earlier attempt (only when the feature code is unchanged)."""
    if eval_window == 'h' and train_window != 'd2':
        raise ValueError('H is evaluated with rankers trained on d2')
    if eval_window == 'h':  # every look at the holdout is logged before its labels are read
        looks = HERE / 'audit' / 'v9dev' / 'h_looks.csv'
        pd.DataFrame([{'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'attempt': attempt, 'note': note,
                       'git': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                             cwd=HERE).stdout.strip()}]).to_csv(looks, mode='a', header=not looks.exists(),
                                                                                 index=False)
    cache_dir = root / 'attempts' / (features_from or attempt)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {'attempt': attempt, 'note': note, 'design': design, 'objective': objective, 'variant': variant,
              'train_window': train_window,
              'eval_window': eval_window, 'feature_cache': str(cache_dir.relative_to(root)),
              'started_utc': pd.Timestamp.now(tz='UTC').isoformat(),
              'git_at_start': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                             cwd=HERE).stdout.strip(),
              'pool': v9.DESIGNS[variant]['pool'], 'dropped_features': v9.DESIGNS[variant]['drop'],
              'features': v9.FEATURE_GROUPS, 'param_grid': ltr.PARAM_GRID, 'policies': {}}
    for policy in POLICIES:
        result['policies'][policy] = ladder_policy(policy, train_window, eval_window, cache_dir, design, objective,
                                                   variant)
        print(policy, json.dumps(result['policies'][policy]['ladder'], indent=1), flush=True)
        print(pd.DataFrame(result['policies'][policy]['metrics']).T.round(4).to_string(), flush=True)
    result['finished_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    out = HERE / 'audit' / 'v9dev' / f'ladder_{attempt}_{eval_window}.json'
    dump_json(out, result)
    log_attempt(attempt, note, design, eval_window, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('prepare', 'bases', 'ladder'))
    parser.add_argument('--window', choices=('d1', 'd2', 'h'))
    parser.add_argument('--attempt')
    parser.add_argument('--note', default='')
    parser.add_argument('--design', choices=('weighted', 'policy_truth'), default='weighted')
    parser.add_argument('--features-from')
    parser.add_argument('--objective', choices=('binary', 'lambdarank'), default='binary')
    parser.add_argument('--variant', choices=sorted(v9.DESIGNS), default='a03')
    parser.add_argument('--eval-window', choices=('d2', 'h'), default='d2',
                        help='h: the development holdout, with rankers and the alpha fit on d2 (logged look)')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.command == 'bases':
        fit_bases(args.window)
        print('done', args.window)
    else:
        ladder(args.attempt, args.note, args.design, 'd2' if args.eval_window == 'h' else 'd1', args.eval_window,
               features_from=args.features_from, objective=args.objective, variant=args.variant)
