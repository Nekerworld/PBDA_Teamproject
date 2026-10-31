"""Sensitivity analysis declared before the v9 test was opened (audit/v9_dev_plan.md section 6).

The protocol selects the trained ALS, BPR and GNN configurations on repeat validation, while the ladder rankers are
trained per policy. For the novel policy this favours the ladder. Here the novel baselines are rebuilt with every
choice made on novel validation, from the protocol's own validation runs (no new grid): ALS, BPR (configuration and
epoch) and GNN by the novel validation score of `validation_mf_grid.csv` / `validation_gnn.json`, ALS+BPR alpha on
novel validation with those fits. They are refitted on the full training period with the protocol seeds and compared
with the sealed ladder rankers' test lists (unchanged). The test opening is appended to the root's access log
(same plan and selection) before any test label is read.

Usage: python audit/v9protocol/novel_tuned_baselines.py audit_runs/v9_taobao"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402  (forces one BLAS thread before numpy)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import hybrid_v9 as v9  # noqa: E402
from data_protocol import dump_json, sha256  # noqa: E402
from experiments import clone_protocol  # noqa: E402
from policy_evaluation import EvaluationContext, FactorScores, selection_score  # noqa: E402
from stats_tools import compare_pairs  # noqa: E402

POLICY = 'novel'
TUNED = {'ALS': 'ALS_n', 'BPR': 'BPR_n', 'ALS+BPR': 'ALS+BPR_n', 'GNN': 'GNN_n'}
PAIRS = [('Hybrid', 'ALS_n'), ('Hybrid', 'BPR_n'), ('Hybrid', 'ALS+BPR_n'), ('Hybrid', 'GNN_n'),
         ('Hybrid_MF_Graph', 'ALS+BPR_n'), ('ALS+BPR_n', 'ALS_n'), ('ALS+BPR_n', 'BPR_n'), ('ALS+BPR_n', 'GNN_n')]


def clone_with_retry(source, destination, attempts=5):
    """Windows can refuse the final folder rename right after the hard links are made (WinError 5, I-151/I-157);
    clone_protocol removes a left-over .partial folder, so a retry starts clean."""
    import time
    for attempt in range(attempts):
        if (Path(destination) / 'manifest.json').exists():
            return
        try:
            clone_protocol(source, destination)
            return
        except PermissionError as error:
            print(f'clone {destination}: {error}; retry {attempt + 1}/{attempts}', flush=True)
            time.sleep(10)
    raise RuntimeError(f'Could not clone {source} to {destination}')


def first_best(frame, score):
    return frame.assign(_s=score).sort_values('_s', ascending=False, kind='stable').iloc[0]


def choices(fold_dir, pick_by):
    grid = pd.read_csv(fold_dir / 'validation_mf_grid.csv')
    score = lambda part: (part[f'{pick_by}_ndcg'] + part[f'{pick_by}_recall']) / 2
    als = first_best(grid.loc[grid.family == 'ALS'], score(grid.loc[grid.family == 'ALS']))
    bpr = first_best(grid.loc[grid.family == 'BPR'], score(grid.loc[grid.family == 'BPR']))
    gnn_rows = json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8'))
    gnn = sorted(gnn_rows, key=lambda r: -selection_score(r[pick_by]))[0]  # stable: first best
    return {'als': {'factors': int(als.factors), 'regularization': float(als.regularization), 'alpha': float(als.alpha),
                    'iterations': int(als.iterations)},
            'bpr': {'factors': int(bpr.factors), 'learning_rate': float(bpr.learning_rate),
                    'regularization': float(bpr.regularization)},
            'bpr_epochs': int(bpr.epochs),
            'gnn': {k: gnn[k] for k in ('name', 'config', 'epochs', 'at_cap', 'job_dir')} | {'extended': bool(gnn.get('extended'))},
            'validation': {'ALS': float(score(als.to_frame().T).iloc[0]), 'BPR': float(score(bpr.to_frame().T).iloc[0]),
                           'GNN': selection_score(gnn[pick_by])}}


def run(root, pick_by=POLICY, out_name='sensitivity_novel'):
    root = Path(root)
    fold = rp.FOLDS[0]
    fold_dir = root / fold['name']
    stored = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    plan_hash = stored['plan_sha256']
    if hashlib.sha256(json.dumps(rp.plan_payload(), sort_keys=True).encode()).hexdigest() != plan_hash:
        raise RuntimeError('The code plan differs from the root plan')
    selection = rp.sealed_selection(fold_dir, plan_hash)
    access = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()] \
        if (root / 'test_access_log.jsonl').exists() else []
    if not any(a['fold'] == fold['name'] and a['purpose'] == 'sealed_test' for a in access) \
            or not (fold_dir / 'test_metrics.csv').exists():
        raise RuntimeError('The sealed test evaluation must be complete before this analysis')
    out = fold_dir / out_name
    out.mkdir(exist_ok=True)
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text(encoding='utf-8'))
    chosen = choices(fold_dir, pick_by)
    mf_choice = {k: chosen[k] for k in ('als', 'bpr', 'bpr_epochs')}

    # alpha on novel validation with the novel-chosen fits (validation labels only)
    val_mf = out / 'val_mf'
    clone_with_retry(fold_dir / 'prepared' / 'validation', val_mf)
    rp.require_inner_validation(fold_dir, val_mf, 'The MF fit used for the novel alpha')
    rp.fit_mf(val_mf, mf_choice, rp.SELECTION_SEED)
    ctx = EvaluationContext(val_mf, POLICY)
    als_v, bpr_v = rp.load_factors(val_mf, ctx)
    fs = FactorScores(ctx, als_v, bpr_v)
    alpha_rows = []
    for alpha in rp.MF_ALPHAS:
        m = ctx.metrics(ctx.rank_scores({u: fs.mf(u, alpha) for u in ctx.active}, rp.PRIMARY_K), rp.PRIMARY_K)
        alpha_rows.append({'alpha': alpha, **m, 'score': selection_score(m)})
    pd.DataFrame(alpha_rows).to_csv(out / 'validation_mf_alpha_novel.csv', index=False)
    alpha = rp.best_row(alpha_rows, ['alpha'])['alpha']
    del als_v, bpr_v, fs

    # refits on the full training period with the protocol seeds
    refits = json.loads((fold_dir / 'refits.json').read_text(encoding='utf-8'))
    same_gnn = chosen['gnn']['config'] == selection['gnn']['config'] and chosen['gnn']['epochs'] == selection['gnn']['epochs']
    dirs = {}
    for seed in rp.SEEDS:
        seed_dir = out / f'seed_{seed}'
        clone_with_retry(fold_dir / 'prepared', seed_dir)
        rp.fit_mf(seed_dir, mf_choice, seed)
        if same_gnn:
            gnn_dir = root / refits[str(seed)]['gnn']
        else:
            gnn_dir = out / f'gnn_seed_{seed}'
            spec = rp.gnn_spec(gnn_dir, fold_dir / 'prepared', chosen['gnn']['config'], seed, 'refit', chosen['gnn']['epochs'])
            spec_path = out / f'gnn_seed_{seed}_spec.json'
            spec_path.write_text(json.dumps(spec), encoding='utf-8')
            clone_with_retry(fold_dir / 'prepared', gnn_dir)
            rp.gnn_job(spec_path)
        dirs[seed] = (seed_dir, gnn_dir)

    rp.access_log(root, {'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'fold': fold['name'],
                         'purpose': f'sensitivity_{pick_by}_tuned_baselines', 'plan_sha256': plan_hash,
                         'selection_sha256': sha256(fold_dir / 'selection.json'), 'git': rp.git_state()['commit']})
    rows, details = [], {}
    for seed, (seed_dir, gnn_dir) in dirs.items():
        context = EvaluationContext(seed_dir, POLICY)
        als, bpr = rp.load_factors(seed_dir, context)
        std = FactorScores(context, als, bpr, rp.load_gnn(gnn_dir, context))
        k = max(rp.KS)
        rankings = {'ALS_n': context.rank_scores(std.als, k), 'BPR_n': context.rank_scores(std.bpr, k),
                    'ALS+BPR_n': context.rank_scores({u: std.mf(u, alpha) for u in context.active}, k),
                    'GNN_n': context.rank_scores(std.gnn, k)}
        for model, ranked in rankings.items():
            for kk in rp.KS:
                frame = context.details(ranked, kk)
                rows.append({'policy': POLICY, 'seed': seed, 'model': model, 'k': kk, 'users': len(frame),
                             **{m: float(frame[m].mean()) for m in ('ndcg', 'recall', 'precision', 'hit_rate')}})
                if kk == rp.PRIMARY_K:
                    details.setdefault(model, []).append(frame)
        protocol_dir = root / refits[str(seed)]['mf']
        for model in ['Hybrid', 'Hybrid_MF_Graph', *TUNED, 'Popularity', 'Recent', 'RecentPop', 'CoVisit']:
            frame = pd.read_csv(protocol_dir / f'per_user_{POLICY}_{model.replace("+", "_")}_k50.csv')
            details.setdefault(model, []).append(frame)
    users = {m: sorted(set(f.visitorid)) for m, frames in details.items() for f in frames[:1]}
    if len({tuple(u) for u in users.values()}) != 1:
        raise RuntimeError('Evaluation users differ between the protocol lists and the rebuilt baselines')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'test_metrics.csv', index=False)
    per_user = {m: pd.concat(f).groupby('visitorid')[['ndcg', 'recall', 'precision', 'hit_rate']].mean()
                for m, f in details.items()}
    tests = pd.DataFrame(compare_pairs(per_user, PAIRS))
    tests.to_csv(out / 'paired_tests.csv', index=False)
    ndcg = {m: float(u.ndcg.mean()) for m, u in per_user.items()}
    tuned = {**ndcg, **{base: ndcg[new] for base, new in TUNED.items()}}
    baselines = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['plan']['baselines']
    result = {'policy': POLICY, 'picked_by': pick_by, 'choices': chosen, 'mf_alpha': alpha,
              'gnn_reused_from_protocol': same_gnn, 'seed_mean_ndcg50': ndcg,
              'ladder_with_tuned_baselines': v9.ladder_summary(tuned, baselines),
              'hybrid_above_every_baseline_of_both_tunings': bool(
                  ndcg['Hybrid'] > max(ndcg[m] for m in [*baselines, *TUNED.values()])),
              'paired': tests.to_dict('records')}
    dump_json(out / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--protocol', default='v9')
    args = parser.parse_args()
    rp.use_protocol(args.protocol)
    result = run(args.root)
    print(json.dumps({k: v for k, v in result.items() if k != 'paired'}, indent=1, default=str))
    for row in result['paired']:
        print(f"{row['comparison']:28s} {row['metric']:6s} {row['difference']:+.5f} [{row['ci_low']:+.5f}, "
              f"{row['ci_high']:+.5f}] Holm {row['p_holm']:.4f} {row['claim']}")
