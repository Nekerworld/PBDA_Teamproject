"""Independent integrity check of the committed v6/v7 results (written without the pipeline's selection, split or
statistics code; it imports nothing from rolling_protocol).

L  test access: plan hash recomputed from plan.json, one sealed opening per fold under that plan, the opened selection
   is byte-identical to the committed selection.json, the selection was locked before the opening, post-hoc openings
   only after the sealed one, the logged git commit exists and predates the opening.
S  selection: every selected setting (ALS, BPR, epochs, GNN, ALS+BPR alpha, repeat Hybrid backbone, per-policy Hybrid
   parameters) re-derived from the committed inner-validation tables alone with the preregistered rule
   (max mean(NDCG@50, Recall@50), first row on ties); validation tables score the validation population, not the test one.
T  time boundaries (v7, local data): cutoffs re-derived from raw events.csv; every file a selection fit or a test fit
   trained on, and the item properties (stock signal) it saw, end before the outer test window; selection data end
   before the validation labels; train/test event ids disjoint; selection and refit folders hold the same training
   bytes as the split they claim.
P  statistics (v7): per-user K=50 files averaged over seeds re-derive each paired difference and the better/worse/tied
   counts; Holm re-derived from the stored p-values; sign-flip p-values re-estimated with an independent RNG.
Usage: python audit/cycle23/pre_loop_integrity.py <output.json>
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
RAW = HERE.parent.parent / 'data'
checks, unavailable = [], []


def check(group, name, ok, **detail):
    checks.append({'group': group, 'check': name, 'pass': bool(ok), **detail})
    print(('PASS ' if ok else 'FAIL ') + group + ' ' + name + ' ' + json.dumps(detail, default=str)[:300], flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc(text):
    return pd.Timestamp(text).tz_convert('UTC') if pd.Timestamp(text).tzinfo else pd.Timestamp(text, tz='UTC')


# ----------------------------------------------------------------------------- L: test access
for version in ('v6', 'v7'):
    root = HERE / 'audit' / f'results_{version}'
    stored = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    plan_hash = hashlib.sha256(json.dumps(stored['plan'], sort_keys=True).encode()).hexdigest()
    check('L', f'{version} plan hash recomputed', plan_hash == stored['plan_sha256'], hash=plan_hash[:12])
    log = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    check('L', f'{version} every opening under the plan', all(e['plan_sha256'] == plan_hash for e in log), openings=len(log))
    for fold in [f['name'] for f in stored['plan']['folds']]:
        entries = [e for e in log if e['fold'] == fold]
        sealed = [e for e in entries if e['purpose'] == 'sealed_test']
        check('L', f'{version} {fold} one sealed opening', len(sealed) == 1, purposes=[e['purpose'] for e in entries])
        if len(sealed) != 1:
            continue
        opened = utc(sealed[0]['utc'])
        selection_path = root / fold / 'selection.json'
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        check('L', f'{version} {fold} opened selection is the committed one',
              all(e['selection_sha256'] == sha(selection_path) for e in entries))
        check('L', f'{version} {fold} selection sealed under the plan, outer test unused',
              selection['plan_sha256'] == plan_hash and selection.get('outer_test_used_for_selection') is False)
        check('L', f'{version} {fold} selection locked before the opening', utc(selection['locked_utc']) < opened,
              locked=selection['locked_utc'], opened=sealed[0]['utc'])
        check('L', f'{version} {fold} post-hoc openings after the sealed one',
              all(utc(e['utc']) > opened for e in entries if e['purpose'] != 'sealed_test'))
        commit_time = subprocess.run(['git', 'show', '-s', '--format=%cI', sealed[0]['git']], cwd=HERE,
                                     capture_output=True, encoding='utf-8')
        check('L', f'{version} {fold} logged commit exists and predates the opening',
              commit_time.returncode == 0 and utc(commit_time.stdout.strip()) <= opened,
              commit=sealed[0]['git'][:7], commit_time=commit_time.stdout.strip())
    check('L', f'{version} plan created before every opening', all(utc(stored['created_utc']) < utc(e['utc']) for e in log))


# ----------------------------------------------------------------------------- S: selection from validation tables
def first_best(frame):
    frame = frame.assign(_score=(frame.ndcg + frame.recall) / 2)
    return frame.loc[frame._score.idxmax()] if frame._score.notna().all() else None


for version, folds in (('v6', ('F1_hist', 'F2_mid', 'F3_early')), ('v7', ('F4_unseen',))):
    plan = json.loads((HERE / 'audit' / f'results_{version}' / 'plan.json').read_text(encoding='utf-8'))['plan']
    backbones_used = int(plan['hybrid_backbones'].split()[1])
    for fold in folds:
        d = HERE / 'audit' / f'results_{version}' / fold
        sel = json.loads((d / 'selection.json').read_text(encoding='utf-8'))
        grid = pd.read_csv(d / 'validation_mf_grid.csv')
        for family, keys in (('ALS', ('factors', 'regularization', 'alpha', 'iterations')),
                             ('BPR', ('factors', 'learning_rate', 'regularization'))):
            part = grid.loc[grid.family == family].rename(columns={'repeat_ndcg': 'ndcg', 'repeat_recall': 'recall'})
            best = first_best(part.reset_index(drop=True))
            wanted = sel['mf'][family.lower()]
            same = all(np.isclose(float(best[k]), float(wanted[k])) for k in keys)
            if family == 'BPR':
                same = same and int(best['epochs']) == int(sel['mf']['bpr_epochs'])
            check('S', f'{version} {fold} {family} = validation argmax', same, rows=len(part))
        gnn = json.loads((d / 'validation_gnn.json').read_text(encoding='utf-8'))
        scores = [(r['repeat']['ndcg'] + r['repeat']['recall']) / 2 for r in gnn]
        check('S', f'{version} {fold} GNN = validation argmax',
              all(np.isclose(s, r['score']) for s, r in zip(scores, gnn)) and scores == sorted(scores, reverse=True)
              and sel['gnn']['name'] == gnn[0]['name'], chosen=gnn[0]['name'])
        population = {p['policy']: p for p in json.loads((d / 'validation_population.json').read_text(encoding='utf-8'))}
        tables = {}
        for policy in ('repeat', 'novel'):
            alpha = pd.read_csv(d / f'validation_mf_alpha_{policy}.csv')
            check('S', f'{version} {fold} {policy} ALS+BPR alpha = validation argmax',
                  np.isclose(float(first_best(alpha)['alpha']), float(sel['policies'][policy]['mf_alpha'])))
            paths = [d / f'validation_hybrid_{policy}_{row["name"]}.csv' for row in gnn[:backbones_used]]
            if all(p.exists() for p in paths):
                tables[policy] = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            users = set(alpha.users) | (set(tables[policy].users) if policy in tables else set())
            check('S', f'{version} {fold} {policy} tables score the validation population',
                  users == {population[policy]['evaluation_users']}, users=sorted(users),
                  validation=population[policy]['evaluation_users'])
        test_population = pd.read_csv(d / 'report' / 'populations.csv').set_index('policy')
        check('S', f'{version} {fold} validation population is not the test population',
              all(population[p]['evaluation_users'] != test_population.loc[p, 'evaluation_users']
                  or population[p]['catalog_items'] != test_population.loc[p, 'catalog_items'] for p in population),
              validation=[population[p]['evaluation_users'] for p in ('repeat', 'novel')],
              test=test_population.evaluation_users.tolist())
        if len(tables) < 2:
            unavailable.append(f'{version} {fold} Hybrid grid tables not committed (kept on the run PC): '
                               'Hybrid backbone and parameters not re-derived')
            continue
        repeat_backbone = first_best(tables['repeat'])['backbone']
        check('S', f'{version} {fold} Hybrid backbone = repeat validation argmax',
              sel['hybrid_backbone']['name'] == repeat_backbone, backbone=repeat_backbone)
        for policy in ('repeat', 'novel'):
            best = first_best(tables[policy].loc[tables[policy].backbone == repeat_backbone].reset_index(drop=True))
            chosen = sel['policies'][policy]['hybrid']
            same = all(str(best[k]) == str(chosen[k]) if k in ('backbone', 'strategy', 'fusion', 'candidate_k')
                       else np.isclose(float(best[k]), float(chosen[k])) for k in
                       ('backbone', 'alpha', 'strategy', 'candidate_k', 'fusion', 'omega', 'gamma'))
            check('S', f'{version} {fold} {policy} Hybrid parameters = validation argmax', same,
                  rows=int((tables[policy].backbone == repeat_backbone).sum()))


# ----------------------------------------------------------------------------- T: time boundaries (v7 local)
def tie_safe(frame, share):
    ordered = frame.sort_values(['timestamp', 'order'], kind='stable')
    index = min(max(int(len(ordered) * (1 - share)), 1), len(ordered) - 1)
    return int(ordered.timestamp.iloc[index])


fold_dir = HERE / 'audit_runs' / 'v7' / 'F4_unseen'
spec = json.loads((HERE / 'audit' / 'results_v7' / 'plan.json').read_text(encoding='utf-8'))['plan']['folds'][0]
if fold_dir.exists() and (RAW / 'events.csv').exists():
    raw = pd.read_csv(RAW / 'events.csv')
    raw['order'] = np.arange(len(raw))
    fold_end = tie_safe(raw, 1 - spec['end_fraction'])
    fold = raw.loc[raw.timestamp < fold_end]
    outer = tie_safe(fold, spec['test_size'])
    inner = tie_safe(fold.loc[fold.timestamp < outer], spec['validation_size'])
    check('T', 'v7 cutoffs from raw', fold_end > outer > inner, fold_end=fold_end, outer_test_start=outer,
          validation_label_start=inner)

    def span(path):
        frame = pd.read_csv(path, usecols=lambda c: c in ('timestamp', 'event_id'))
        return frame, int(frame.timestamp.min()), int(frame.timestamp.max())

    selection_dirs = [fold_dir / 'prepared' / 'validation', fold_dir / 'val_mf'] + sorted(
        p for p in (fold_dir / 'gnn_val').iterdir() if p.is_dir())
    test_dirs = [fold_dir / 'prepared'] + sorted(p for p in (fold_dir / 'runs').glob('seed_*') if p.is_dir())
    reference = {'selection': sha(fold_dir / 'prepared' / 'validation' / 'events_train_clean.csv'),
                 'test': sha(fold_dir / 'prepared' / 'events_train_clean.csv')}
    for role, dirs, label_start in (('selection', selection_dirs, inner), ('test', test_dirs, outer)):
        for directory in dirs:
            name = str(directory.relative_to(fold_dir))
            train, _, train_end = span(directory / 'events_train_clean.csv')
            labels, label_min, label_max = span(directory / 'events_test_original.csv')
            props = pd.read_csv(directory / 'item_properties_train_clean.csv', usecols=['timestamp'])
            check('T', f'v7 {name} training ends before its labels and the outer test window',
                  train_end < label_start <= label_min and train_end < outer, train_end=train_end, label_min=label_min)
            check('T', f'v7 {name} labels inside their window',
                  label_min >= label_start and label_max < (outer if role == 'selection' else fold_end),
                  label_min=label_min, label_max=label_max)
            check('T', f'v7 {name} item properties end before its labels', int(props.timestamp.max()) < label_start,
                  props_max=int(props.timestamp.max()))
            check('T', f'v7 {name} train and label event ids disjoint',
                  not set(train.event_id) & set(labels.event_id), train=len(train), labels=len(labels))
            if directory.name != ('validation' if role == 'selection' else 'prepared'):
                check('T', f'v7 {name} trains on the bytes of its split',
                      sha(directory / 'events_train_clean.csv') == reference[role])
    manifest = json.loads((fold_dir / 'prepared' / 'manifest.json').read_text(encoding='utf-8'))
    check('T', 'v7 outlier filter (IF) off in the evaluated split',
          manifest['use_if'] is False and manifest['config']['use_if'] is False and manifest['if_removed_events'] == 0,
          if_removed_events=manifest['if_removed_events'])
else:
    check('T', 'v7 local data present', False, note='audit_runs/v7 or raw events.csv missing on this PC')


# ----------------------------------------------------------------------------- P: paired statistics (v7)
stored = pd.read_csv(HERE / 'audit' / 'results_v7' / 'report_paired_tests_all_folds.csv')
rng = np.random.default_rng(20261027)
runs = fold_dir / 'runs'
if runs.exists():
    for (policy, comparison, metric), row in stored.set_index(['policy', 'comparison', 'metric']).iterrows():
        left, right = [m.strip() for m in comparison.split(' - ')]

        def per_user(model):
            frames = [pd.read_csv(runs / f'seed_{s}' / f'per_user_{policy}_{model.replace("+", "_")}_k50.csv')
                      for s in (42, 43, 44)]
            return pd.concat(frames).groupby('visitorid')[metric].mean()
        a, b = per_user(left), per_user(right)
        diff = (a - b.reindex(a.index)).to_numpy().copy()
        diff[np.abs(diff) < 1e-12] = 0.  # seed averages of equal lists can differ in the last bit
        flips = rng.choice([-1., 1.], size=(10000, len(diff)))
        p = (1 + (np.abs((flips * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-12).sum()) / 10001
        same = (np.isclose(diff.mean(), row.difference, atol=1e-12) and len(diff) == row.users
                and (diff > 0).sum() == row.users_better and (diff < 0).sum() == row.users_worse
                and (diff == 0).sum() == row.users_tied)
        check('P', f'v7 {policy} {comparison} {metric} difference and counts', same, difference=round(diff.mean(), 6),
              stored=round(row.difference, 6))
        se = max(np.sqrt(row.p_value * (1 - row.p_value) / 10000), 1e-4)
        check('P', f'v7 {policy} {comparison} {metric} sign-flip p within 4 Monte Carlo SE', abs(p - row.p_value) <= 4 * se + 1e-4,
              p_recomputed=round(p, 5), p_stored=round(row.p_value, 5))
    for (fold_name, policy), group in stored.groupby(['fold', 'policy']):
        order = np.argsort(group.p_value.to_numpy(), kind='stable')
        p = group.p_value.to_numpy()[order]
        adjusted = np.minimum(1, np.maximum.accumulate(p * (len(p) - np.arange(len(p)))))
        holm = np.empty_like(adjusted)
        holm[order] = adjusted
        check('P', f'v7 {policy} Holm over {len(p)} comparisons', np.allclose(holm, group.p_holm.to_numpy())
              and ((holm < .05) == group.significant_holm.to_numpy()).all(), significant=int((holm < .05).sum()))

result = {'started_from': str(HERE), 'finished_utc': datetime.now(timezone.utc).isoformat(),
          'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, encoding='utf-8').stdout.strip(),
          'checks': len(checks), 'failed': [c for c in checks if not c['pass']], 'all_pass': all(c['pass'] for c in checks),
          'not_checked': unavailable,
          'by_group': {g: {'pass': sum(c['pass'] for c in checks if c['group'] == g),
                           'total': sum(c['group'] == g for c in checks)} for g in 'LSTP'},
          'detail': checks}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'detail'}, indent=2, default=str))
