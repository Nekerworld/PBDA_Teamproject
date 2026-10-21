"""Preregistered single-pass rolling-origin evaluation (protocols v6 and v7).

Every grid, fold, seed and selection rule below is frozen into plan.json before
any model is trained. Selection reads inner-validation labels only. Each fold's
outer test is opened once, after selection.json is sealed, and every opening is
appended to an access log. A changed plan or a re-made selection cannot reopen a
fold's test in the same root, and no root can open a test window on the same raw
input that a committed result set (audit/results_*) opened under another plan.

Usage (from Model/paper_revision_codex):
  python rolling_protocol.py run --root audit_runs/v6 --raw ../../data
  python rolling_protocol.py run --root audit_runs/v7 --raw ../../data --protocol v7
  python rolling_protocol.py gnn-job <spec.json>      (internal worker)
"""
import os
# ALS results depend on the BLAS thread count, so it is forced (not defaulted) before numpy is imported.
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for _name in THREAD_VARIABLES:
    os.environ[_name] = '1'
import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_protocol import UnifiedPreprocessor, dump_json, positive_sets, require_artifacts, sha256  # noqa: E402
from experiments import clone_protocol  # noqa: E402
from policy_evaluation import (POLICIES, EvaluationContext, FactorScores, candidate_indices,  # noqa: E402
                               factor_vectors, hybrid_ranking, selection_score, top_k)
from recommendation_scoring import hybrid_scores, minmax, ranks_from_scores  # noqa: E402
from stats_tools import compare_family  # noqa: E402

KS = (10, 20, 30, 40, 50)
PRIMARY_K = 50
FOLDS = [
    {'name': 'F1_hist', 'end_fraction': 1.0, 'test_size': 0.2, 'validation_size': 0.2,
     'test_window': 'Historical outer test of v1-v5; inspected repeatedly before this protocol.'},
    {'name': 'F2_mid', 'end_fraction': 0.8, 'test_size': 0.25, 'validation_size': 0.2,
     'test_window': 'Mostly the v1-v5 inner-validation window (used for selection, never as a test).'},
    {'name': 'F3_early', 'end_fraction': 0.6, 'test_size': 0.25, 'validation_size': 0.2,
     'test_window': 'Training data in v1-v5; never used for evaluation or selection before.'},
]
SEEDS = [42, 43, 44]
SELECTION_SEED = 42
ALS_GRID = [dict(factors=f, regularization=r, alpha=a, iterations=i)
            for f in (64, 128, 256) for r in (.02, .2) for a in (2., 10.) for i in (32, 256)]
BPR_GRID = [dict(factors=128, learning_rate=lr, regularization=r)
            for lr in (1e-3, 5e-3) for r in (.003, .03)]
BPR_EPOCH_CHECKPOINTS = (16, 32, 64, 128, 256)
GNN_GRID = [dict(layers=l, embedding_dim=d, learning_rate=lr)
            for l in (2, 3) for d in (64, 128) for lr in (1e-3, 3e-3)]
GNN_FIXED = dict(batch_size=8192, reg=1e-4, graph_mode='bipartite', loss_mode='infonce', n_negatives=32,
                 temperature=.2, edge_dropout=.1, propagation='unweighted', time_decay=True,
                 half_life_days=7., negative_mode='uniform', validation_interval=10,
                 early_stopping_patience=60, min_epochs=20, validation_metric='ndcg_recall_mean')
GNN_MAX_EPOCHS = 400
GNN_CAP_EXTENSION_EPOCHS = 800
MF_ALPHAS = [round(x, 1) for x in np.arange(0, 1.01, .1)]
HYBRID_BACKBONES = 3
CANDIDATE_KS = [500, 1000, 2000, 5000, 10000, 20000, 'all']
HYBRID_GRID = [dict(alpha=a, strategy=s, candidate_k=c, fusion=f, omega=w, gamma=g)
               for a in (.25, .5, .75) for c in CANDIDATE_KS
               for s in (('blend',) if c == 'all' else ('blend', 'balanced_union'))
               for f in ('minmax', 'rrf') for w in (.1, .2, .3, .4, .5, .6, .7, .8, .9)
               for g in (1., 1.1, 1.25, 1.5)]
MF_COMMON = dict(behavior_weighted=True, time_decay=True, half_life_days=7.)
BPR_NEGATIVES = 'popularity'
BASELINES = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Popularity', 'Recent', 'RecentPop']
POSTHOC_CANDIDATE_KS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 'all']
POSTHOC_GAMMAS = [1., 1.1, 1.25, 1.5, 2.]
MAX_PARALLEL_GNN = 4
PROTOCOL = 'rolling-origin-v6'
POSTHOC = True
PLAN_EXTRA = {}

# v7: one fold on the only window never evaluated or used for selection in v1-v6
# (v6 opened 0.45-1.0 as tests and 0.36-0.45 as F3 inner validation), and every
# trained family's grid widened past the edges v6 selected (I-23).
V7 = dict(
    PROTOCOL='rolling-origin-v7',
    FOLDS=[{'name': 'F4_unseen', 'end_fraction': 0.36, 'test_size': 0.25, 'validation_size': 0.2,
            'test_window': 'First 36% of events: training-only in v1-v6, never evaluated or used for selection.'}],
    ALS_GRID=[dict(factors=f, regularization=r, alpha=a, iterations=i)
              for f in (128, 256, 512, 1024) for r in (.002, .02, .2, 2.) for a in (10., 20., 40., 80.)
              for i in (8, 16, 32)],
    BPR_GRID=[dict(factors=f, learning_rate=lr, regularization=r)
              for f in (128, 256) for lr in (2.5e-3, 5e-3, 1e-2, 2e-2) for r in (3e-4, 3e-3, 3e-2)],
    BPR_EPOCH_CHECKPOINTS=(64, 128, 256, 384, 512, 768, 1024),
    GNN_GRID=[dict(layers=l, embedding_dim=d, learning_rate=lr)
              for l in (1, 2, 3, 4) for d in (64, 128, 256) for lr in (5e-4, 1e-3, 3e-3)],
    POSTHOC=False,
    PLAN_EXTRA={
        'purpose': ('Check whether v6 conclusions hold when ALS, BPR and GNN grids extend past the edges v6 '
                    'selected (issue I-23), on a window whose test was never opened.'),
        'grid_edge_rule': ('Grids are not widened again. A selection on an edge of a widened axis is recorded in '
                           'selection.json (grid_edges) and reported as a limitation.'),
        'power_note': ('About 413 test users: estimated unadjusted 95% CI half-widths 0.005-0.014 for Hybrid vs '
                       'GNN or ALS+BPR, larger than most v6 differences; non-significance is the expected outcome '
                       'for differences of v6 size.'),
        'posthoc_test_analyses': 'none'},
)
PROTOCOLS = {'v6': {}, 'v7': V7}


def use_protocol(name):
    globals().update(PROTOCOLS[name])


def grid_edges(selected, grid):
    """Axes (with more than one grid value) where the selected value is the grid minimum or maximum."""
    edges = {}
    for key, value in selected.items():
        values = sorted({c[key] for c in grid if key in c})
        if len(values) > 1 and value in (values[0], values[-1]):
            edges[key] = 'min' if value == values[0] else 'max'
    return edges


def plan_payload():
    return {
        'protocol': PROTOCOL, 'folds': FOLDS, 'seeds': SEEDS, 'selection_seed': SELECTION_SEED,
        'selection_rule': 'maximize mean(NDCG@50, Recall@50) on inner validation, identical for every model family',
        'policy_rule': ('Trained model configurations are selected under the repeat policy. Inference-time '
                        'parameters (ALS+BPR alpha, all Hybrid fusion parameters) are re-selected per policy.'),
        'als_grid': ALS_GRID, 'bpr_grid': BPR_GRID, 'bpr_epoch_checkpoints': list(BPR_EPOCH_CHECKPOINTS),
        'bpr_negatives': BPR_NEGATIVES, 'mf_common': MF_COMMON, 'gnn_grid': GNN_GRID, 'gnn_fixed': GNN_FIXED,
        'gnn_max_epochs': GNN_MAX_EPOCHS,
        'gnn_cap_rule': (f'If a top-{HYBRID_BACKBONES} GNN configuration selects an epoch within one validation '
                         f'interval of {GNN_MAX_EPOCHS}, it is retrained on validation with cap '
                         f'{GNN_CAP_EXTENSION_EPOCHS} and the extended run replaces it.'),
        'mf_alphas': MF_ALPHAS, 'hybrid_backbones': f'top {HYBRID_BACKBONES} GNN configurations by validation score',
        'hybrid_grid_size': len(HYBRID_GRID), 'candidate_ks': CANDIDATE_KS,
        'hybrid_grid_axes': {'alpha': [.25, .5, .75], 'strategy': ['blend', 'balanced_union'],
                             'fusion': ['minmax', 'rrf'], 'omega': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
                             'gamma': [1., 1.1, 1.25, 1.5]},
        'baselines': BASELINES, 'ks': list(KS), 'primary_k': PRIMARY_K,
        'statistics': ('Per policy and fold: user-level paired bootstrap 95% CI (10,000) and two-sided sign-flip '
                       'test (10,000) on seed-averaged per-user metrics; Holm correction over the family of '
                       'Hybrid vs every baseline on NDCG@50 and Recall@50. Seed SD reported separately.'),
        'posthoc_test_analyses': ('F1 only, after the sealed test evaluation, never used for selection: '
                                  f'candidate depth {POSTHOC_CANDIDATE_KS} with timing/memory, gamma '
                                  f'{POSTHOC_GAMMAS}, and an IF preprocessing ablation with the sealed configs.'),
        'preprocessing': dict(use_if=False, core_mode='events', min_user_events=3, min_item_events=3,
                              random_state=42),
        **PLAN_EXTRA,
    }


# ----------------------------------------------------------------------------- utilities
def git_state():
    def run(*args):
        try:
            return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty_files': [line for line in (run('status', '--porcelain', '--', '.') or '').splitlines()
                            if line.strip() and not line[3:].startswith(('audit_runs', 'Model/paper_revision_codex/audit_runs'))]}


def native_threads():
    """Threads the loaded BLAS/OpenMP libraries actually use (the variables can be rewritten after loading)."""
    import threadpoolctl
    return sorted({(info['internal_api'], info['num_threads']) for info in threadpoolctl.threadpool_info()})


def environment():
    import torch, implicit, scipy, sklearn
    return {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'torch': torch.__version__, 'implicit': implicit.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'thread_env': {name: os.environ.get(name) for name in THREAD_VARIABLES},
            'native_threads': native_threads(), 'torch_threads': torch.get_num_threads()}


# Line endings are normalised so that a checkout's core.autocrlf cannot change the hash of unchanged code.
# Plans written before this scheme existed (v6, v7) hashed raw bytes and are checked with 'sha256-raw'.
CODE_HASH_SCHEME = 'sha256-lf'


def code_hashes(root=HERE, scheme=CODE_HASH_SCHEME):
    if scheme not in ('sha256-lf', 'sha256-raw'):
        raise ValueError(f'Unknown code hash scheme {scheme}')
    root = Path(root)
    files = sorted(root.glob('*.py')) + sorted(root.glob('audit_tools/*.py'))
    digest = {}
    for path in files:
        data = path.read_bytes()
        if scheme == 'sha256-lf':
            data = data.replace(b'\r\n', b'\n')
        digest[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return digest


def rel(path, root):
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def log(root, stage, **extra):
    payload = {'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'stage': stage, **extra}
    with open(Path(root) / 'progress.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, default=str) + '\n')
    print(json.dumps(payload, default=str), flush=True)


def config_name(prefix, config):
    return prefix + '_' + '_'.join(f'{k}{v}' for k, v in config.items()).replace('.', 'p')


# ----------------------------------------------------------------------------- data
def make_fold_raw(raw, destination, end_fraction):
    destination = Path(destination)
    if (destination / 'fold_raw.json').exists():
        return json.loads((destination / 'fold_raw.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('item_properties_part1.csv', 'item_properties_part2.csv', 'category_tree.csv'):
        if not (destination / name).exists():
            os.link(Path(raw) / name, destination / name)
    if end_fraction >= 1:
        if not (destination / 'events.csv').exists():
            os.link(Path(raw) / 'events.csv', destination / 'events.csv')
        end_cutoff = None
    else:
        events = pd.read_csv(Path(raw) / 'events.csv')
        order = np.lexsort((np.arange(len(events)), events.timestamp.to_numpy()))
        end_cutoff = int(events.timestamp.to_numpy()[order[int(len(events) * end_fraction)]])
        events.loc[events.timestamp < end_cutoff].to_csv(destination / 'events.csv', index=False)
    info = {'end_fraction': end_fraction, 'end_cutoff_timestamp': end_cutoff,
            'source_events_sha256': sha256(Path(raw) / 'events.csv'),
            'fold_events_sha256': sha256(destination / 'events.csv'),
            'rule': 'keep raw events with timestamp < end cutoff (tie-safe, file order preserved)'}
    dump_json(destination / 'fold_raw.json', info)
    return info


def prepare_fold(fold, root, raw, use_if=False):
    base = Path(root) / fold['name']
    fold_raw = make_fold_raw(raw, base / 'fold_raw', fold['end_fraction'])
    prepared = base / ('prepared_if' if use_if else 'prepared')
    if not (prepared / 'manifest.json').exists():
        if prepared.exists() and any(prepared.iterdir()):
            raise FileExistsError(f'Incomplete preprocessing output: {prepared}')
        UnifiedPreprocessor(data_dir=base / 'fold_raw', processed_dir=prepared, test_size=fold['test_size'],
                            validation_size=fold['validation_size'], use_if=use_if, core_mode='events',
                            min_user_events=3, min_item_events=3, random_state=42).run()
    return prepared, fold_raw


# ----------------------------------------------------------------------------- MF training
def interaction_matrix(directory):
    import scipy.sparse as sp
    import learning
    train = pd.read_csv(Path(directory) / 'events_train_clean.csv')
    pairs = learning.interaction_table(train, MF_COMMON['behavior_weighted'], MF_COMMON['time_decay'],
                                       MF_COMMON['half_life_days'])
    users, user_ids = pd.factorize(pairs.visitorid, sort=True)
    items, item_ids = pd.factorize(pairs.itemid, sort=True)
    matrix = sp.coo_matrix((pairs.weight.astype(float), (users, items)),
                           shape=(len(user_ids), len(item_ids))).tocsr()
    return matrix, np.asarray(user_ids, dtype=np.int64), np.asarray(item_ids, dtype=np.int64)


def train_als(matrix, config, seed):
    import als_debug
    model = als_debug.train_als(matrix.copy(), config['factors'], config['regularization'], config['iterations'],
                                seed, config['alpha'], num_threads=8, use_gpu=False)
    return np.asarray(model.user_factors), np.asarray(model.item_factors)


def train_bpr(matrix, config, epochs, seed, checkpoints=(), callback=None):
    import torch
    import learning
    result = learning.train_weighted_bpr(matrix.copy(), config['factors'], config['regularization'], epochs, seed,
                                         'cuda' if torch.cuda.is_available() else 'cpu', True,
                                         negative_mode=BPR_NEGATIVES, learning_rate=config['learning_rate'],
                                         checkpoint_epochs=checkpoints, checkpoint_callback=callback)
    return result.user_factors, result.item_factors


def save_factors(directory, users, items, als, bpr, meta):
    directory = Path(directory)
    files = []
    for prefix, (uf, itf) in (('als', als), ('bpr', bpr)):
        for entity, value in (('user', uf), ('item', itf)):
            name = f'{prefix}_{entity}_factors.npy'
            np.save(directory / name, value)
            files.append(name)
    for entity, ids, column in (('user', users, 'visitorid'), ('item', items, 'itemid')):
        name = f'als_{entity}_mapping.csv'
        pd.DataFrame({f'{entity}_index': range(len(ids)), column: ids}).to_csv(directory / name, index=False)
        files.append(name)
    from data_protocol import stamp_artifacts
    stamp_artifacts(directory, 'als', files, **meta)


def load_factors(directory, context):
    directory = Path(directory)
    users = pd.read_csv(directory / 'als_user_mapping.csv').visitorid.to_numpy()
    items = pd.read_csv(directory / 'als_item_mapping.csv').itemid.to_numpy()
    als = factor_vectors(context, users, np.load(directory / 'als_user_factors.npy'), items,
                         np.load(directory / 'als_item_factors.npy'))
    bpr = factor_vectors(context, users, np.load(directory / 'bpr_user_factors.npy'), items,
                         np.load(directory / 'bpr_item_factors.npy'))
    return als, bpr


def load_gnn(directory, context):
    require_artifacts(directory, 'gnn')
    users = pd.read_csv(Path(directory) / 'gnn_user_embeddings.csv')
    items = pd.read_csv(Path(directory) / 'gnn_item_embeddings.csv')
    return factor_vectors(context, users.visitorid.to_numpy(), users.drop(columns='visitorid').to_numpy(),
                          items.itemid.to_numpy(), items.drop(columns='itemid').to_numpy())


def evaluate_vectors(contexts, vectors):
    out = {}
    for policy, ctx in contexts.items():
        out[policy] = ctx.metrics(ctx.rank_scores({u: vectors[u] for u in ctx.active}, PRIMARY_K), PRIMARY_K)
    return out


def mf_validation(fold_dir, val_dir, contexts):
    """ALS and BPR grids on inner validation; returns the selection record."""
    out = Path(fold_dir) / 'selection_mf.json'
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))
    matrix, users, items = interaction_matrix(val_dir)
    rows, timing = [], {'ALS': 0., 'BPR': 0.}
    for config in ALS_GRID:
        start = time.perf_counter()
        uf, itf = train_als(matrix, config, SELECTION_SEED)
        timing['ALS'] += time.perf_counter() - start
        scores = evaluate_vectors(contexts, factor_vectors(contexts['repeat'], users, uf, items, itf))
        rows.append({'family': 'ALS', **config, **{f'{p}_{m}': v for p, d in scores.items() for m, v in d.items()}})
        print('ALS', config, scores['repeat'], flush=True)
    for config in BPR_GRID:
        def callback(epoch, uf, itf, config=config):
            scores = evaluate_vectors(contexts, factor_vectors(contexts['repeat'], users, uf, items, itf))
            rows.append({'family': 'BPR', **config, 'epochs': epoch,
                         **{f'{p}_{m}': v for p, d in scores.items() for m, v in d.items()}})
            print('BPR', config, epoch, scores['repeat'], flush=True)
        start = time.perf_counter()
        train_bpr(matrix, config, max(BPR_EPOCH_CHECKPOINTS), SELECTION_SEED, BPR_EPOCH_CHECKPOINTS, callback)
        timing['BPR'] += time.perf_counter() - start
    frame = pd.DataFrame(rows)
    frame['repeat_score'] = (frame.repeat_ndcg + frame.repeat_recall) / 2
    frame.to_csv(Path(fold_dir) / 'validation_mf_grid.csv', index=False)
    best = {}
    for family in ('ALS', 'BPR'):
        part = frame.loc[frame.family == family].sort_values('repeat_score', ascending=False, kind='stable')
        best[family] = {k: v for k, v in part.iloc[0].to_dict().items() if not isinstance(v, float) or not np.isnan(v)}
    record = {'als': {k: best['ALS'][k] for k in ('factors', 'regularization', 'alpha', 'iterations')},
              'bpr': {k: best['BPR'][k] for k in ('factors', 'learning_rate', 'regularization')},
              'bpr_epochs': int(best['BPR']['epochs']), 'validation': best,
              'fits': {'ALS': len(ALS_GRID), 'BPR': len(BPR_GRID)},
              'evaluated_configs': {'ALS': len(ALS_GRID), 'BPR': len(BPR_GRID) * len(BPR_EPOCH_CHECKPOINTS)},
              'train_seconds': timing}
    record['als']['factors'] = int(record['als']['factors'])
    record['als']['iterations'] = int(record['als']['iterations'])
    record['bpr']['factors'] = int(record['bpr']['factors'])
    dump_json(out, record)
    return record


def fit_mf(directory, selection, seed):
    directory = Path(directory)
    if (directory / 'als_artifacts.json').exists():
        return json.loads((directory / 'mf_fit.json').read_text(encoding='utf-8'))
    matrix, users, items = interaction_matrix(directory)
    start = time.perf_counter()
    als = train_als(matrix, selection['als'], seed)
    als_seconds = time.perf_counter() - start
    start = time.perf_counter()
    bpr = train_bpr(matrix, selection['bpr'], selection['bpr_epochs'], seed)
    bpr_seconds = time.perf_counter() - start
    info = {'seed': seed, 'als': selection['als'], 'bpr': selection['bpr'], 'bpr_epochs': selection['bpr_epochs'],
            'train_seconds': {'ALS': als_seconds, 'BPR': bpr_seconds}}
    dump_json(directory / 'mf_fit.json', info)
    save_factors(directory, users, items, als, bpr, {'v6_fit': info, 'bpr_used': True})
    return info


# ----------------------------------------------------------------------------- GNN jobs
def gnn_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def gnn_spec(job_dir, source, config, seed, mode, epochs):
    return {'job_dir': str(Path(job_dir).resolve()), 'source': str(Path(source).resolve()), 'config': config,
            'fixed': GNN_FIXED, 'device': gnn_device(), 'seed': seed, 'mode': mode, 'epochs': epochs}


def at_cap(chosen, cap, interval):
    """True when the selected epoch is within one validation interval of the cap (e.g. 190 of 200)."""
    return chosen >= cap - interval


def gnn_job(spec_path):
    import torch
    from model import GNNEmbeddingGenerator
    spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    job = Path(spec['job_dir'])
    if (job / 'job_result.json').exists():
        return
    if not (job / 'manifest.json').exists():
        clone_protocol(spec['source'], job)
    kwargs = {**spec['fixed'], **spec['config'], 'seed': spec['seed'], 'epochs': spec['epochs'],
              'device': spec['device']}
    start = time.perf_counter()
    if spec['mode'] == 'select':
        generator = GNNEmbeddingGenerator(processed_dir=job, **kwargs)
        graph = generator._prepare_graph_structures(generator._load_inputs())
        truth = positive_sets(pd.read_csv(job / 'events_test_original.csv'))
        model, chosen, history = generator._fit_graph(graph, spec['epochs'], torch, truth)
        generator._publish(model, graph, history, chosen, start, {'mode': 'v6_inner_validation_selection'})
    else:
        kwargs['early_stopping_patience'] = None
        generator = GNNEmbeddingGenerator(processed_dir=job, **kwargs)
        generator.run()
        chosen, history = spec['epochs'], []
    scores = [row.get('validation_original_ndcg') for row in history if 'validation_original_ndcg' in row]
    dump_json(job / 'job_result.json', {
        'spec': spec, 'chosen_epochs': int(chosen), 'seconds': time.perf_counter() - start,
        'evaluated_checkpoints': len(scores), 'last_epoch': history[-1]['epoch'] if history else spec['epochs'],
        'at_cap': bool(spec['mode'] == 'select' and at_cap(chosen, spec['epochs'], spec['fixed']['validation_interval']))})


def run_gnn_jobs(root, specs):
    pending = [s for s in specs if not (Path(s['job_dir']) / 'job_result.json').exists()]
    running = []
    spec_dir = Path(root) / 'job_specs'
    spec_dir.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running) < MAX_PARALLEL_GNN:
            spec = pending.pop(0)
            path = spec_dir / (hashlib.sha1(spec['job_dir'].encode()).hexdigest()[:16] + '.json')
            dump_json(path, spec)
            Path(spec['job_dir']).parent.mkdir(parents=True, exist_ok=True)
            logfile = open(Path(spec['job_dir'] + '.log'), 'w', encoding='utf-8')
            process = subprocess.Popen([sys.executable, str(HERE / 'rolling_protocol.py'), 'gnn-job', str(path)],
                                       cwd=HERE, stdout=logfile, stderr=subprocess.STDOUT)
            running.append((process, spec, logfile))
        time.sleep(5)
        for entry in list(running):
            process, spec, logfile = entry
            if process.poll() is not None:
                logfile.close()
                running.remove(entry)
                if process.returncode != 0 or not (Path(spec['job_dir']) / 'job_result.json').exists():
                    raise RuntimeError(f'GNN job failed: {spec["job_dir"]} (see .log)')
                log(root, 'gnn_job_done', job=rel(spec['job_dir'], root))
    return [json.loads((Path(s['job_dir']) / 'job_result.json').read_text(encoding='utf-8')) for s in specs]


# ----------------------------------------------------------------------------- hybrid grid
def hybrid_grid(context, als, bpr, gnn, backbone):
    scores = FactorScores(context, als, bpr, gnn)
    rows = []
    for (alpha, count, strategy), group in itertools.groupby(
            HYBRID_GRID, key=lambda c: (c['alpha'], c['candidate_k'], c['strategy'])):
        prepared, cached = {}, {}
        for uid in context.active:
            idx = candidate_indices(scores, uid, alpha, count, strategy)
            prepared[uid] = idx
            if not len(idx):  # novel policy: the user has already seen the whole catalog
                continue
            mf, g = scores.mf(uid, alpha)[idx], scores.gnn[uid][idx]
            cached[uid] = (minmax(mf), minmax(g), ranks_from_scores(mf), ranks_from_scores(g), context.available[idx])
        for config in group:
            rankings = {}
            for uid, idx in prepared.items():
                if uid not in cached:
                    rankings[uid] = idx
                    continue
                mf_n, g_n, mr, gr, available = cached[uid]
                w = config['omega']
                fused = (w * mf_n + (1 - w) * g_n) if config['fusion'] == 'minmax' else (w / (60 + mr) + (1 - w) / (60 + gr))
                fused = fused * np.where(available, config['gamma'], 1.)
                rankings[uid] = idx[top_k(fused, PRIMARY_K, tie=idx)] if len(idx) else idx
            metrics = context.metrics(rankings, PRIMARY_K)
            rows.append({'backbone': backbone, **config, **metrics, 'score': selection_score(metrics)})
    return rows


def best_row(rows, keys):
    frame = pd.DataFrame(rows).sort_values('score', ascending=False, kind='stable')
    return {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in frame.iloc[0][keys].to_dict().items()}


# ----------------------------------------------------------------------------- test evaluation
def model_rankings(context, als, bpr, gnn_std, gnn_hybrid, policy_selection):
    std = FactorScores(context, als, bpr, gnn_std)
    hyb = FactorScores(context, als, bpr, gnn_hybrid)
    alpha = policy_selection['mf_alpha']
    config = policy_selection['hybrid']
    rankings = {
        'ALS': context.rank_scores(std.als, max(KS)), 'BPR': context.rank_scores(std.bpr, max(KS)),
        'ALS+BPR': context.rank_scores({u: std.mf(u, alpha) for u in context.active}, max(KS)),
        'GNN': context.rank_scores(std.gnn, max(KS)),
        'Hybrid': hybrid_ranking(hyb, config, max(KS)),
        'Hybrid_ALS_only': hybrid_ranking(hyb, config, max(KS), mf_source='als'),
        'Popularity': context.popularity(max(KS)), 'Recent': context.recent(max(KS)),
        'RecentPop': context.recent(max(KS), fill=True)}
    return rankings


def access_log(root, entry):
    path = Path(root) / 'test_access_log.jsonl'
    previous = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    for item in previous:
        if item['fold'] == entry['fold'] and item['purpose'] == entry['purpose'] and item['plan_sha256'] != entry['plan_sha256']:
            raise RuntimeError(f'Fold {entry["fold"]} test was already opened under plan {item["plan_sha256"]}; '
                               'a changed plan cannot reopen it.')
        if (item['fold'] == entry['fold'] and item['purpose'] == entry['purpose']
                and item.get('selection_sha256') != entry.get('selection_sha256')):
            raise RuntimeError(f'Fold {entry["fold"]} test was already opened under selection '
                               f'{item.get("selection_sha256")}; a re-made selection cannot reopen it.')
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry) + '\n')


COMMITTED_RESULTS = HERE / 'audit'


def committed_test_conflicts(raw_hashes, plan_hash, folds, records=None):
    """Committed sealed tests on the same raw input and test window that were opened under another plan.

    The per-root access log cannot see other result roots, so a new root would otherwise be free to
    reopen a window that a committed protocol already evaluated.
    """
    conflicts = []
    for plan_path in sorted(Path(records or COMMITTED_RESULTS).glob('results_*/plan.json')):
        log_path = plan_path.parent / 'test_access_log.jsonl'
        stored = json.loads(plan_path.read_text(encoding='utf-8'))
        if (not raw_hashes or not log_path.exists() or stored.get('raw_input_sha256') != raw_hashes
                or stored['plan_sha256'] == plan_hash):
            continue
        windows = {f['name']: (f['end_fraction'], f['test_size']) for f in stored['plan']['folds']}
        opened = {json.loads(line)['fold'] for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()}
        for fold in folds:
            for name in opened:
                if windows.get(name) == (fold['end_fraction'], fold['test_size']):
                    conflicts.append({'fold': fold['name'], 'opened_as': name, 'record': str(plan_path.parent)})
    return conflicts


def fold_statistics(policy, details):
    """details: model -> list of per-seed per-user frames (k=50)."""
    per_user = {m: pd.concat(frames).groupby('visitorid')[['ndcg', 'recall', 'precision', 'hit_rate']].mean()
                for m, frames in details.items()}
    rows = compare_family(per_user, 'Hybrid', BASELINES)
    frame = pd.DataFrame(rows)
    frame.insert(0, 'policy', policy)
    distribution = []
    for model, frames in details.items():
        user = per_user[model]
        for metric in ('ndcg', 'recall'):
            q = user[metric].quantile([.25, .5, .75, .9]).to_dict()
            distribution.append({'policy': policy, 'model': model, 'metric': metric, 'mean': user[metric].mean(),
                                 'zero_share': float((user[metric] == 0).mean()), 'p25': q[.25], 'median': q[.5],
                                 'p75': q[.75], 'p90': q[.9],
                                 'hit_users_any_seed': int((pd.concat(frames).groupby('visitorid').hits.max() > 0).sum())})
    return frame, pd.DataFrame(distribution)


def evaluate_fold_test(root, fold, prepared, selection, refits, plan_hash):
    fold_dir = Path(root) / fold['name']
    if (fold_dir / 'test_metrics.csv').exists():
        return
    access_log(root, {'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'fold': fold['name'], 'purpose': 'sealed_test',
                      'plan_sha256': plan_hash, 'selection_sha256': sha256(fold_dir / 'selection.json'),
                      'git': git_state()['commit']})
    metric_rows, stats_rows, dist_rows, populations = [], [], [], []
    for policy in POLICIES:
        details = {}
        for seed in SEEDS:
            seed_dir = Path(refits[seed]['mf'])
            context = EvaluationContext(seed_dir, policy)
            if seed == SEEDS[0]:
                populations.append(context.population())
            als, bpr = load_factors(seed_dir, context)
            gnn_std = load_gnn(refits[seed]['gnn'], context)
            gnn_hyb = load_gnn(refits[seed]['hybrid_gnn'], context)
            rankings = model_rankings(context, als, bpr, gnn_std, gnn_hyb, selection['policies'][policy])
            for model, ranked in rankings.items():
                records = [(u, int(context.ids[i]), r) for u, items in ranked.items() for r, i in enumerate(items, 1)]
                pd.DataFrame(records, columns=['visitorid', 'itemid', 'rank']).to_csv(
                    seed_dir / f'recommendations_{policy}_{model.replace("+", "_")}.csv', index=False)
                for k in KS:
                    frame = context.details(ranked, k)
                    metric_rows.append({'fold': fold['name'], 'policy': policy, 'seed': seed, 'model': model, 'k': k,
                                        'users': len(frame), 'hit_users': int(frame.hit_rate.sum()),
                                        'empty_lists': int((frame.recommendations == 0).sum()),
                                        **{m: float(frame[m].mean()) for m in ('ndcg', 'recall', 'precision', 'hit_rate')}})
                    if k == PRIMARY_K:
                        frame.to_csv(seed_dir / f'per_user_{policy}_{model.replace("+", "_")}_k50.csv', index=False)
                        details.setdefault(model, []).append(frame.assign(seed=seed))
        stats, dist = fold_statistics(policy, details)
        stats_rows.append(stats)
        dist_rows.append(dist)
    report = fold_dir / 'report'
    report.mkdir(exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(report / 'test_metrics.csv', index=False)
    summary = metrics.groupby(['fold', 'policy', 'model', 'k'])[['ndcg', 'recall', 'precision', 'hit_rate', 'hit_users']].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary.to_csv(report / 'test_summary.csv', index=False)
    pd.concat(stats_rows).to_csv(report / 'paired_tests.csv', index=False)
    pd.concat(dist_rows).to_csv(report / 'per_user_distribution.csv', index=False)
    pd.DataFrame(populations).to_csv(report / 'populations.csv', index=False)
    # Written last: its presence marks the fold's test as complete when a run resumes.
    metrics.to_csv(fold_dir / 'test_metrics.csv', index=False)


# ----------------------------------------------------------------------------- post-hoc analyses (F1)
def posthoc_candidates(root, fold, selection, refits):
    fold_dir = Path(root) / fold['name']
    out = fold_dir / 'report' / 'posthoc_candidate_depth.csv'
    if out.exists():
        return
    access_log(root, {'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'fold': fold['name'],
                      'purpose': 'posthoc_candidate_gamma', 'plan_sha256': selection['plan_sha256'],
                      'selection_sha256': sha256(fold_dir / 'selection.json'), 'git': git_state()['commit']})
    seed = SEEDS[0]
    seed_dir = Path(refits[seed]['mf'])
    context = EvaluationContext(seed_dir, 'repeat')
    config = selection['policies']['repeat']['hybrid']
    users = pd.read_csv(seed_dir / 'als_user_mapping.csv').visitorid.to_numpy()
    items = pd.read_csv(seed_dir / 'als_item_mapping.csv').itemid.to_numpy()
    if not np.array_equal(items, context.ids):
        raise ValueError('MF item order differs from the catalog')
    row_of = {int(u): i for i, u in enumerate(users)}
    au, ai = np.load(seed_dir / 'als_user_factors.npy'), np.load(seed_dir / 'als_item_factors.npy')
    bu, bi = np.load(seed_dir / 'bpr_user_factors.npy'), np.load(seed_dir / 'bpr_item_factors.npy')
    gdir = Path(refits[seed]['hybrid_gnn'])
    gu = pd.read_csv(gdir / 'gnn_user_embeddings.csv').set_index('visitorid')
    gi = pd.read_csv(gdir / 'gnn_item_embeddings.csv').set_index('itemid').loc[context.ids].to_numpy(dtype=np.float32)
    gu_rows = {int(u): v for u, v in zip(gu.index, gu.to_numpy(dtype=np.float32))}
    rows = []
    for depth in POSTHOC_CANDIDATE_KS:
        cfg = dict(config, candidate_k=depth)
        timings = {'mf_scoring': 0., 'candidate_selection': 0., 'gnn_scoring': 0., 'fusion_topk': 0.}
        rankings, found, reachable = {}, 0, 0
        tracemalloc.start()
        for uid in context.active:
            t = time.perf_counter()
            als = minmax(ai @ au[row_of[uid]]); bpr = minmax(bi @ bu[row_of[uid]])
            timings['mf_scoring'] += time.perf_counter() - t
            scores = FactorScores.__new__(FactorScores)
            scores.context, scores.als, scores.bpr = context, {uid: als}, {uid: bpr}
            t = time.perf_counter()
            idx = candidate_indices(scores, uid, cfg['alpha'], depth, cfg['strategy'])
            timings['candidate_selection'] += time.perf_counter() - t
            t = time.perf_counter()
            g = gi[idx] @ gu_rows[uid]
            timings['gnn_scoring'] += time.perf_counter() - t
            t = time.perf_counter()
            fused = hybrid_scores(scores.mf(uid, cfg['alpha'])[idx], g, context.available[idx], cfg['omega'],
                                  cfg['gamma'], cfg['fusion'])
            rankings[uid] = idx[top_k(fused, PRIMARY_K, tie=idx)]
            timings['fusion_topk'] += time.perf_counter() - t
            found += len(context.truth_idx[uid] & set(idx.tolist()))
            reachable += len(context.truth_idx[uid])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics = context.metrics(rankings, PRIMARY_K)
        n = len(context.active)
        rows.append({'candidate_k': depth, 'candidate_share_of_catalog': (len(context.ids) if depth == 'all' else depth) / len(context.ids),
                     **metrics, 'candidate_recall_of_reachable_truth': found / max(reachable, 1),
                     **{f'{k}_ms_per_user': 1000 * v / n for k, v in timings.items()},
                     'total_ms_per_user': 1000 * sum(timings.values()) / n, 'peak_python_mb': peak / 2**20})
    gnn_full = 0.
    for uid in context.active:
        t = time.perf_counter()
        top_k(gi @ gu_rows[uid], PRIMARY_K)
        gnn_full += time.perf_counter() - t
    frame = pd.DataFrame(rows)
    frame['gnn_full_catalog_ms_per_user'] = 1000 * gnn_full / len(context.active)
    frame['note'] = 'Post-hoc on sealed test, seed 42, repeat policy; never used for selection.'
    frame.to_csv(out, index=False)
    hyb = FactorScores(context, *load_factors(seed_dir, context), load_gnn(gdir, context))
    gamma_rows = []
    for gamma in POSTHOC_GAMMAS:
        ranked = hybrid_ranking(hyb, dict(config, gamma=gamma), PRIMARY_K)
        shown = np.concatenate([v for v in ranked.values() if len(v)])
        gamma_rows.append({'gamma': gamma, **context.metrics(ranked, PRIMARY_K),
                           'available_share_top50': float(context.available[shown].mean()),
                           'note': 'Post-hoc on sealed test, seed 42; never used for selection.'})
    pd.DataFrame(gamma_rows).to_csv(fold_dir / 'report' / 'posthoc_gamma.csv', index=False)


def if_ablation(root, fold, raw, selection, plan_hash):
    fold_dir = Path(root) / fold['name']
    out = fold_dir / 'report' / 'if_ablation.csv'
    if out.exists():
        return
    prepared_if, _ = prepare_fold(fold, root, raw, use_if=True)
    stats = pd.read_csv(prepared_if / 'dataset_statistics.csv').set_index('stage')
    removal = []
    for kind in ('view', 'addtocart', 'transaction'):
        before = int(stats.loc['train_before_filter', f'{kind}_events'])
        removal.append({'event': kind, 'train_before': before,
                        'removed_by_if': int(stats.loc['train_if_removed', f'{kind}_events']),
                        'after_if': int(stats.loc['train_after_if', f'{kind}_events']),
                        'final_after_core': int(stats.loc['train_final', f'{kind}_events'])})
    removal = pd.DataFrame(removal)
    removal['removed_share'] = removal.removed_by_if / removal.train_before
    removal.to_csv(fold_dir / 'report' / 'if_removal_by_event.csv', index=False)
    manifest = json.loads((prepared_if / 'manifest.json').read_text(encoding='utf-8'))
    dump_json(fold_dir / 'report' / 'if_summary.json', {
        'if_features': manifest['if_features'], 'if_item_percent_of_train_items': manifest['if_item_percent_of_train_items'],
        'if_event_percent_of_prefilter_train': manifest['if_event_percent_of_prefilter_train'],
        'if_event_percent_of_raw_input': manifest['if_event_percent_of_raw_input'],
        'evaluation_users': manifest['evaluation_users'], 'warnings': manifest['data_quality_warnings']})
    access_log(root, {'utc': pd.Timestamp.now(tz='UTC').isoformat(), 'fold': fold['name'], 'purpose': 'posthoc_if_ablation',
                      'plan_sha256': plan_hash, 'selection_sha256': sha256(fold_dir / 'selection.json'),
                      'git': git_state()['commit']})
    seed = SEEDS[0]
    mf_dir = fold_dir / 'if_ablation' / f'seed_{seed}'
    if not (mf_dir / 'manifest.json').exists():
        clone_protocol(prepared_if, mf_dir)
    fit_mf(mf_dir, selection['mf'], seed)
    specs = [gnn_spec(fold_dir / 'if_ablation' / 'gnn', prepared_if, selection['gnn']['config'], seed, 'refit',
                      selection['gnn']['epochs'])]
    if selection['hybrid_backbone']['name'] != selection['gnn']['name']:
        specs.append(gnn_spec(fold_dir / 'if_ablation' / 'hybrid_gnn', prepared_if,
                              selection['hybrid_backbone']['config'], seed, 'refit', selection['hybrid_backbone']['epochs']))
    run_gnn_jobs(root, specs)
    hybrid_dir = Path(specs[-1]['job_dir'])
    context = EvaluationContext(mf_dir, 'repeat')
    als, bpr = load_factors(mf_dir, context)
    rankings = model_rankings(context, als, bpr, load_gnn(specs[0]['job_dir'], context),
                              load_gnn(hybrid_dir, context), selection['policies']['repeat'])
    rows = [{'preprocessing': 'IF', 'model': m, **context.metrics(r, PRIMARY_K), **context.population()}
            for m, r in rankings.items()]
    reference = pd.read_csv(fold_dir / 'test_metrics.csv')
    reference = reference.loc[(reference.policy == 'repeat') & (reference.seed == seed) & (reference.k == PRIMARY_K)]
    for _, r in reference.iterrows():
        rows.append({'preprocessing': 'no-IF', 'model': r.model, 'ndcg': r.ndcg, 'recall': r.recall,
                     'precision': r.precision, 'hit_rate': r.hit_rate, 'hit_users': r.hit_users, 'users': r.users})
    frame = pd.DataFrame(rows)
    frame['note'] = 'Seed 42, repeat policy, configurations sealed on no-IF validation.'
    frame.to_csv(out, index=False)


# ----------------------------------------------------------------------------- fold driver
def extend_capped_backbones(gnn_rows, extend):
    """Extend capped rows among the top HYBRID_BACKBONES until none is left unextended.

    gnn_rows is re-sorted in place after each round, because an extension can lift another capped
    configuration into the backbone set. extend(rows) returns one update per row, carrying 'result'.
    """
    results = []
    while True:
        todo = [row for row in gnn_rows[:HYBRID_BACKBONES] if row['at_cap'] and not row.get('extended')]
        if not todo:
            return results
        for row, update in zip(todo, extend(todo)):
            update = dict(update)
            results.append(update.pop('result'))
            row.update(update, extended=True)
        gnn_rows.sort(key=lambda r: r['score'], reverse=True)


def leakage_gate(prepared, fold_dir):
    """leakage_check.json is written whether or not the check passes, so a resumed run must read its verdict."""
    from audit_tools.leakage_check import main as leakage_main
    path = Path(fold_dir) / 'leakage_check.json'
    if not path.exists():
        leakage_main(prepared, Path(fold_dir) / 'fold_raw', gnn=True, output=path)
    if not json.loads(path.read_text(encoding='utf-8')).get('all_pass'):
        raise RuntimeError(f'Leakage check failed for {Path(fold_dir).name} ({path})')


def run_fold(root, fold, raw, plan_hash):
    fold_dir = Path(root) / fold['name']
    prepared, fold_raw = prepare_fold(fold, root, raw)
    log(root, 'prepared', fold=fold['name'])
    leakage_gate(prepared, fold_dir)
    val_dir = prepared / 'validation'
    contexts = {p: EvaluationContext(val_dir, p) for p in POLICIES}
    dump_json(fold_dir / 'validation_population.json', [c.population() for c in contexts.values()])
    # GNN validation jobs start first (GPU); MF grid runs in this process meanwhile.
    specs = [gnn_spec(fold_dir / 'gnn_val' / config_name('gnn', c), val_dir, c, SELECTION_SEED, 'select', GNN_MAX_EPOCHS)
             for c in GNN_GRID]
    import threading
    results, errors = {}, []

    def worker():
        try:
            results['gnn'] = run_gnn_jobs(root, specs)
        except Exception as error:  # surfaced after join
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    mf = mf_validation(fold_dir, val_dir, contexts)
    log(root, 'mf_selected', fold=fold['name'], als=mf['als'], bpr=mf['bpr'], bpr_epochs=mf['bpr_epochs'])
    thread.join()
    if errors:
        raise errors[0]
    val_mf = fold_dir / 'val_mf'
    if not (val_mf / 'manifest.json').exists():
        clone_protocol(val_dir, val_mf)
    fit_mf(val_mf, mf, SELECTION_SEED)
    als_v, bpr_v = load_factors(val_mf, contexts['repeat'])
    gnn_rows = []
    for spec, result in zip(specs, results['gnn']):
        vectors = load_gnn(spec['job_dir'], contexts['repeat'])
        scores = evaluate_vectors(contexts, vectors)
        gnn_rows.append({'name': Path(spec['job_dir']).name, 'config': spec['config'], 'job_dir': spec['job_dir'],
                         'epochs': result['chosen_epochs'], 'at_cap': result['at_cap'], 'seconds': result['seconds'],
                         'repeat': scores['repeat'], 'novel': scores['novel'],
                         'score': selection_score(scores['repeat'])})
    gnn_rows.sort(key=lambda r: r['score'], reverse=True)

    def extend(rows):
        extensions = [gnn_spec(Path(row['job_dir']).parent / (row['name'] + '_cap800'), val_dir, row['config'],
                               SELECTION_SEED, 'select', GNN_CAP_EXTENSION_EPOCHS) for row in rows]
        updates = []
        for row, spec, result in zip(rows, extensions, run_gnn_jobs(root, extensions)):
            scores = evaluate_vectors(contexts, load_gnn(spec['job_dir'], contexts['repeat']))
            updates.append(dict(job_dir=spec['job_dir'], epochs=result['chosen_epochs'], at_cap=result['at_cap'],
                                seconds=row['seconds'] + result['seconds'], repeat=scores['repeat'],
                                novel=scores['novel'], score=selection_score(scores['repeat']), result=result))
        return updates
    extension_results = extend_capped_backbones(gnn_rows, extend)
    dump_json(fold_dir / 'validation_gnn.json', gnn_rows)
    policies, grid_rows = {}, {}
    for policy in POLICIES:
        ctx = contexts[policy]
        als_p, bpr_p = {u: als_v[u] for u in ctx.active}, {u: bpr_v[u] for u in ctx.active}
        alpha_rows = []
        for alpha in MF_ALPHAS:
            fs = FactorScores(ctx, als_p, bpr_p)
            m = ctx.metrics(ctx.rank_scores({u: fs.mf(u, alpha) for u in ctx.active}, PRIMARY_K), PRIMARY_K)
            alpha_rows.append({'alpha': alpha, **m, 'score': selection_score(m)})
        pd.DataFrame(alpha_rows).to_csv(fold_dir / f'validation_mf_alpha_{policy}.csv', index=False)
        rows = []
        for backbone in gnn_rows[:HYBRID_BACKBONES]:
            path = fold_dir / f'validation_hybrid_{policy}_{backbone["name"]}.csv'
            if path.exists():
                part = pd.read_csv(path).to_dict('records')
            else:
                part = hybrid_grid(ctx, als_p, bpr_p, load_gnn(backbone['job_dir'], ctx), backbone['name'])
                pd.DataFrame(part).to_csv(path, index=False)
                log(root, 'hybrid_grid_done', fold=fold['name'], policy=policy, backbone=backbone['name'])
            rows.extend(part)
        grid_rows[policy] = rows
        policies[policy] = {'mf_alpha': best_row(alpha_rows, ['alpha'])['alpha']}
    keys = ['backbone', 'alpha', 'strategy', 'candidate_k', 'fusion', 'omega', 'gamma', 'ndcg', 'recall', 'score']
    repeat_backbone = best_row(grid_rows['repeat'], keys)['backbone']
    for policy in POLICIES:
        # Trained configurations follow the repeat policy (preregistered), so every
        # policy re-selects inference-time parameters on the repeat-selected backbone.
        best = best_row([r for r in grid_rows[policy] if r['backbone'] == repeat_backbone], keys)
        best['candidate_k'] = best['candidate_k'] if str(best['candidate_k']) == 'all' else int(best['candidate_k'])
        unrestricted = best_row(grid_rows[policy], keys)
        policies[policy].update(hybrid=best, unrestricted_best_backbone=unrestricted['backbone'],
                                hybrid_at_grid_boundary={
                                    'candidate_k': str(best['candidate_k']) in (str(CANDIDATE_KS[0]), str(CANDIDATE_KS[-1])),
                                    'omega': float(best['omega']) in (.1, .9), 'gamma': float(best['gamma']) in (1., 1.5)})
    gnn_results = list(results['gnn']) + list(extension_results)
    selection = {'plan_sha256': plan_hash, 'fold': fold['name'], 'mf': mf, 'gnn': gnn_rows[0],
                 'hybrid_backbone': next(r for r in gnn_rows if r['name'] == repeat_backbone),
                 'policies': policies, 'locked_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                 'outer_test_used_for_selection': False,
                 'grid_edges': {
                     'ALS': grid_edges(mf['als'], ALS_GRID),
                     'BPR': grid_edges({**mf['bpr'], 'epochs': mf['bpr_epochs']},
                                       [dict(c, epochs=e) for c in BPR_GRID for e in BPR_EPOCH_CHECKPOINTS]),
                     'GNN': grid_edges(gnn_rows[0]['config'], GNN_GRID),
                     'Hybrid_backbone': grid_edges(next(r for r in gnn_rows if r['name'] == repeat_backbone)['config'],
                                                   GNN_GRID)},
                 'budget': {'fits': {'ALS': mf['fits']['ALS'], 'BPR': mf['fits']['BPR'],
                                     'GNN': len(gnn_results), 'Hybrid': 0},
                            'evaluated_configs': {'ALS': mf['evaluated_configs']['ALS'],
                                                  'BPR': mf['evaluated_configs']['BPR'],
                                                  'ALS+BPR': len(MF_ALPHAS),
                                                  'GNN': sum(r['evaluated_checkpoints'] for r in gnn_results),
                                                  'Hybrid': len(HYBRID_GRID) * HYBRID_BACKBONES},
                            'train_seconds': {'ALS': mf['train_seconds']['ALS'], 'BPR': mf['train_seconds']['BPR'],
                                              'GNN': sum(r['seconds'] for r in gnn_results)},
                            'note': 'Hybrid adds no training fits; it reuses the MF and GNN fits above.'}}
    if not (fold_dir / 'selection.json').exists():
        dump_json(fold_dir / 'selection.json', selection)
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    log(root, 'selection_sealed', fold=fold['name'], selection=sha256(fold_dir / 'selection.json'))
    refits, specs = {}, []
    for seed in SEEDS:
        seed_dir = fold_dir / 'runs' / f'seed_{seed}'
        if not (seed_dir / 'manifest.json').exists():
            clone_protocol(prepared, seed_dir)
        gnn_dir = fold_dir / 'runs' / f'gnn_seed_{seed}'
        specs.append(gnn_spec(gnn_dir, prepared, selection['gnn']['config'], seed, 'refit', selection['gnn']['epochs']))
        hybrid_dir = gnn_dir
        if selection['hybrid_backbone']['name'] != selection['gnn']['name']:
            hybrid_dir = fold_dir / 'runs' / f'hybrid_gnn_seed_{seed}'
            specs.append(gnn_spec(hybrid_dir, prepared, selection['hybrid_backbone']['config'], seed, 'refit',
                                  selection['hybrid_backbone']['epochs']))
        refits[seed] = {'mf': str(seed_dir), 'gnn': str(gnn_dir), 'hybrid_gnn': str(hybrid_dir)}
    thread = threading.Thread(target=lambda: results.__setitem__('refit', run_gnn_jobs(root, specs)))
    thread.start()
    for seed in SEEDS:
        fit_mf(refits[seed]['mf'], selection['mf'], seed)
    thread.join()
    if 'refit' not in results:
        raise RuntimeError('GNN refit jobs failed')
    dump_json(fold_dir / 'refits.json', {s: {k: rel(v, root) for k, v in d.items()} for s, d in refits.items()})
    evaluate_fold_test(root, fold, prepared, selection, refits, plan_hash)
    log(root, 'test_evaluated', fold=fold['name'])
    return selection, refits


def write_reports(root):
    root = Path(root)
    tables = []
    for fold in FOLDS:
        path = root / fold['name'] / 'report' / 'test_summary.csv'
        if path.exists():
            tables.append(pd.read_csv(path))
    if not tables:
        return
    summary = pd.concat(tables)
    summary.to_csv(root / 'report_all_folds_summary.csv', index=False)
    primary = summary.loc[summary.k == PRIMARY_K]
    temporal = primary.groupby(['policy', 'model'])[['ndcg_mean', 'recall_mean']].agg(['mean', 'std', 'min', 'max'])
    temporal.columns = ['_'.join(c) for c in temporal.columns]
    temporal.reset_index().to_csv(root / 'report_temporal_variation.csv', index=False)
    tests = [pd.read_csv(root / f['name'] / 'report' / 'paired_tests.csv').assign(fold=f['name'])
             for f in FOLDS if (root / f['name'] / 'report' / 'paired_tests.csv').exists()]
    if tests:
        pd.concat(tests).to_csv(root / 'report_paired_tests_all_folds.csv', index=False)


def run(root, raw, folds=None, posthoc=True, protocol=None):
    if protocol:
        use_protocol(protocol)
    root, raw = Path(root), Path(raw).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = plan_payload()
    plan_text = json.dumps(plan, sort_keys=True)
    plan_hash = hashlib.sha256(plan_text.encode()).hexdigest()
    plan_path = root / 'plan.json'
    if plan_path.exists():
        stored = json.loads(plan_path.read_text(encoding='utf-8'))
        if stored['plan_sha256'] != plan_hash:
            raise RuntimeError('plan.json differs from the code plan; use a new root (no silent re-planning).')
    else:
        state = git_state()
        if state['dirty_files']:
            raise RuntimeError(f'Commit code before a preregistered run; dirty: {state["dirty_files"]}')
        dump_json(plan_path, {'plan_sha256': plan_hash, 'plan': plan, 'git': state, 'environment': environment(),
                              'raw_input_sha256': {n: sha256(raw / n) for n in ('events.csv', 'item_properties_part1.csv',
                                                                               'item_properties_part2.csv', 'category_tree.csv')},
                              'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                              'code_hash_scheme': CODE_HASH_SCHEME, 'code_sha256': code_hashes()})
    stored = json.loads(plan_path.read_text(encoding='utf-8'))
    code_now = code_hashes(scheme=stored.get('code_hash_scheme', 'sha256-raw'))
    if code_now != stored['code_sha256']:
        changed = sorted(n for n in set(code_now) | set(stored['code_sha256']) if code_now.get(n) != stored['code_sha256'].get(n))
        raise RuntimeError(f'Code changed since the plan was frozen: {changed}')
    conflicts = committed_test_conflicts(stored.get('raw_input_sha256'), plan_hash,
                                         [f for f in FOLDS if not folds or f['name'] in folds])
    if conflicts:
        raise RuntimeError(f'These test windows were already opened under another committed plan: {conflicts}')
    for fold in FOLDS:
        if folds and fold['name'] not in folds:
            continue
        selection, refits = run_fold(root, fold, raw, plan_hash)
        if posthoc and POSTHOC and fold['name'] == FOLDS[0]['name']:
            posthoc_candidates(root, fold, selection, refits)
            if_ablation(root, fold, raw, selection, plan_hash)
            log(root, 'posthoc_done', fold=fold['name'])
        write_reports(root)
    log(root, 'completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--raw', type=Path, required=True)
    r.add_argument('--folds', nargs='*')
    r.add_argument('--no-posthoc', action='store_true')
    r.add_argument('--protocol', choices=sorted(PROTOCOLS), default='v6')
    j = sub.add_parser('gnn-job')
    j.add_argument('spec', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.root, args.raw, args.folds, not args.no_posthoc, args.protocol)
    else:
        gnn_job(args.spec)
