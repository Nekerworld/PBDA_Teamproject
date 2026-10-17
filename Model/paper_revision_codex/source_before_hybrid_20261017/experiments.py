"""Reproducible filter and architecture runs. Gamma is chosen on validation only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

from data_protocol import (UnifiedPreprocessor, chronological_split, dump_json, sha256,
    positive_sets, require_protocol, require_artifacts, load_source_tables, DEFAULT_DATA, DEFAULT_PROCESSED)
from model import ALSRecommender, GNNEmbeddingGenerator, ReRanker
from ranking_evaluation import TestSetEvaluator, ranking_metrics
from recommendation_scoring import (FactorScorer, GNNScorer, top_indices, hybrid_scores,
    hybrid_rerank_scores, append_ranking, recommendation_frame)


CONDITIONS = {
    'no_if_events': (False, 'events'),
    'if_events': (True, 'events'),
    'no_if_neighbors': (False, 'neighbors'),
    'if_neighbors': (True, 'neighbors'),
}
PAPER_K_VALUES = [10, 20, 30, 40, 50]
TABLE_K_VALUES = [10, 30, 50]
MODEL_ORDER = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']
GAMMA_GRID = [round(0.2 * step, 1) for step in range(1, 16)]
GAMMA_REFERENCE = [5.0, 10.0]

# Default experiment settings (paper_v2). Outer test is never used for selection.
LOCKED = {
    'primary': 'if_events',
    'primary_reason': 'Isolation Forest is enabled for the primary experiment; condition is fixed before test metrics.',
    'core_threshold': 3,
    'core_definition': 'event rows; distinct-neighbor core is an ablation only',
    'behavior_weights': {'view': 1.0, 'addtocart': 6.0, 'transaction': 12.0},
    'als_factors': 128,
    'als_iterations': 256,
    'als_alpha': 2.0,
    'als_regularization': 0.02,
    'als_device': 'cpu',
    'als_use_gpu': False,
    'als_threads': 8,
    'time_decay': True,
    'half_life_days': 7.0,
    'negative_mode': 'popularity',
    'bpr_epochs': 128,
    'bpr_implementation': 'torch weighted pairwise loss',
    'gnn_max_epochs': 200,
    'gnn_patience': 30,
    'gnn_min_epochs': 20,
    'gnn_embedding_dim': 64,
    'gnn_layers': 3,
    'gnn_edge_dropout': 0.1,
    'gnn_propagation': 'unweighted',
    'gnn_item_item_edges': False,
    'gnn_graph_mode': 'bipartite',
    'gnn_loss_mode': 'infonce',
    'gnn_n_negatives': 32,
    'gnn_temperature': 0.2,
    'gnn_batch_size': 8192,
    'gnn_learning_rate': 0.001,
    'gnn_negative_mode': 'uniform',
    'ensemble_alpha': 0.5,
    'als_weight': 0.8,
    'fusion': 'rrf',
    'rerank_m': 100,
    'als_weight_grid': [0.6, 0.7, 0.8, 0.9, 1.0],
    'fusion_grid': ['minmax', 'rrf', 'gate'],
    'rerank_m_grid': [50, 100, None],
    'candidate_k_grid': [200, 500, 1000],
    'gamma_grid': GAMMA_GRID,
    'gamma_reference': GAMMA_REFERENCE,
    'gamma_rule': 'maximum original-truth NDCG on inner validation; ties prefer gamma closest to 1',
    'selection_rule': 'maximum validation original NDCG; ties prefer higher als_weight then fusion minmax/rrf/gate order then smaller rerank_m then gamma closer to 1',
    'primary_k': 50,
    'test_size': 0.1,
    'candidate_k': 500,
    'recommendation_depth': 200,
    'seeds': [42, 43, 44],
    'threads': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'torch_num_threads': 1, 'als_threads': 8},
    'mode': 'paper_v2',
    'hybrid_sweep': True,
}

# Previous sealed revision_v3 settings (opt-in via --legacy-v3).
LEGACY_V3 = {
    'primary': 'if_events',
    'primary_reason': 'Isolation Forest is enabled for the primary experiment; condition is fixed before test metrics.',
    'core_threshold': 3,
    'core_definition': 'event rows; distinct-neighbor core is an ablation only',
    'behavior_weights': {'view': 1.0, 'addtocart': 6.0, 'transaction': 12.0},
    'als_factors': 128,
    'als_iterations': 256,
    'als_alpha': 1.5,
    'als_regularization': 0.03,
    'als_device': 'cpu',
    'als_use_gpu': False,
    'als_threads': 8,
    'time_decay': False,
    'half_life_days': 7.0,
    'negative_mode': 'uniform',
    'bpr_epochs': 128,
    'bpr_implementation': 'torch weighted pairwise loss',
    'gnn_max_epochs': 200,
    'gnn_patience': 25,
    'gnn_embedding_dim': 8,
    'gnn_layers': 2,
    'gnn_edge_dropout': 0.0,
    'gnn_propagation': 'unweighted',
    'gnn_item_item_edges': False,
    'ensemble_alpha': 0.5,
    'als_weight': 0.4,
    'fusion': 'minmax',
    'rerank_m': None,
    'gamma_grid': [0.5, 1.0, 1.5, 2.0],
    'gamma_rule': 'maximum original-truth NDCG on inner validation; ties prefer gamma closest to 1',
    'primary_k': 50,
    'test_size': 0.2,
    'candidate_k': 500,
    'recommendation_depth': 200,
    'seeds': [42, 43, 44],
    'threads': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'torch_num_threads': 1, 'als_threads': 8},
    'mode': 'revision_v3',
    'hybrid_sweep': False,
}


def code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name.startswith('test_'):
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_conditions(data_dir, output, test_size=0.1):
    """Outer test is chronological 90:10. Validation stays 20% of that train block."""
    data_dir, output = Path(data_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new condition directory')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between zero and one')
    output.mkdir(parents=True)
    events, clean, invalid, props, categories = load_source_tables(data_dir)
    invalid.to_csv(output / 'invalid_events.csv', index=False)
    train, test, cutoff = chronological_split(clean, test_size)
    inner, val, val_cutoff = chronological_split(train)
    hashes = {name: sha256(data_dir / name) for name in (
        'events.csv', 'item_properties_part1.csv', 'item_properties_part2.csv', 'category_tree.csv')}
    summaries, original_hashes, reference_hashes, stage_summaries = [], [], [], []
    for name, (use_if, mode) in CONDITIONS.items():
        print('Preparing condition', name, flush=True)
        pre = UnifiedPreprocessor(data_dir, output / name, use_if=use_if, core_mode=mode)
        pre._write_partition(inner, val, props, categories, val_cutoff, output / name / 'validation', 'validation', len(train))
        manifest = pre._write_partition(train, test, props, categories, cutoff, output / name, 'test', len(events))
        manifest = pre.finalize_outer(manifest, output / name, events, clean, invalid)
        stats = pd.read_csv(output / name / 'dataset_statistics.csv').set_index('stage')
        stage_stats = stats.reset_index()
        stage_stats.insert(0, 'condition', name)
        raw_valid_events = int(stats.loc['raw_valid', 'events'])
        outer_train_events = int(stats.loc['train_before_filter', 'events'])
        stage_stats['events_percent_of_raw_valid'] = 100 * stage_stats.events / raw_valid_events
        stage_stats['events_percent_of_outer_train'] = np.where(
            stage_stats.stage.str.startswith('train_'),
            100 * stage_stats.events / outer_train_events, np.nan)
        stage_summaries.append(stage_stats)
        primary_stats = stats.loc['train_final']
        summaries.append({
            'condition': name, 'reference_id': manifest['reference_id'],
            'original_truth_id': manifest['original_truth_id'],
            'fixed_evaluation_users': manifest['evaluation_users'],
            'conditional_users': manifest['conditional_evaluation_users'],
            'mean_original_truth_retention': manifest['mean_original_truth_retention'],
            'mean_reference_truth_retention': manifest['mean_reference_truth_retention'],
            'test_positive_event_coverage': manifest['test_positive_coverage'],
            'if_removed_events': manifest['if_removed_events'],
            'if_removed_event_percent_of_outer_train': manifest['if_event_percent_of_prefilter_train'],
            'if_removed_event_percent_of_raw_input': manifest['if_event_percent_of_raw_input'],
            'train_final_unique_user_item_pairs': int(primary_stats['unique_user_item_pairs']),
            'train_final_sparsity_percent': float(primary_stats['sparsity_percent']),
            'train_final_event_retention_percent': 100 * int(primary_stats['events']) / outer_train_events,
            'warnings': ' | '.join(manifest.get('data_quality_warnings', [])),
            **{column: int(stats.loc['train_final', column]) for column in
               ['events', 'users', 'items', 'addtocart_events', 'transaction_events']}})
        original_hashes.append(sha256(output / name / 'events_test_original.csv'))
        reference_hashes.append(sha256(output / name / 'events_test_reference.csv'))
    frame = pd.DataFrame(summaries)
    if frame.reference_id.nunique() != 1 or frame.original_truth_id.nunique() != 1:
        raise ValueError('Conditions do not share an identical evaluation reference')
    if len(set(original_hashes)) != 1 or len(set(reference_hashes)) != 1:
        raise ValueError('Fixed original or reference event files differ across conditions')
    frame.to_csv(output / 'filter_comparison.csv', index=False)
    stage_frame = pd.concat(stage_summaries, ignore_index=True)
    stage_frame.to_csv(output / 'filter_stage_statistics.csv', index=False)
    dump_json(output / 'condition_manifest.json', {
        'input_hashes': hashes, 'conditions': list(CONDITIONS),
        'selection_basis': 'Primary condition is selected in experiment_plan.json before test evaluation.',
        'files': {name: sha256(output / name / 'manifest.json') for name in CONDITIONS},
        'filter_comparison_hash': sha256(output / 'filter_comparison.csv'),
        'filter_stage_statistics_hash': sha256(output / 'filter_stage_statistics.csv'),
        'original_events_hash': original_hashes[0], 'reference_events_hash': reference_hashes[0]})
    return frame


def write_dataset_statistics_tables(datasets_dir, output_dir, condition='if_events'):
    """Write paper-ready stage tables from a prepared condition's audited statistics."""
    datasets_dir, output_dir = Path(datasets_dir), Path(output_dir)
    stats_path = datasets_dir / condition / 'dataset_statistics.csv'
    if not stats_path.exists():
        raise FileNotFoundError(f'Prepared dataset statistics are missing: {stats_path}')
    stats = pd.read_csv(stats_path).set_index('stage')
    stages = [
        ('raw_valid', '1. Raw Dataset'),
        ('train_before_filter', '2. Initial Train Set'),
        ('train_after_if', '3. Train Set: After IF'),
        ('train_final', '4. Final Train Set'),
        ('test_before_filter', '5. Test Set'),
    ]
    missing = {stage for stage, _ in stages} - set(stats.index)
    if missing:
        raise ValueError(f'Dataset statistics are missing stages: {sorted(missing)}')
    raw = stats.loc['raw_valid']

    def retained_count(value, baseline):
        percentage = 100.0 * float(value) / float(baseline) if baseline else 0.0
        return f'{int(value):,} ({percentage:.1f}%)'

    columns = ['Metric', *[label for _, label in stages]]
    table1_rows = []
    for label, field in [('Users (|U|)', 'users'), ('Items (|I|)', 'items'),
                         ('Interactions (|E|)', 'events')]:
        baseline = raw[field]
        table1_rows.append([label, *[
            retained_count(stats.loc[stage, field], baseline) for stage, _ in stages]])
    sparsity, average = [], []
    for stage, _ in stages:
        users = float(stats.loc[stage, 'users'])
        events = float(stats.loc[stage, 'events'])
        items = float(stats.loc[stage, 'items'])
        pairs = float(stats.loc[stage, 'unique_user_item_pairs'])
        sparsity.append(f'{100.0 * (1.0 - pairs / (users * items)):.4f}%'
                        if users * items else 'N/A')
        average.append(f'{events / users:.2f}' if users else 'N/A')
    table1_rows.extend([['Sparsity (%)', *sparsity], ['Avg. Inter./User', *average]])
    table1 = pd.DataFrame(table1_rows, columns=columns)

    breakdown_rows = []
    for label, field in [('Users (|U|)', 'users'), ('Items (|I|)', 'items'),
                         ('Total Interactions (|E|)', 'events')]:
        breakdown_rows.append([
            label, *[retained_count(stats.loc[stage, field], raw[field]) for stage, _ in stages]])
    for label, field in [('• View', 'view_events'), ('• Add-to-cart', 'addtocart_events'),
                         ('• Transaction', 'transaction_events')]:
        values = []
        for stage, _ in stages:
            count = int(stats.loc[stage, field])
            total = int(stats.loc[stage, 'events'])
            share = 100.0 * count / total if total else 0.0
            values.append(f'{count:,} ({share:.2f}%)')
        breakdown_rows.append([label, *values])
    breakdown_rows.extend([['Sparsity (%)', *sparsity], ['Avg. Inter./User', *average]])
    table_s1 = pd.DataFrame(breakdown_rows, columns=columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    table1_path = output_dir / 'table1_dataset_statistics.csv'
    table_s1_path = output_dir / 'table_s1_event_breakdown_reference.csv'
    table1.to_csv(table1_path, index=False)
    table_s1.to_csv(table_s1_path, index=False)

    def print_table(frame):
        text = frame.to_string(index=False)
        encoding = sys.stdout.encoding or 'utf-8'
        try:
            text.encode(encoding)
        except UnicodeEncodeError:
            text = text.replace('•', '-')
            text = text.encode(encoding, errors='backslashreplace').decode(encoding)
        print(text)

    print('\nTable 1. Dataset statistics')
    print_table(table1)
    print('\nTable S1. Event breakdown reference')
    print_table(table_s1)
    return table1, table_s1


def clone_protocol(source, destination):
    """Immutable shared data are hard-linked; model artifacts stay run-local."""
    source, destination = Path(source), Path(destination)
    manifest = require_protocol(source)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in {'manifest.json', *manifest['hashes'], *manifest['audit_hashes']}:
        try:
            os.link(source / name, destination / name)
        except OSError as error:
            raise OSError(f'Could not hard-link {name} from {source} to {destination}') from error
    if (source / 'validation').exists():
        clone_protocol(source / 'validation', destination / 'validation')


def available_share(records, available_ids, k):
    """Fraction of top-k recommended items that are in stock."""
    frame = recommendation_frame(records)
    top = frame.loc[frame['rank'] <= k]
    return float(top.itemid.isin(available_ids).mean()) if len(top) else 0.0


def gamma_sweep(directory, gammas, k=50, als_weight=0.4, candidate_k=500,
                fusion='minmax', rerank_m=None, rrf_k=60, k_values=None, reference_gammas=()):
    """Validation only. Reference gammas are reported but never selected."""
    directory = Path(directory)
    meta = require_protocol(directory)
    if meta['role'] != 'validation':
        raise ValueError('Gamma selection requires a validation protocol')
    als = require_artifacts(directory, 'als')
    require_artifacts(directory, 'gnn')
    factor = FactorScorer(directory, als['ensemble_alpha'])
    graph = GNNScorer(directory, factor.item_ids)
    available = pd.read_csv(directory / 'feature_train_inliers.csv').set_index('itemid').reindex(factor.item_ids).has_available.eq(1).to_numpy()
    available_ids = set(factor.item_ids[available].tolist())
    truth = positive_sets(pd.read_csv(directory / 'events_test_original.csv'))
    eval_ks = sorted(set([k, *(k_values or [])]))
    grid = [float(gamma) for gamma in gammas]
    reference = [float(gamma) for gamma in reference_gammas if float(gamma) not in grid]
    rows = {gamma: [] for gamma in grid + reference}
    candidate_shares = []
    for uid in truth:
        if uid not in factor.users:
            continue
        component, _, retrieval = factor.scores(uid)
        candidate = top_indices(retrieval, factor.item_ids, min(candidate_k, len(factor.item_ids)))
        candidate_shares.append(float(available[candidate].mean()))
        graph_scores = graph.scores(uid, candidate)
        for gamma, records in rows.items():
            scores = hybrid_rerank_scores(component[candidate], graph_scores, available[candidate],
                                          als_weight, gamma, fusion, rrf_k, rerank_m)
            append_ranking(records, uid, factor.item_ids[candidate], scores, max(eval_ks))
    candidate_share = float(np.mean(candidate_shares)) if candidate_shares else 0.0
    result = pd.DataFrame([
        {'gamma': gamma, 'reference': gamma in reference, 'k': eval_k, 'role': 'validation', 'scope': 'original',
         'fusion': fusion, 'als_weight': als_weight, 'rerank_m': rerank_m, 'candidate_k': candidate_k,
         **ranking_metrics(recommendation_frame(records), truth, eval_k)[0],
         'topk_available_share': available_share(records, available_ids, eval_k),
         'candidate_available_share': candidate_share,
         'catalog_available_share': float(available.mean())}
        for gamma, records in rows.items() for eval_k in eval_ks])
    result['distance_from_neutral'] = (result.gamma - 1).abs()
    selection_rows = result.loc[(result.k == k) & ~result.reference]
    selected = float(selection_rows.sort_values(
        ['ndcg', 'distance_from_neutral', 'gamma'],
        ascending=[False, True, True]).iloc[0].gamma)
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / 'gamma_validation.csv', index=False)
    dump_json(directory / 'gamma_selection.json', {
        'selected_gamma': selected, 'criterion': LOCKED['gamma_rule'], 'protocol_id': meta['protocol_id'],
        'als_weight': als_weight, 'fusion': fusion, 'rerank_m': rerank_m, 'candidate_k': candidate_k,
        'als_artifact_hash': sha256(directory / 'als_artifacts.json'),
        'gnn_artifact_hash': sha256(directory / 'gnn_artifacts.json'), 'grid': grid,
        'reference_gammas': reference})
    return selected


def hybrid_config_sweep(directory, plan, k=50, k_values=None):
    """Choose fusion settings on validation only; no outer-test labels."""
    directory = Path(directory)
    meta = require_protocol(directory)
    if meta['role'] != 'validation':
        raise ValueError('Hybrid config selection requires a validation protocol')
    als = require_artifacts(directory, 'als')
    require_artifacts(directory, 'gnn')
    factor = FactorScorer(directory, als['ensemble_alpha'])
    graph = GNNScorer(directory, factor.item_ids)
    available = pd.read_csv(directory / 'feature_train_inliers.csv').set_index('itemid').reindex(factor.item_ids).has_available.eq(1).to_numpy()
    available_ids = set(factor.item_ids[available].tolist())
    truth = positive_sets(pd.read_csv(directory / 'events_test_original.csv'))
    weight_grid = plan.get('als_weight_grid', [plan['als_weight']])
    fusion_grid = plan.get('fusion_grid', [plan.get('fusion', 'minmax')])
    rerank_grid = plan.get('rerank_m_grid', [plan.get('rerank_m')])
    candidate_grid = plan.get('candidate_k_grid', [plan['candidate_k']])
    gamma_grid = [float(gamma) for gamma in plan['gamma_grid']]
    reference_grid = [float(gamma) for gamma in plan.get('gamma_reference', []) if float(gamma) not in gamma_grid]
    eval_ks = sorted(set([k, *(k_values or [])]))
    max_k = max(eval_ks)
    fusion_order = {name: index for index, name in enumerate(['minmax', 'rrf', 'gate'])}
    cache = {}
    for uid in truth:
        if uid not in factor.users:
            continue
        component, _, retrieval = factor.scores(uid)
        cache[uid] = (component, retrieval, graph.scores(uid))
    rows = []
    for candidate_k in candidate_grid:
        candidates = {uid: top_indices(retrieval, factor.item_ids, min(candidate_k, len(factor.item_ids)))
                      for uid, (_, retrieval, _) in cache.items()}
        candidate_share = (float(np.mean([available[candidate].mean() for candidate in candidates.values()]))
                           if candidates else 0.0)
        for als_weight in weight_grid:
            for fusion in fusion_grid:
                for rerank_m in rerank_grid:
                    for gamma in gamma_grid + reference_grid:
                        records = []
                        for uid, (component, _, full_gnn) in cache.items():
                            candidate = candidates[uid]
                            scores = hybrid_rerank_scores(component[candidate], full_gnn[candidate],
                                                          available[candidate], als_weight, gamma, fusion, 60, rerank_m)
                            append_ranking(records, uid, factor.item_ids[candidate], scores, max_k)
                        for eval_k in eval_ks:
                            metrics = ranking_metrics(recommendation_frame(records), truth, eval_k)[0]
                            rows.append({'candidate_k': candidate_k, 'als_weight': als_weight, 'fusion': fusion,
                                         'rerank_m': -1 if rerank_m is None else rerank_m, 'gamma': gamma,
                                         'reference': gamma in reference_grid,
                                         'k': eval_k, 'role': 'validation', 'scope': 'original', **metrics,
                                         'topk_available_share': available_share(records, available_ids, eval_k),
                                         'candidate_available_share': candidate_share,
                                         'catalog_available_share': float(available.mean()),
                                         'distance_from_neutral': abs(gamma - 1.0),
                                         'fusion_order': fusion_order.get(fusion, 99)})
    result = pd.DataFrame(rows)
    directory.mkdir(parents=True, exist_ok=True)
    result.to_csv(directory / 'hybrid_config_validation.csv', index=False)
    selection_rows = result.loc[(result.k == k) & ~result.reference]
    best = selection_rows.sort_values(
        ['ndcg', 'als_weight', 'fusion_order', 'rerank_m', 'distance_from_neutral', 'gamma'],
        ascending=[False, False, True, True, True, True]).iloc[0]
    selected = {
        'candidate_k': int(best.candidate_k),
        'als_weight': float(best.als_weight),
        'fusion': str(best.fusion),
        'rerank_m': None if int(best.rerank_m) < 0 else int(best.rerank_m),
        'gamma': float(best.gamma),
        'validation_ndcg': float(best.ndcg),
        'criterion': plan.get('selection_rule', LOCKED['gamma_rule']),
        'protocol_id': meta['protocol_id'],
    }
    dump_json(directory / 'hybrid_config_selection.json', selected)
    curve = result.loc[
        (result.candidate_k == best.candidate_k) & (result.als_weight == best.als_weight)
        & (result.fusion == best.fusion) & (result.rerank_m == best.rerank_m),
        ['gamma', 'reference', 'k', 'role', 'scope', 'ndcg', 'recall', 'precision', 'hit_rate', 'f1',
         'users_evaluated', 'missing_recommendation_users', 'topk_available_share',
         'candidate_available_share', 'catalog_available_share', 'distance_from_neutral']].copy()
    curve.to_csv(directory / 'gamma_validation.csv', index=False)
    dump_json(directory / 'gamma_selection.json', {
        'selected_gamma': selected['gamma'], **selected,
        'als_artifact_hash': sha256(directory / 'als_artifacts.json'),
        'gnn_artifact_hash': sha256(directory / 'gnn_artifacts.json'),
        'grid': gamma_grid, 'reference_gammas': reference_grid})
    return selected


def paired_bootstrap(a, b=None, samples=2000, seed=2026):
    a = np.asarray(a, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise ValueError('Paired samples must be aligned')
        a = a - b
    if len(a) < 1:
        raise ValueError('Bootstrap sample is empty')
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 100):
        take = min(100, samples - start)
        means[start:start + take] = a[rng.integers(len(a), size=(take, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def assert_same_users(frames):
    index = frames[0].index
    for frame in frames[1:]:
        if not frame.index.equals(index):
            raise ValueError('User alignment mismatch')
    return index


def summarize(root, primary_k=50):
    root = Path(root)
    plan = json.loads((root / 'experiment_plan.json').read_text(encoding='utf-8'))
    primary_condition = plan['primary_condition']
    metrics, topk_frames, details, candidates, provenance = [], [], {}, [], []
    for run in sorted((root / 'runs').iterdir()):
        if not (run / 'complete.json').exists():
            continue
        completion = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
        config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
        require_artifacts(run, 'evaluation')
        if completion['evaluation_hash'] != sha256(run / 'evaluation_artifacts.json'):
            raise ValueError('Completed run was changed')
        frame = pd.read_csv(run / 'ranking_metrics.csv')
        for column in ['condition', 'ablation', 'seed']:
            frame[column] = config[column]
        frame['run'] = run.name
        metrics.append(frame)
        topk = pd.read_csv(run / 'topk_metrics.csv')
        for column in ['condition', 'ablation', 'seed']:
            topk[column] = config[column]
        topk_frames.append(topk)
        candidate = pd.read_csv(run / 'candidate_metrics.csv')
        candidate['run'] = run.name
        candidate['condition'] = config['condition']
        candidate['ablation'] = config['ablation']
        candidate['seed'] = config['seed']
        candidates.append(candidate)
        for name in frame.model:
            per = pd.read_csv(run / f"metrics_original_{name.replace('+', '_')}_at_{primary_k}_users.csv").set_index('visitorid').sort_index()
            details[(config['condition'], config['ablation'], config['seed'], name)] = per
        provenance.append({'run': run.name, 'config': sha256(run / 'run_config.json'),
            'evaluation': sha256(run / 'evaluation_artifacts.json'), 'manifest': sha256(run / 'manifest.json')})
    if not metrics:
        raise ValueError('No completed experiments')
    allmetrics = pd.concat(metrics, ignore_index=True)
    if allmetrics.reference_id.nunique() != 1 or allmetrics.original_truth_id.nunique() != 1:
        raise ValueError('Cannot compare experiments with different reference truth')
    allmetrics.to_csv(root / 'experiment_metrics.csv', index=False)
    pd.concat(candidates, ignore_index=True).to_csv(root / 'candidate_comparison.csv', index=False)
    allmetrics.groupby(['condition', 'ablation', 'model'])[['ndcg', 'recall']].agg(['mean', 'std', 'count']).to_csv(root / 'seed_summary.csv')
    all_topk = pd.concat(topk_frames, ignore_index=True)
    comparison = all_topk.loc[
        (all_topk.condition == primary_condition)
        & (all_topk.ablation == 'main')
        & (all_topk.scope == 'original')
        & (all_topk.k.isin(TABLE_K_VALUES))
    ]
    expected_pairs = {(model, k_value) for model in MODEL_ORDER for k_value in TABLE_K_VALUES}
    if (set(zip(comparison.model, comparison.k)) != expected_pairs
            or comparison.duplicated(['model', 'k', 'seed']).any()):
        raise ValueError(f'The model comparison requires all five models once per seed at K={TABLE_K_VALUES}')
    grouped = comparison.groupby(['model', 'k'])[['recall', 'ndcg']].mean()
    columns = ['Model']
    for k_value in TABLE_K_VALUES:
        columns += [f'Recall@{k_value}', f'NDCG@{k_value}']
    rows = []
    for model in MODEL_ORDER:
        row = {'Model': 'LightGCN' if model == 'GNN' else model}
        for k_value in TABLE_K_VALUES:
            row[f'Recall@{k_value}'] = float(grouped.loc[(model, k_value), 'recall'])
            row[f'NDCG@{k_value}'] = float(grouped.loc[(model, k_value), 'ndcg'])
        rows.append(row)
    test_size = float(plan.get('test_size', 0.1))
    train_percent, test_percent = int(round((1 - test_size) * 100)), int(round(test_size * 100))
    comparison_path = root / f'model_comparison_{train_percent}{test_percent}.csv'
    per_seed_path = root / f'model_comparison_{train_percent}{test_percent}_per_seed.csv'
    comparison_table = pd.DataFrame(rows, columns=columns)
    comparison_table.to_csv(comparison_path, index=False)
    per_seed = comparison[['model', 'seed', 'k', 'recall', 'ndcg', 'users_evaluated']].copy()
    per_seed.insert(1, 'Model', per_seed.model.replace({'GNN': 'LightGCN'}))
    per_seed['model'] = pd.Categorical(per_seed.model, MODEL_ORDER, ordered=True)
    per_seed.sort_values(['model', 'k', 'seed']).to_csv(per_seed_path, index=False)
    print(f'\nModel comparison, chronological {train_percent}:{test_percent}, K={TABLE_K_VALUES}')
    print(comparison_table.to_string(index=False))
    cis, paired = [], []
    groups = allmetrics[['condition', 'ablation', 'model']].drop_duplicates()
    for condition, ablation, model in groups.itertuples(index=False, name=None):
        selected = [value for (c, a, s, m), value in details.items() if (c, a, m) == (condition, ablation, model)]
        assert_same_users(selected)
        for metric in ['ndcg', 'recall']:
            values = np.stack([value[metric].to_numpy() for value in selected]).mean(axis=0)
            mean, low, high = paired_bootstrap(values)
            cis.append({'condition': condition, 'ablation': ablation, 'model': model, 'metric': metric,
                'mean': mean, 'ci_low': low, 'ci_high': high, 'users': len(values), 'seeds': len(selected),
                'ci_scope': 'users, conditional on the evaluated seeds; not a temporal confidence interval'})
    for (condition, ablation, seed, model), per in details.items():
        if model != 'Hybrid':
            continue
        comparisons = [('ALS+BPR', details[(condition, ablation, seed, 'ALS+BPR')])]
        primary = (primary_condition, 'main', seed, 'Hybrid')
        if (condition, ablation) != (primary_condition, 'main') and primary in details:
            comparisons.append(('primary Hybrid', details[primary]))
        for label, other in comparisons:
            assert_same_users([per, other])
            for metric in ['ndcg', 'recall']:
                mean, low, high = paired_bootstrap(per[metric], other[metric])
                paired.append({'condition': condition, 'ablation': ablation, 'seed': seed,
                    'comparison': 'Hybrid - ' + label, 'metric': metric,
                    'difference': mean, 'ci_low': low, 'ci_high': high})
    pd.DataFrame(cis).to_csv(root / 'confidence_intervals.csv', index=False)
    pd.DataFrame(paired).to_csv(root / 'paired_differences.csv', index=False)
    dump_json(root / 'experiment_manifest.json', {
        'runs': provenance, 'reference_id': allmetrics.reference_id.iloc[0],
        'original_truth_id': allmetrics.original_truth_id.iloc[0],
        'model_comparison': comparison_path.name, 'model_comparison_per_seed': per_seed_path.name,
        'files': {name: sha256(root / name) for name in (
            'experiment_metrics.csv', 'candidate_comparison.csv', 'seed_summary.csv',
            'confidence_intervals.csv', 'paired_differences.csv',
            comparison_path.name, per_seed_path.name, 'table1_dataset_statistics.csv',
            'table_s1_event_breakdown_reference.csv')}})
    return allmetrics


def run_suite(root, data_dir=DEFAULT_DATA, seeds=(42, 43, 44), iterations=256, bpr_epochs=128,
              epochs=200, device='cuda', include_ablations=False, factors=None, embedding_dim=None,
              legacy_v3=False, use_if=True):
    import torch
    torch.set_num_threads(1)
    root = Path(root)
    datasets = root / 'datasets'
    root.mkdir(parents=True, exist_ok=True)
    base = LEGACY_V3 if legacy_v3 else LOCKED
    primary_condition = 'if_events' if use_if else 'no_if_events'
    factors = base['als_factors'] if factors is None else factors
    embedding_dim = base['gnn_embedding_dim'] if embedding_dim is None else embedding_dim
    plan = {
        **base,
        'primary': primary_condition,
        'primary_condition': primary_condition,
        'primary_reason': (
            'Isolation Forest is enabled for the primary experiment; condition is fixed before test metrics.'
            if use_if else
            'Isolation Forest is disabled for the primary experiment; condition is fixed before test metrics.'
        ),
        'isolation_forest': bool(use_if),
        'gamma_k_values': PAPER_K_VALUES,
        'seeds': list(seeds), 'als_iterations': iterations, 'bpr_epochs': bpr_epochs,
        'gnn_max_epochs': epochs, 'als_factors': factors, 'gnn_embedding_dim': embedding_dim,
        'device': device, 'include_ablations': include_ablations, 'code_hash': code_hash(),
        'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else None,
        'locked_before_test_evaluation': True,
        'legacy_v3': legacy_v3,
        'test_size': 0.2 if legacy_v3 else float(base.get('test_size', 0.1)),
    }
    plan_path = root / 'experiment_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Existing experiment plan or source revision differs; use a new output directory')
    dump_json(plan_path, plan)
    if not (datasets / 'condition_manifest.json').exists():
        prepare_conditions(data_dir, datasets, test_size=float(plan['test_size']))
    condition_manifest = json.loads(
        (datasets / 'condition_manifest.json').read_text(encoding='utf-8'))
    if set(condition_manifest['conditions']) != set(CONDITIONS):
        raise ValueError('Prepared dataset conditions differ from this code revision; use a new output directory')
    write_dataset_statistics_tables(datasets, root, primary_condition)
    schedule = [(primary_condition, 'main', seed) for seed in seeds]
    if include_ablations:
        neighbor_condition = 'if_neighbors' if use_if else 'no_if_neighbors'
        alternate_if_condition = 'no_if_events' if use_if else 'if_events'
        schedule += [
            (alternate_if_condition, 'main', seeds[0]),
            (neighbor_condition, 'main', seeds[0]),
            (primary_condition, 'binary_pairwise', seeds[0]),
            (primary_condition, 'property_names', seeds[0]),
            (primary_condition, 'historical_metadata', seeds[0]),
        ]
    (root / 'runs').mkdir(exist_ok=True)
    hybrid_sweep = bool(plan.get('hybrid_sweep', True))
    for index, (condition, ablation, seed) in enumerate(schedule, 1):
        name = f'{condition}__{ablation}__s{seed}'
        out = root / 'runs' / name
        print(f'RUN {index}/{len(schedule)} {name}', flush=True)
        dump_json(root / 'progress.json', {'current': name, 'index': index, 'total': len(schedule),
            'completed': [item.name for item in (root / 'runs').iterdir() if (item / 'complete.json').exists()]})
        if (out / 'complete.json').exists():
            require_artifacts(out, 'evaluation')
            continue
        if not out.exists():
            clone_protocol(datasets / condition, out)
        elif not (out / 'validation' / 'manifest.json').exists():
            # Partial run directory without validation cannot select fusion/gamma.
            if (out / 'validation').exists():
                raise FileExistsError(f'Incomplete validation directory in {out}; remove it and resume')
            clone_protocol(datasets / condition / 'validation', out / 'validation')
        config = plan | {'condition': condition, 'ablation': ablation, 'seed': seed}
        config_path = out / 'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError(f'Run configuration changed for {name}; use a new output directory')
        dump_json(config_path, config)
        weighted = ablation != 'binary_pairwise'
        started = time.perf_counter()
        als_kwargs = dict(factors=plan['als_factors'], iterations=iterations, bpr_epochs=bpr_epochs,
            random_state=seed, device=device, behavior_weighted=weighted, top_k=plan['recommendation_depth'],
            ensemble_alpha=plan['ensemble_alpha'], alpha=plan['als_alpha'],
            time_decay=bool(plan.get('time_decay', False)),
            half_life_days=float(plan.get('half_life_days', 7.0)),
            negative_mode=plan.get('negative_mode', 'uniform'),
            regularization=float(plan.get('als_regularization', 0.03)))
        graph_mode = plan.get('gnn_graph_mode', 'bipartite')
        if ablation in {'property_names', 'historical_metadata'}:
            graph_mode = 'heterogeneous'
        gnn_kwargs = dict(epochs=epochs, embedding_dim=plan['gnn_embedding_dim'], layers=plan.get('gnn_layers', 2),
            seed=seed, device=device, behavior_weighted=weighted,
            property_mode='names' if ablation == 'property_names' else 'values',
            metadata_policy='history' if ablation == 'historical_metadata' else 'latest',
            propagation=plan.get('gnn_propagation', 'unweighted'),
            edge_dropout=float(plan.get('gnn_edge_dropout', 0.0)),
            negative_mode=plan.get('gnn_negative_mode', plan.get('negative_mode', 'uniform')),
            item_item_edges=bool(plan.get('gnn_item_item_edges', False)),
            time_decay=bool(plan.get('time_decay', False)),
            half_life_days=float(plan.get('half_life_days', 7.0)),
            graph_mode=graph_mode,
            loss_mode=plan.get('gnn_loss_mode', 'bpr'),
            n_negatives=int(plan.get('gnn_n_negatives', 1)),
            temperature=float(plan.get('gnn_temperature', 0.2)),
            min_epochs=int(plan.get('gnn_min_epochs', 1)),
            early_stopping_patience=int(plan.get('gnn_patience', 25)),
            batch_size=int(plan.get('gnn_batch_size', 65536)),
            learning_rate=float(plan.get('gnn_learning_rate', 0.001)))
        if not (out / 'validation' / 'als_artifacts.json').exists():
            ALSRecommender(out / 'validation', **als_kwargs).run()
        if not (out / 'gnn_artifacts.json').exists():
            GNNEmbeddingGenerator(out, **gnn_kwargs).run()
        if hybrid_sweep:
            selected = hybrid_config_sweep(
                out / 'validation', plan, plan['primary_k'], plan['gamma_k_values'])
            gamma = selected['gamma']
            als_weight = selected['als_weight']
            fusion = selected['fusion']
            rerank_m = selected['rerank_m']
            candidate_k = selected['candidate_k']
        else:
            gamma = gamma_sweep(out / 'validation', plan['gamma_grid'], plan['primary_k'],
                                plan['als_weight'], plan['candidate_k'], plan.get('fusion', 'minmax'),
                                plan.get('rerank_m'), 60, plan['gamma_k_values'],
                                plan.get('gamma_reference', []))
            als_weight = plan['als_weight']
            fusion = plan.get('fusion', 'minmax')
            rerank_m = plan.get('rerank_m')
            candidate_k = plan['candidate_k']
        if not (out / 'als_artifacts.json').exists():
            ALSRecommender(out, **als_kwargs).run()
        ReRanker(out, top_k=plan['recommendation_depth'], als_candidate_k=candidate_k,
                 als_weight=als_weight, inventory_gamma=gamma, fusion=fusion, rerank_m=rerank_m).run()
        TestSetEvaluator(out, plan['primary_k']).run(k_values=PAPER_K_VALUES)
        dump_json(out / 'complete.json', {'evaluation_hash': sha256(out / 'evaluation_artifacts.json'),
            'gamma': gamma, 'als_weight': als_weight, 'fusion': fusion, 'rerank_m': rerank_m,
            'candidate_k': candidate_k, 'seconds': time.perf_counter() - started})
        summarize(root)
    return summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results' / 'research_revision_paper')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--isolation-forest', choices=['on', 'off'], default='on',
                        help='Use Isolation Forest for the primary training condition (default: on)')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--with-ablations', action='store_true',
                        help='Also run IF/neighbor/binary/property/history ablations on the first seed')
    parser.add_argument('--legacy-v3', action='store_true',
                        help='Use the older revision_v3 settings instead of paper_v2')
    args = parser.parse_args()
    if not (args.data_dir / 'events.csv').exists():
        parser.error(f'events.csv was not found in {args.data_dir}')
    if args.prepare_only:
        datasets = args.output_dir / 'datasets'
        prepare_conditions(args.data_dir, datasets, test_size=0.2 if args.legacy_v3 else 0.1)
        condition = 'if_events' if args.isolation_forest == 'on' else 'no_if_events'
        write_dataset_statistics_tables(datasets, args.output_dir, condition)
    else:
        run_suite(args.output_dir, args.data_dir, args.seeds, device=args.device,
                  include_ablations=args.with_ablations, legacy_v3=args.legacy_v3,
                  use_if=args.isolation_forest == 'on')


if __name__ == '__main__':
    main()
