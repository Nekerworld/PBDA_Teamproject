"""Create paper figures from one verified experiment; never use embedded results."""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

try:
    from .data_protocol import DEFAULT_PROCESSED, require_protocol, require_artifacts, sha256, dump_json
except ImportError:
    from data_protocol import DEFAULT_PROCESSED, require_protocol, require_artifacts, sha256, dump_json

MODEL_ORDER = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']
COLORS = ['#4979a8', '#76a5b5', '#d2a84a', '#917ca9', '#c06454']
TABLE_K_VALUES = [10, 30, 50]
K_COLORS = {10: '#4979a8', 30: '#d2a84a', 50: '#c06454'}
PAPER_VIZ_DIR = Path(__file__).resolve().parent / 'viz'


def generate_figures(processed_dir=DEFAULT_PROCESSED, output_dir=None, dpi=600):
    directory = Path(processed_dir)
    manifest = require_protocol(directory)
    stats = pd.read_csv(directory / 'dataset_statistics.csv').set_index('stage')
    evaluation = None
    # A stale evaluation is an error, not a reason to draw old or fabricated values.
    if (directory / 'evaluation_artifacts.json').exists():
        evaluation = require_artifacts(directory, 'evaluation')
    elif (directory / 'ranking_metrics.csv').exists():
        raise ValueError('Evaluation provenance is missing; rerun model.py evaluate')
    output = Path(output_dir) if output_dir else directory / 'figures'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new empty figure directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    sources, generated = {}, []

    def record(name):
        sources[name] = sha256(directory / name)

    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
            generated.append(f'{name}.{ext}')
        plt.close(fig)

    record('dataset_statistics.csv')
    stages = ['train_before_filter', 'train_after_if', 'train_final']
    labels = ['Before filtering', 'After Isolation Forest', 'After iterative core']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, field in zip(axes, ['events', 'users', 'items']):
        values = stats.loc[stages, field].to_numpy(dtype=int)
        bars = ax.bar(range(3), values, color=COLORS[:3])
        ax.set_xticks(range(3), labels, rotation=25, ha='right', fontsize=8)
        ax.set_title(field.capitalize())
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 4))
        ax.bar_label(bars, labels=[f'{v:,}' for v in values], fontsize=8, padding=3)
        ax.margins(y=.2)
    fig.suptitle('Training data retained at each filtering stage')
    save(fig, 'filtering_counts')

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for idx, (field, label) in enumerate([('view_events', 'View'), ('addtocart_events', 'Add to cart'),
                                         ('transaction_events', 'Transaction')]):
        values = stats.loc[stages, field].to_numpy(dtype=int)
        bars = ax.bar(x + (idx-1)*.25, values, .25, label=label, color=COLORS[idx])
        ax.bar_label(bars, labels=[f'{v:,}' for v in values], fontsize=7, padding=3)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylabel('Event count (symlog; zero shown)')
    ax.set_xticks(x, labels)
    ax.set_title('Behavior retained after filtering')
    ax.legend(loc='upper right', fontsize=8)
    ax.margins(y=.25)
    save(fig, 'filtering_behavior')

    coverage = pd.read_csv(directory / 'test_exclusion_summary.csv')
    record('test_exclusion_summary.csv')
    fig, ax = plt.subplots(figsize=(9, max(3.5, .38*len(coverage))))
    bars = ax.barh(coverage.reason, coverage.events, color=COLORS[0])
    ax.bar_label(bars, labels=[f'{v:,}' for v in coverage.events], padding=3, fontsize=8)
    ax.set_xlabel('Test events')
    ax.set_title('Warm-start evaluation eligibility (user status / item status)')
    ax.margins(x=.2)
    save(fig, 'test_coverage')

    retention = pd.read_csv(directory / 'truth_retention.csv')
    record('truth_retention.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(retention.original_retention, bins=10, range=(0, 1), color=COLORS[0])
    axes[0].set_title('Original positive-item retention')
    axes[1].hist(retention.reference_retention, bins=10, range=(0, 1), color=COLORS[3])
    axes[1].set_title('Reference warm-item retention')
    for ax in axes:
        ax.set_xlabel('Fraction of fixed positives still conditional')
        ax.set_ylabel('Reference users')
    fig.suptitle(f"Fixed reference users: {len(retention):,}; train-retained {int(retention.train_user_retained.sum()):,}")
    save(fig, 'truth_retention')

    skipped = []
    if evaluation:
        record('ranking_metrics.csv')
        record('topk_metrics.csv')
        record('timing_metrics.csv')
        base = pd.read_csv(directory / 'ranking_metrics.csv')
        topk = pd.read_csv(directory / 'topk_metrics.csv')
        timing = pd.read_csv(directory / 'timing_metrics.csv')
        for frame in [base, topk, timing]:
            if set(frame.protocol_id) != {manifest['protocol_id']}:
                raise ValueError('Figure input contains mixed experiment protocols')
        if 'scope' in base.columns and set(base.scope) != {'original'}:
            raise ValueError('The primary table must use the fixed original-truth scope')
        if set(base.model) != set(MODEL_ORDER) or len(base) != 5 or base.k.nunique() != 1:
            raise ValueError('The baseline table must contain five models at a single K')
        base = base.set_index('model').loc[MODEL_ORDER]
        k = int(base.k.iloc[0])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, metric, title in zip(axes, ['ndcg', 'recall'], ['NDCG', 'Recall']):
            bars = ax.bar(MODEL_ORDER, base[metric], color=COLORS)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
            ax.set_ylabel(f'{title}@{k}')
            ax.set_ylim(0, min(1.1, max(.1, float(base[metric].max())*1.22)))
            ax.tick_params(axis='x', labelsize=8)
        fig.suptitle(f'Original-truth evaluation: {int(base.users_evaluated.iloc[0]):,} fixed reference users')
        save(fig, 'baseline_comparison')

        if 'scope' in topk.columns:
            topk = topk.loc[topk.scope == 'original'].copy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, metric, title in zip(axes, ['ndcg', 'recall'], ['NDCG@K', 'Recall@K']):
            for name, color in zip(MODEL_ORDER, COLORS):
                frame = topk.loc[topk.model == name].sort_values('k')
                ax.plot(frame.k, frame[metric], marker='o', label=name, color=color, markersize=3)
            ax.set_xlabel('K')
            ax.set_ylabel(title)
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        save(fig, 'topk_comparison')
        if (directory / 'candidate_metrics.csv').exists():
            record('candidate_metrics.csv')
            candidates = pd.read_csv(directory / 'candidate_metrics.csv')
            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.bar(candidates.scope, candidates.recall, color=COLORS[:len(candidates)])
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
            ax.set_ylim(0, min(1.1, max(.1, float(candidates.recall.max()) * 1.25)))
            ax.set_ylabel(f"Candidate Recall@{int(candidates.candidate_k.iloc[0])}")
            ax.set_title('ALS+BPR candidate coverage of each truth scope')
            save(fig, 'candidate_recall')

        frame = timing.set_index('model').loc[MODEL_ORDER]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, field, title in zip(axes, ['training_seconds', 'recommendation_seconds'],
                                   ['Training incl. selection/refit', 'Batch recommendation generation']):
            ax.bar(MODEL_ORDER, frame[field], color=COLORS)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel('Seconds')
            ax.tick_params(axis='x', labelsize=8)
        save(fig, 'measured_runtime')
    else:
        skipped.append('Performance figures: training and evaluation have not been completed.')
    for message in skipped:
        logging.warning(message)
    dump_json(output / 'figure_manifest.json', {'protocol_id': manifest['protocol_id'],
        'input_directory': str(directory.resolve()), 'source_hashes': sources, 'figures': generated,
        'dpi': dpi, 'skipped': skipped, 'data_quality_warnings': manifest.get('data_quality_warnings', []),
        'evaluation_artifact_hash': sha256(directory / 'evaluation_artifacts.json') if evaluation else None})
    # Keep the exact numeric inputs beside the paper figures.
    for name in sources:
        (output / name).write_bytes((directory / name).read_bytes())
    print(f'Created {len(generated)} files in {output.resolve()}')
    return output


def _draw_model_table(table, seeds, split_label):
    """Booktabs-style table; values in percent, best value per column in bold."""
    metric_columns = [column for column in table.columns if column != 'Model']
    cells = [[row['Model'], *[f'{100 * row[column]:.2f}' for column in metric_columns]]
             for _, row in table.iterrows()]
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(metric_columns), 0.42 * (len(table) + 2)))
    ax.axis('off')
    grid = ax.table(cellText=cells, colLabels=['Model', *metric_columns], loc='center', cellLoc='center')
    grid.auto_set_font_size(False)
    grid.set_fontsize(9)
    grid.scale(1, 1.45)
    last_row = len(table)
    for (row, _), cell in grid.get_celld().items():
        cell.visible_edges = 'TB' if row == 0 else ('B' if row == last_row else '')
        cell.set_linewidth(0.8)
        if row == 0:
            cell.get_text().set_fontweight('bold')
    for column_index, column in enumerate(metric_columns, start=1):
        best = int(np.argmax(table[column].to_numpy()))
        grid[(best + 1, column_index)].get_text().set_fontweight('bold')
    ax.set_title(f'Top-K recommendation performance (%), chronological {split_label} test, '
                 f'mean of {seeds} seed(s); best in bold', fontsize=9)
    return fig


def _draw_gamma_sensitivity(summary, selected_gammas, grid_max):
    seeds = int(summary.seeds.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [('recall', 'Recall@K'), ('ndcg', 'NDCG@K'), ('topk_available_share', 'In-stock items in top-K (%)')]
    reference_max = float(summary.gamma.max())
    for ax, (metric, label) in zip(axes, panels):
        scale = 100 if metric == 'topk_available_share' else 1
        for k_value in TABLE_K_VALUES:
            frame = summary.loc[summary.k == k_value].sort_values('gamma')
            if frame.empty:
                continue
            color = K_COLORS[k_value]
            main = frame.loc[~frame.reference]
            reference = frame.loc[frame.reference]
            mean, std = main[f'{metric}_mean'] * scale, main[f'{metric}_std'] * scale
            ax.plot(main.gamma, mean, marker='o', markersize=3, color=color, label=f'K={k_value}')
            if seeds > 1:
                ax.fill_between(main.gamma, mean - std, mean + std, color=color, alpha=.15, linewidth=0)
            if not reference.empty:
                bridge = pd.concat([main.tail(1), reference])
                ax.plot(bridge.gamma, bridge[f'{metric}_mean'] * scale, linestyle=':', color=color)
                ax.scatter(reference.gamma, reference[f'{metric}_mean'] * scale,
                           facecolors='white', edgecolors=color, zorder=3, s=22)
        if metric == 'topk_available_share':
            pool = float(summary.candidate_available_share_mean.mean()) * 100
            ax.axhline(pool, color='black', linestyle=':', linewidth=1,
                       label=f'Candidate pool ({pool:.1f}%)')
        ax.axvline(1.0, color='grey', linestyle='--', linewidth=1, label='γ = 1 (no inventory weight)')
        for index, gamma in enumerate(sorted(set(selected_gammas))):
            ax.axvline(gamma, color='#2e7d32', linestyle='-.', linewidth=1,
                       label='Selected γ' if index == 0 else None)
        if reference_max > grid_max:
            ax.axvspan(grid_max * 1.15, reference_max * 1.15, color='0.93', zorder=0)
            ax.text(np.sqrt(grid_max * reference_max), 0.98, 'reference', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7, color='0.4')
        ax.set_xscale('log')
        ticks = [tick for tick in [0.2, 0.4, 0.6, 1, 2, 3, 5, 10] if tick <= reference_max * 1.01]
        ax.set_xticks(ticks, [f'{tick:g}' for tick in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Inventory γ (log scale)')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc='best')
    axes[2].legend(fontsize=7, loc='best')
    selected_text = ', '.join(f'{gamma:g}' for gamma in selected_gammas)
    fig.suptitle(f'Inventory γ sensitivity of the Hybrid model on inner validation '
                 f'({seeds} seed(s); selected γ: {selected_text})', fontsize=10)
    return fig


def generate_experiment_figures(root, output_dir=None, dpi=600):
    """Paper outputs for a completed multi-run directory. Every number is read from its CSV.

    Writes the model comparison table, preprocessing statistics, and gamma sensitivity,
    each with the CSV it was drawn from.
    """
    root = Path(root)
    manifest_path = root / 'experiment_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Experiment summary is missing; finish experiments.py before drawing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    plan = json.loads((root / 'experiment_plan.json').read_text(encoding='utf-8'))
    primary_condition = plan['primary_condition']
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Experiment summary changed after it was sealed: {name}')
    if 'model_comparison' not in manifest:
        raise ValueError('Experiment summary predates the K=10/30/50 table; rerun experiments.py')
    output = Path(output_dir) if output_dir else PAPER_VIZ_DIR
    output.mkdir(parents=True, exist_ok=True)
    generated = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
            generated.append(f'{name}.{ext}')
        plt.close(fig)

    def write_csv(frame, name):
        frame.to_csv(output / name, index=False)
        generated.append(name)

    test_size = float(plan.get('test_size', 0.1))
    split_label = f'{int(round((1 - test_size) * 100))}:{int(round(test_size * 100))}'

    table = pd.read_csv(root / manifest['model_comparison'])
    per_seed = pd.read_csv(root / manifest['model_comparison_per_seed'])
    write_csv(table, 'model_comparison.csv')
    write_csv(per_seed, 'model_comparison_per_seed.csv')
    save(_draw_model_table(table, int(per_seed.seed.nunique()), split_label), 'model_comparison')

    condition_manifest = json.loads((root / 'datasets' / 'condition_manifest.json').read_text(encoding='utf-8'))
    filter_path = root / 'datasets' / 'filter_comparison.csv'
    if sha256(filter_path) != condition_manifest['filter_comparison_hash']:
        raise ValueError('Filter comparison does not match the sealed condition manifest')
    stage_stats_path = root / 'datasets' / 'filter_stage_statistics.csv'
    if sha256(stage_stats_path) != condition_manifest['filter_stage_statistics_hash']:
        raise ValueError('Filter stage statistics do not match the sealed condition manifest')
    stage_stats = pd.read_csv(stage_stats_path)
    required_stats = {'condition', 'stage', 'events', 'users', 'items',
                      'unique_user_item_pairs', 'sparsity_percent'}
    if not required_stats <= set(stage_stats.columns):
        raise ValueError('Filter-stage table is missing dataset size or sparsity fields')
    write_csv(pd.read_csv(root / 'table1_dataset_statistics.csv'), 'preprocessing_dataset_statistics.csv')
    write_csv(pd.read_csv(root / 'table_s1_event_breakdown_reference.csv'), 'preprocessing_event_breakdown.csv')
    write_csv(stage_stats, 'preprocessing_filter_stage_statistics.csv')
    write_csv(pd.read_csv(filter_path), 'preprocessing_filter_comparison.csv')

    runs = sorted((root / 'runs').glob(f'{primary_condition}__main__s*'))
    curves, selections = [], []
    required_gamma = {'gamma', 'reference', 'k', 'ndcg', 'recall',
                      'topk_available_share', 'candidate_available_share'}
    for run in runs:
        path = run / 'validation' / 'gamma_validation.csv'
        if not (run / 'complete.json').exists() or not path.exists():
            continue
        frame = pd.read_csv(path)
        if set(frame.role) != {'validation'} or set(frame.scope) != {'original'}:
            raise ValueError(f'Gamma file is not validation original truth: {path}')
        if not required_gamma <= set(frame.columns):
            raise ValueError(f'Gamma file predates the in-stock share columns; rerun experiments.py: {path}')
        seed = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))['seed']
        frame.insert(0, 'seed', seed)
        curves.append(frame)
        selection = json.loads((run / 'validation' / 'gamma_selection.json').read_text(encoding='utf-8'))
        selections.append({'seed': seed, 'selected_gamma': selection['selected_gamma'],
                           **{key: selection.get(key) for key in
                              ('als_weight', 'fusion', 'rerank_m', 'candidate_k', 'validation_ndcg')}})
    if not curves:
        raise ValueError('Validation gamma curves are missing')
    gamma_per_seed = pd.concat(curves, ignore_index=True)
    gamma_summary = gamma_per_seed.groupby(['gamma', 'reference', 'k']).agg(
        recall_mean=('recall', 'mean'), recall_std=('recall', 'std'),
        ndcg_mean=('ndcg', 'mean'), ndcg_std=('ndcg', 'std'),
        topk_available_share_mean=('topk_available_share', 'mean'),
        topk_available_share_std=('topk_available_share', 'std'),
        candidate_available_share_mean=('candidate_available_share', 'mean'),
        seeds=('seed', 'nunique')).reset_index()
    gamma_summary = gamma_summary.fillna({column: 0.0 for column in gamma_summary.columns if column.endswith('_std')})
    selection_frame = pd.DataFrame(selections).sort_values('seed')
    write_csv(gamma_per_seed, 'gamma_sensitivity_per_seed.csv')
    write_csv(gamma_summary, 'gamma_sensitivity.csv')
    write_csv(selection_frame, 'gamma_selection.csv')
    grid_max = float(gamma_summary.loc[~gamma_summary.reference, 'gamma'].max())
    save(_draw_gamma_sensitivity(gamma_summary, selection_frame.selected_gamma.tolist(), grid_max),
         'gamma_sensitivity')

    dump_json(output / 'paper_figure_manifest.json', {
        'experiment_dir': str(root.resolve()), 'experiment_reference_id': manifest['reference_id'],
        'split': split_label, 'primary_condition': primary_condition, 'dpi': dpi, 'files': generated,
        'sources': {**{name: sha256(root / name) for name in manifest['files']},
                    'datasets/filter_comparison.csv': sha256(filter_path),
                    'datasets/filter_stage_statistics.csv': sha256(stage_stats_path)}})
    print(f'Created {len(generated)} paper files in {output.resolve()}')
    print(table.to_string(index=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument('--experiment-dir', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='Default for --experiment-dir: Model/paper_revision/viz')
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error('dpi must be at least 72 (600 is recommended for submission)')
    if args.experiment_dir:
        generate_experiment_figures(args.experiment_dir, args.output_dir, args.dpi)
    else:
        generate_figures(args.processed_dir, args.output_dir, args.dpi)


if __name__ == '__main__':
    main()
