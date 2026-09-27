"""Create paper figures from one verified experiment; never use embedded results."""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .data_protocol import DEFAULT_PROCESSED, require_protocol, require_artifacts, sha256, dump_json
except ImportError:
    from data_protocol import DEFAULT_PROCESSED, require_protocol, require_artifacts, sha256, dump_json

MODEL_ORDER = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']
COLORS = ['#4979a8', '#76a5b5', '#d2a84a', '#917ca9', '#c06454']


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


def generate_experiment_figures(root, output_dir=None, dpi=600):
    """Figures for a completed multi-run directory. Every number is read from its CSV."""
    root = Path(root)
    manifest_path = root / 'experiment_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Experiment summary is missing; finish experiments.py before drawing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Experiment summary changed after it was sealed: {name}')
    output = Path(output_dir) if output_dir else root / 'figures'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new empty figure directory: {output}')
    output.mkdir(parents=True)
    generated = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
            generated.append(f'{name}.{ext}')
        plt.close(fig)

    condition_manifest = json.loads((root / 'datasets' / 'condition_manifest.json').read_text(encoding='utf-8'))
    comparison_path = root / 'datasets' / 'filter_comparison.csv'
    if sha256(comparison_path) != condition_manifest['filter_comparison_hash']:
        raise ValueError('Filter comparison does not match the sealed condition manifest')
    comparison = pd.read_csv(comparison_path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fields = [('transaction_events', 'Training purchases'), ('fixed_evaluation_users', 'Fixed reference users'),
              ('mean_original_truth_retention', 'Mean original-truth retention')]
    for ax, (field, title) in zip(axes, fields):
        bars = ax.bar(comparison.condition, comparison[field], color=COLORS[:len(comparison)])
        ax.bar_label(bars, fmt='%.3g', padding=3, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)
    fig.suptitle('Predeclared filter conditions on one fixed reference')
    save(fig, 'filter_comparison')

    intervals = pd.read_csv(root / 'confidence_intervals.csv')
    primary = intervals.loc[(intervals.condition == 'no_if_events') & (intervals.ablation == 'main') & (intervals.metric == 'ndcg')]
    primary = primary.set_index('model').loc[MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(primary))
    ax.errorbar(primary['mean'], y, xerr=[primary['mean'] - primary.ci_low, primary.ci_high - primary['mean']],
                fmt='o', color=COLORS[0], capsize=3)
    ax.set_yticks(y, primary.index)
    ax.set_xlabel('Original-truth NDCG, user bootstrap conditional on the finished seeds')
    ax.set_title(f"Primary condition, {int(primary.seeds.iloc[0])} seed(s), {int(primary.users.iloc[0]):,} users")
    save(fig, 'confidence_intervals')

    metrics = pd.read_csv(root / 'experiment_metrics.csv')
    hybrid = metrics.loc[metrics.model == 'Hybrid']
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = hybrid.condition + '\n' + hybrid.ablation + '\nseed ' + hybrid.seed.astype(str)
    bars = ax.bar(range(len(hybrid)), hybrid.ndcg, color=COLORS[4])
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=7)
    ax.set_xticks(range(len(hybrid)), labels, fontsize=7)
    ax.set_ylabel(f"Hybrid original NDCG@{int(hybrid.k.iloc[0])}")
    ax.set_title('Finished runs; ablations are not a second test-set search')
    save(fig, 'run_overview')

    curves = sorted((root / 'runs').glob('no_if_events__main__s*/validation/gamma_validation.csv'))
    if not curves:
        raise ValueError('Validation gamma curves are missing')
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for path in curves:
        frame = pd.read_csv(path).sort_values('gamma')
        if set(frame.role) != {'validation'} or set(frame.scope) != {'original'}:
            raise ValueError(f'Gamma file is not validation original truth: {path}')
        ax.plot(frame.gamma, frame.ndcg, marker='o', label=path.parents[1].name)
    ax.set_xlabel('Inventory gamma')
    ax.set_ylabel('Validation original NDCG')
    ax.set_title('Gamma chosen on inner validation, not the outer test')
    ax.legend(fontsize=7)
    save(fig, 'gamma_validation')
    dump_json(output / 'experiment_figure_manifest.json', {
        'experiment_reference_id': manifest['reference_id'], 'figures': generated, 'dpi': dpi,
        'sources': {name: sha256(root / name) for name in manifest['files']}})
    print(f'Created {len(generated)} experiment-figure files in {output.resolve()}')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument('--experiment-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
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
