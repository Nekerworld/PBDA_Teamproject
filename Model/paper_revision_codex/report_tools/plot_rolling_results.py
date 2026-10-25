"""Manuscript figures and tables from committed rolling-origin summaries only.

Reads audit/results_v6 (or another results root with the same layout); no model is
trained and no number is typed in. Every output lists the SHA256 of its inputs in
figure_manifest.json so a figure can be traced to the exact CSV it was drawn from.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
MODELS = ['Popularity', 'Recent', 'RecentPop', 'ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']
COLORS = dict(zip(MODELS, ['#bdbdbd', '#969696', '#737373', '#64748b', '#2b8cbe', '#41ab5d', '#8064a2', '#d45b42']))
METRICS = {'ndcg': 'NDCG@50', 'recall': 'Recall@50'}
POLICY_LABEL = {'repeat': 'Repeat-allowed', 'novel': 'Novel-only'}


class Outputs:
    def __init__(self, results, out):
        self.results, self.out = Path(results), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = {'results_root': self.results.relative_to(HERE).as_posix(), 'sources': {}, 'outputs': []}

    def read(self, relative):
        path = self.results / relative
        self.manifest['sources'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    def figure(self, fig, name):
        fig.tight_layout()
        stable = {'pdf': {'CreationDate': None}, 'svg': {'Date': None}, 'png': {}}
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(self.out / f'{name}.{ext}', dpi=300, bbox_inches='tight', metadata=stable[ext])
        plt.close(fig)
        self.manifest['outputs'].append(f'{name}.png/.pdf/.svg')

    def table(self, frame, name, floatfmt='.4f'):
        frame.to_csv(self.out / f'{name}.csv', index=False)
        lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '---|' * len(frame.columns)]
        for row in frame.itertuples(index=False):
            lines.append('| ' + ' | '.join(format(v, floatfmt) if isinstance(v, float) else str(v) for v in row) + ' |')
        (self.out / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        self.manifest['outputs'].append(f'{name}.csv/.md')

    def close(self):
        (self.out / 'figure_manifest.json').write_text(json.dumps(self.manifest, indent=2), encoding='utf-8')


def folds_of(results):
    return sorted(p.name for p in Path(results).iterdir() if (p / 'report' / 'test_summary.csv').exists())


def model_comparison(o, folds):
    summary = pd.concat([o.read(f'{f}/report/test_summary.csv') for f in folds])
    k50 = summary.loc[summary.k == 50]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    width = .8 / len(MODELS)
    x = np.arange(len(folds))
    for row, policy in enumerate(('repeat', 'novel')):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for i, model in enumerate(MODELS):
                part = k50.loc[(k50.policy == policy) & (k50.model == model)].set_index('fold').reindex(folds)
                ax.bar(x + (i - len(MODELS) / 2 + .5) * width, part[f'{metric}_mean'], width,
                       yerr=part[f'{metric}_std'], color=COLORS[model], label=model, capsize=1.5,
                       error_kw={'elinewidth': .6})
            ax.set_title(f'{POLICY_LABEL[policy]}: {METRICS[metric]}')
            ax.set_xticks(x, folds)
            ax.grid(axis='y', alpha=.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(MODELS), fontsize=8, bbox_to_anchor=(.5, -.03))
    seeds = sorted(pd.concat([o.read(f'{f}/report/test_metrics.csv') for f in folds]).seed.unique())
    title = f'Sealed test, mean of seeds {"/".join(map(str, seeds))} (error bar = seed SD)'
    if 'F1_hist' in folds:
        title += '; F1 window was exposed in v1-v5'
    fig.suptitle(title)
    o.figure(fig, 'model_comparison')
    table = k50.loc[k50.model.isin(MODELS)].pivot_table(index=['policy', 'model'], columns='fold',
                                                         values=['ndcg_mean', 'recall_mean'])
    table.columns = [f'{m.replace("_mean", "")}_{f}' for m, f in table.columns]
    table = table.reset_index()
    table['model'] = pd.Categorical(table.model, MODELS)
    o.table(table.sort_values(['policy', 'model']).assign(model=lambda t: t.model.astype(str)), 'table_test_k50')
    topk = summary.loc[summary.model.isin(MODELS)]
    fig, axes = plt.subplots(2, len(folds), figsize=(4 * len(folds), 6.5), squeeze=False)
    for col, fold in enumerate(folds):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for model in MODELS:
                part = topk.loc[(topk.fold == fold) & (topk.policy == 'repeat') & (topk.model == model)].sort_values('k')
                ax.plot(part.k, part[f'{metric}_mean'], marker='o', ms=3, color=COLORS[model], label=model)
            ax.set_title(f'{fold}, repeat: {METRICS[metric].split("@")[0]}@K')
            ax.set_xlabel('K')
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    o.figure(fig, 'topk_comparison')


def paired_differences(o, folds):
    tests = pd.concat([o.read(f'{f}/report/paired_tests.csv').assign(fold=f) for f in folds])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    comparisons = [f'Hybrid - {m}' for m in MODELS if m != 'Hybrid']
    for row, policy in enumerate(('repeat', 'novel')):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for j, fold in enumerate(folds):
                part = tests.loc[(tests.fold == fold) & (tests.policy == policy) & (tests.metric == metric)]
                part = part.set_index('comparison').reindex(comparisons)
                y = np.arange(len(comparisons)) + (j - (len(folds) - 1) / 2) * .25
                ax.errorbar(part.difference, y, xerr=[part.difference - part.ci_low, part.ci_high - part.difference],
                            fmt='none', ecolor=f'C{j}', elinewidth=1)
                sig = part.significant_holm.astype(bool).to_numpy()
                ax.scatter(part.difference[sig], y[sig], color=f'C{j}', s=18, label=f'{fold} (filled: Holm p<.05)')
                ax.scatter(part.difference[~sig], y[~sig], facecolor='white', edgecolor=f'C{j}', s=18)
            ax.axvline(0, color='black', lw=.8)
            ax.set_yticks(np.arange(len(comparisons)), comparisons)
            ax.set_title(f'{POLICY_LABEL[policy]}: {METRICS[metric]} difference, 95% paired bootstrap CI')
            ax.grid(axis='x', alpha=.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(folds), fontsize=8, bbox_to_anchor=(.5, -.03))
    o.figure(fig, 'paired_differences')
    keep = tests.loc[tests.comparison.isin(['Hybrid - GNN', 'Hybrid - ALS+BPR', 'Hybrid - RecentPop'])]
    o.table(keep[['fold', 'policy', 'comparison', 'metric', 'difference', 'ci_low', 'ci_high', 'p_value', 'p_holm',
                  'claim', 'users']].sort_values(['comparison', 'policy', 'metric', 'fold']), 'table_paired_tests')


def posthoc(o, fold):
    if not (o.results / fold / 'report' / 'posthoc_gamma.csv').exists():
        return
    gamma = o.read(f'{fold}/report/posthoc_gamma.csv')
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(gamma.gamma, gamma.ndcg, marker='o', label='NDCG@50', color='#d45b42')
    ax.plot(gamma.gamma, gamma.recall, marker='s', label='Recall@50', color='#2b8cbe')
    ax.set_xlabel('Inventory weight gamma (1 = no weighting)')
    ax.set_ylabel('Metric')
    ax2 = ax.twinx()
    ax2.plot(gamma.gamma, gamma.available_share_top50, ls='--', color='gray', label='In-stock share of top-50')
    ax2.set_ylabel('In-stock share of top-50')
    ax.legend(loc='lower left', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)
    ax.set_title(f'{fold} post-hoc (seed 42, never used for selection)', fontsize=10)
    o.figure(fig, 'gamma_sensitivity')
    o.table(gamma[['gamma', 'ndcg', 'recall', 'hit_users', 'users', 'available_share_top50']], 'table_gamma_posthoc')
    depth = o.read(f'{fold}/report/posthoc_candidate_depth.csv')
    label = depth.candidate_k.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(label, depth.ndcg, marker='o', label='NDCG@50', color='#d45b42')
    axes[0].plot(label, depth.recall, marker='s', label='Recall@50', color='#2b8cbe')
    axes[0].plot(label, depth.candidate_recall_of_reachable_truth, marker='^', label='Candidate recall of reachable truth',
                 color='gray')
    axes[0].set_xlabel('Candidate count')
    axes[0].legend(fontsize=8)
    axes[1].plot(label, depth.total_ms_per_user, marker='o', label='Hybrid path (MF + candidates + GNN + fusion)')
    axes[1].axhline(depth.gnn_full_catalog_ms_per_user.iloc[0], color='#8064a2', ls='--',
                    label='GNN alone, full catalog')
    axes[1].set_ylabel('ms per user')
    axes[1].set_xlabel('Candidate count')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{fold} post-hoc candidate depth (seed 42, never used for selection)', fontsize=10)
    o.figure(fig, 'candidate_depth')
    o.table(depth[['candidate_k', 'candidate_share_of_catalog', 'ndcg', 'recall', 'candidate_recall_of_reachable_truth',
                   'total_ms_per_user', 'gnn_full_catalog_ms_per_user', 'peak_python_mb']], 'table_candidate_depth')


def dataset_tables(o, folds):
    rows = []
    for fold in folds:
        pops = o.read(f'{fold}/report/populations.csv')
        for record in pops.to_dict('records'):
            rows.append({'fold': fold, **record})
    o.table(pd.DataFrame(rows), 'table_populations', floatfmt='.0f')
    if (o.results / folds[0] / 'dataset_statistics.csv').exists():
        stats = o.read(f'{folds[0]}/dataset_statistics.csv')
        o.table(stats[['stage', 'events', 'users', 'items', 'unique_user_item_pairs', 'view_events', 'addtocart_events',
                       'transaction_events']], 'table_dataset_statistics', floatfmt='.0f')


def pipeline(o):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis('off')
    texts = ['Chronological split\n(rolling origins; inner\nvalidation for selection)',
             'ALS + BPR min-max scores\ncandidates: alpha-blend top-n\nor ALS/BPR top-n/2 union',
             'User-item bipartite graph\nLightGCN propagation (mean\nof layers), InfoNCE, 32 neg.',
             'Fusion on candidates\nomega * MF + (1 - omega) * GNN\n(min-max or RRF)',
             'Inventory weight\nx gamma if available\nbefore cutoff -> Top-K']
    width, gap = .172, .03
    for i, text in enumerate(texts):
        x = .01 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, .3), width, .45, boxstyle='round,pad=.01', fc='#edf4f8', ec='#487b96',
                                    transform=ax.transAxes))
        ax.text(x + width / 2, .525, text, ha='center', va='center', transform=ax.transAxes, fontsize=8.5)
        if i:
            ax.annotate('', xy=(x - .004, .525), xytext=(x - gap + .004, .525), xycoords='axes fraction',
                        arrowprops={'arrowstyle': '->', 'color': '#487b96'})
    ax.text(.5, .1, 'Pair weight w = view 1 / cart 6 / purchase 12, decayed with a 7-day half-life: ALS confidence '
            '1 + alpha * w;\nBPR loss weight w (popularity negatives); GNN InfoNCE loss weight w (unweighted '
            'propagation, edge dropout 0.1, uniform negatives).', ha='center', va='center', transform=ax.transAxes,
            fontsize=8.5)
    ax.text(.5, -.08, 'No Isolation Forest in the final model. All parameters chosen on inner validation; '
            'final models refit with fixed epochs on the full pre-test window.', ha='center', transform=ax.transAxes,
            fontsize=8.5)
    o.figure(fig, 'pipeline')


def main(results, out):
    o = Outputs(results, out)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'rolling-origin'})
    folds = folds_of(results)
    model_comparison(o, folds)
    paired_differences(o, folds)
    posthoc(o, folds[0])
    dataset_tables(o, folds)
    pipeline(o)
    o.close()
    print(json.dumps(o.manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'audit' / 'results_v6')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.results.resolve(), args.output.resolve())
