"""Paper figures comparing revision v3, improvement, and paper v2.

v3 seed rows are no longer on disk. Means and sample standard deviations are the
sealed values in revision_v3_results_report.md. The other two protocols are
read from experiment_metrics.csv.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid']
COLORS = ['#4979a8', '#76a5b5', '#d2a84a', '#917ca9', '#c06454']
PROTOCOLS = ['Revision v3', 'Improvement', 'Paper v2']
# Sealed original-truth K=50, three seeds, sample standard deviation.
V3 = {
    'ALS': (0.06746, 0.00030, 0.11834, 0.00175),
    'BPR': (0.10178, 0.00124, 0.13536, 0.00279),
    'ALS+BPR': (0.09944, 0.00049, 0.14583, 0.00079),
    'GNN': (0.02233, 0.00103, 0.04140, 0.00360),
    'Hybrid': (0.05817, 0.00086, 0.11217, 0.00169),
}


def _summary(frame, protocol):
    rows = []
    for model in MODEL_ORDER:
        part = frame.loc[frame.model == model]
        rows.append({
            'protocol': protocol, 'model': model,
            'ndcg_mean': float(part.ndcg.mean()), 'ndcg_std': float(part.ndcg.std(ddof=1)),
            'recall_mean': float(part.recall.mean()), 'recall_std': float(part.recall.std(ddof=1)),
            'seeds': int(part.seed.nunique()),
        })
    return pd.DataFrame(rows)


def load_table(data_root):
    data_root = Path(data_root)
    rows = []
    for model, (ndcg, ndcg_std, recall, recall_std) in V3.items():
        rows.append({
            'protocol': 'Revision v3', 'model': model,
            'ndcg_mean': ndcg, 'ndcg_std': ndcg_std,
            'recall_mean': recall, 'recall_std': recall_std, 'seeds': 3,
            'source': 'revision_v3_results_report.md',
        })
    v3 = pd.DataFrame(rows)
    pieces = [v3]
    for folder, label in [
        ('research_revision_improve', 'Improvement'),
        ('research_revision_paper', 'Paper v2'),
    ]:
        path = data_root / folder / 'experiment_metrics.csv'
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame.condition == 'no_if_events') & (frame.ablation == 'main')
            & (frame.scope == 'original') & (frame.k == 50)
        ]
        summary = _summary(frame, label)
        summary['source'] = str(path)
        pieces.append(summary)
    table = pd.concat(pieces, ignore_index=True)
    table['protocol'] = pd.Categorical(table.protocol, PROTOCOLS, ordered=True)
    table['model'] = pd.Categorical(table.model, MODEL_ORDER, ordered=True)
    return table.sort_values(['protocol', 'model']).reset_index(drop=True)


def _grouped(ax, table, mean_col, std_col, ylabel):
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for offset, protocol, hatch in zip(
        [-width, 0, width], PROTOCOLS, ['///', '...', '']
    ):
        part = table.loc[table.protocol == protocol].set_index('model').loc[MODEL_ORDER]
        ax.bar(
            x + offset, part[mean_col], width, yerr=part[std_col],
            label=protocol, color=COLORS[PROTOCOLS.index(protocol)],
            hatch=hatch, edgecolor='black', linewidth=0.4, capsize=2,
            error_kw={'elinewidth': 0.7, 'ecolor': 'black'},
        )
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=8)


def save(fig, output, name, dpi):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main(data_root, output, dpi=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = load_table(data_root)
    table.to_csv(output / 'three_protocol_metrics.csv', index=False)

    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    _grouped(axes[0], table, 'ndcg_mean', 'ndcg_std', 'NDCG@50')
    _grouped(axes[1], table, 'recall_mean', 'recall_std', 'Recall@50')
    axes[1].legend(frameon=False, loc='upper left')
    fig.suptitle('Original truth, 854 fixed users, mean ± sample std over 3 seeds', fontsize=10)
    save(fig, output, 'three_protocol_metrics', dpi)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    positions = np.arange(len(PROTOCOLS))
    for ax, mean_col, std_col, ylabel in zip(
        axes, ['ndcg_mean', 'recall_mean'], ['ndcg_std', 'recall_std'], ['NDCG@50', 'Recall@50']
    ):
        for model, color in zip(MODEL_ORDER, COLORS):
            part = table.loc[table.model == model].sort_values('protocol')
            ax.errorbar(
                positions, part[mean_col], yerr=part[std_col], marker='o',
                color=color, label=model, capsize=2, linewidth=1.2, markersize=4,
            )
        ax.set_xticks(positions, PROTOCOLS, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Same evaluation protocol; only the training stack changes', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'three_protocol_trajectory.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    manifest = {
        'figures': [
            'three_protocol_metrics.png', 'three_protocol_metrics.pdf',
            'three_protocol_trajectory.png', 'three_protocol_trajectory.pdf',
        ],
        'dpi': dpi,
        'scope': 'original',
        'k': 50,
        'users': 854,
        'seeds': [42, 43, 44],
        'v3_source': 'sealed means in revision_v3_results_report.md; seed-level CSV is not in data/research_revision',
        'table': 'three_protocol_metrics.csv',
    }
    (output / 'three_protocol_figure_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote comparison figures to {output.resolve()}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    main(root / 'data', Path(__file__).resolve().parent / 'figures' / 'revision_v3')
