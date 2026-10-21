"""How much each policy would gain if trained configurations were also selected per policy (validation only).

The preregistered rule selects trained configurations (ALS, BPR, GNN, Hybrid backbone) under the repeat
policy and re-selects only inference-time parameters (ALS+BPR alpha, Hybrid fusion) per policy. For each
fold and policy this reports, on inner validation, the score of the selected configuration, the best score
any configuration of the same grid reaches for that policy, and the selected Hybrid score. It reads the
committed selection files only; no model is trained and no outer-test file is read.

Usage: python report_tools/policy_tuning_gap.py --root audit/results_v7 --output <json>
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def score(metrics):
    return (metrics['ndcg'] + metrics['recall']) / 2


def ranked(items, policy_score, chosen):
    order = sorted(items, key=lambda item: policy_score(item[1]), reverse=True)
    names = [name for name, _ in order]
    return {'selected': policy_score(dict(items)[chosen]), 'best': policy_score(order[0][1]), 'best_config': names[0],
            'selected_rank': names.index(chosen) + 1, 'configs': len(names)}


def fold_gap(fold_dir):
    fold_dir = Path(fold_dir)
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    gnn = [(row['name'], row) for row in json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8'))]
    grid = pd.read_csv(fold_dir / 'validation_mf_grid.csv')
    out = {}
    for policy in ('repeat', 'novel'):
        families = {'GNN': ranked(gnn, lambda row: score(row[policy]), selection['gnn']['name'])}
        for family in ('ALS', 'BPR'):
            part = grid.loc[grid.family == family].sort_values('repeat_score', ascending=False, kind='stable')
            items = [(str(i), {'ndcg': r[f'{policy}_ndcg'], 'recall': r[f'{policy}_recall']})
                     for i, r in zip(part.index, part.to_dict('records'))]
            families[family] = ranked(items, score, items[0][0])
        alpha = pd.read_csv(fold_dir / f'validation_mf_alpha_{policy}.csv')
        chosen_alpha = selection['policies'][policy]['mf_alpha']
        families['ALS+BPR'] = {'selected': float(alpha.loc[alpha.alpha == chosen_alpha, 'score'].iloc[0]),
                               'best': float(alpha.score.max())}
        hybrid = float(selection['policies'][policy]['hybrid']['score'])
        out[policy] = {'hybrid_selected': hybrid, 'families': families,
                       'baselines_whose_best_config_beats_hybrid': [f for f, v in families.items() if v['best'] > hybrid]}
    return out


def main(root, output):
    root = Path(root)
    result = {'root': root.as_posix(), 'score': 'mean(NDCG@50, Recall@50) on inner validation',
              'folds': {d.name: fold_gap(d) for d in sorted(root.iterdir()) if (d / 'selection.json').exists()}}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    for fold, policies in result['folds'].items():
        for policy, row in policies.items():
            parts = ', '.join(f"{f} {v['selected']:.4f}/{v['best']:.4f}" for f, v in row['families'].items())
            print(f"{fold} {policy}: Hybrid {row['hybrid_selected']:.4f}; selected/best {parts}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
