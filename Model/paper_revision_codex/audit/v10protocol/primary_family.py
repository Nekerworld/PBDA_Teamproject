"""Primary family of audit/v10_rr_plan.md section 5: one Holm correction over the four folds, NDCG@50 only.

Per fold: Hybrid_LTR, Hybrid_LTR_W and Hybrid_v9 each vs the 8 baselines (confirmatory rows) and the 5 ladder steps
(ladder rows), 29 comparisons; the fold p-values of report/paired_tests.csv are used as they are. Also writes the
verdicts the plan defines: the ladder per fold (ratios of report/ladder.csv and both steps significant here) and, for
the v8 re-rankers, the baselines each one is significantly above.

usage: python audit/v10protocol/primary_family.py <run root> <output dir> [--folds F1_hist ...]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import stats_tools  # noqa: E402

FOLDS = ['F1_hist', 'F2_mid', 'F3_early', 'F4_unseen']
BASELINES = ['ALS', 'BPR', 'ALS+BPR', 'GNN', 'Popularity', 'Recent', 'RecentPop', 'CoVisit']
PRIMARY = ['Hybrid_LTR', 'Hybrid_LTR_W', 'Hybrid_v9']
STEPS = ['Hybrid_v9 - Hybrid_v9_MF_Graph', 'Hybrid_v9_MF_Graph - ALS+BPR', 'ALS+BPR - ALS', 'ALS+BPR - BPR',
         'ALS+BPR - GNN']
L3_L2, L2_L1 = STEPS[0], STEPS[1]
ALPHA = .05


def wanted():
    return ([('confirmatory', f'{h} - {b}') for h in PRIMARY for b in BASELINES]
            + [('ladder', step) for step in STEPS])


def primary_family(root, folds=FOLDS):
    rows = []
    for fold in folds:
        tests = pd.read_csv(Path(root) / fold / 'report' / 'paired_tests.csv')
        tests = tests.loc[(tests.policy == 'repeat') & (tests.metric == 'ndcg')]
        for family, comparison in wanted():
            match = tests.loc[(tests.family == family) & (tests.comparison == comparison)]
            if len(match) != 1:
                raise ValueError(f'{fold}: {len(match)} rows for {family} {comparison}')
            rows.append({'fold': fold, **match.iloc[0].to_dict()})
    frame = pd.DataFrame(rows).rename(columns={'p_holm': 'p_holm_fold', 'significant_holm': 'significant_fold'})
    frame['p_holm_primary'] = stats_tools.holm(frame.p_value.to_numpy())
    frame['significant_primary'] = frame.p_holm_primary < ALPHA
    frame['higher_primary'] = frame.significant_primary & (frame.difference > 0)
    return frame


def verdicts(root, frame, folds=FOLDS):
    ladder, reranker = [], []
    for fold in folds:
        ratios = pd.read_csv(Path(root) / fold / 'report' / 'ladder.csv').set_index('policy').loc['repeat']
        part = frame.loc[frame.fold == fold].set_index('comparison')
        steps_significant = bool(part.loc[L3_L2, 'higher_primary'] and part.loc[L2_L1, 'higher_primary'])
        ladder.append({'fold': fold, **{k: ratios[k] for k in ('L0_max', 'L1', 'L2', 'L3', 'L1/L0', 'L2/L1', 'L3/L2',
                                                               'best_baseline')},
                       'ratios_pass': bool(ratios['pass']), 'L3-L2_p_holm_primary': part.loc[L3_L2, 'p_holm_primary'],
                       'L2-L1_p_holm_primary': part.loc[L2_L1, 'p_holm_primary'],
                       'significant_result': bool(ratios['pass']) and steps_significant})
        for model in PRIMARY:
            above = [b for b in BASELINES if part.loc[f'{model} - {b}', 'higher_primary']]
            reranker.append({'fold': fold, 'model': model, 'significantly_above': ' '.join(above),
                             'above_all_baselines': len(above) == len(BASELINES)})
    return pd.DataFrame(ladder), pd.DataFrame(reranker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--folds', nargs='+', default=FOLDS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = primary_family(args.root, args.folds)
    ladder, models = verdicts(args.root, frame, args.folds)
    frame.to_csv(args.output / 'primary_family.csv', index=False)
    ladder.to_csv(args.output / 'primary_ladder_verdicts.csv', index=False)
    models.to_csv(args.output / 'primary_model_verdicts.csv', index=False)
    summary = {'comparisons': len(frame), 'folds': args.folds, 'alpha': ALPHA,
               'ladder_significant_folds': ladder.loc[ladder.significant_result, 'fold'].tolist(),
               'ladder_ratio_pass_folds': ladder.loc[ladder.ratios_pass, 'fold'].tolist(),
               'above_all_baselines': models.loc[models.above_all_baselines, ['fold', 'model']].to_dict('records')}
    (args.output / 'primary_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))
