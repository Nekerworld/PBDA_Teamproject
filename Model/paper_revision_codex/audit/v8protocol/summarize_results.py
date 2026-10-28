"""Tables of the v8 sealed test, written before the test was opened (reads only the run's report files).

Usage: python audit/v8protocol/summarize_results.py audit/results_v8 audit/v8protocol/results_summary.md"""
import json
import sys
from pathlib import Path

import pandas as pd

LADDER = ['Popularity', 'Recent', 'RecentPop', 'ALS', 'BPR', 'ALS+BPR', 'GNN', 'Hybrid_ALS_only', 'Hybrid',
          'Hybrid_LTR_mf', 'Hybrid_LTR_mf_gnn', 'Hybrid_LTR_no_history', 'Hybrid_LTR',
          'Hybrid_LTR_W_mf', 'Hybrid_LTR_W_mf_gnn', 'Hybrid_LTR_W_no_history', 'Hybrid_LTR_W']


def markdown(frame):
    cell = lambda v: f'{v:.5f}' if isinstance(v, float) else str(v)
    rows = [list(frame.columns)] + [[cell(v) for v in row] for row in frame.itertuples(index=False)]
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '|' + '---|' * len(rows[0])]
                     + ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])


root, out = Path(sys.argv[1]), Path(sys.argv[2])
fold = root / 'TB1_full'
summary = pd.read_csv(fold / 'report' / 'test_summary.csv')
tests = pd.read_csv(fold / 'report' / 'paired_tests.csv')
populations = pd.read_csv(fold / 'report' / 'populations.csv')
access = [json.loads(line) for line in (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()]
lines = ['# v8 sealed test (Taobao 1%): tables', '',
         f'Test openings: {len(access)} ({", ".join(a["purpose"] for a in access)}), git {access[0]["git"][:7]}', '',
         '## Populations', '', markdown(populations), '']
for policy in ('repeat', 'novel'):
    part = summary.loc[(summary.policy == policy) & (summary.k == 50)].set_index('model')
    order = [m for m in LADDER if m in part.index] + sorted(set(part.index) - set(LADDER))
    table = part.loc[order, ['ndcg_mean', 'ndcg_std', 'recall_mean', 'recall_std', 'hit_users_mean']].reset_index()
    lines += [f'## {policy}: K=50, mean and SD over seeds 42/43/44', '', markdown(table), '']
    for family in ('confirmatory', 'secondary'):
        rows = tests.loc[(tests.policy == policy) & (tests.family == family),
                         ['comparison', 'metric', 'difference', 'ci_low', 'ci_high', 'p_value', 'p_holm', 'claim',
                          'users_better', 'users_worse', 'users_tied']]
        lines += [f'### {policy}: {family} family (Holm within the family)', '', markdown(rows), '']
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.read_text(encoding='utf-8'))
