"""How many evaluation users a Hybrid - baseline difference of the observed size needs to pass Holm at 80% power.

SE is read off the paired bootstrap 95% CI (width / 3.92). The Holm multiplier is the one each comparison actually got
(p_holm / p_value, at least 1), so the level is the one the reported claim was judged at. Users scale as SE^-2.
Inputs: the committed v7 F4 paired tests and the v8 development results (D2, development window, not a test).
"""
import json, sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

src = Path(__file__).resolve().parents[2]
rows = []


def add(source, policy, frame, users):
    for _, r in frame.iterrows():
        se = (r.ci_high - r.ci_low) / 3.92
        multiplier = max(r.p_holm / r.p_value, 1.) if r.p_value > 0 else 1.
        z = norm.ppf(1 - .05 / multiplier / 2) + norm.ppf(.8)
        needed = users * (z * se / abs(r.difference)) ** 2 if r.difference else float('inf')
        rows.append({'source': source, 'policy': policy, 'comparison': r.comparison, 'metric': r.metric,
                     'difference': r.difference, 'se': se, 'p_holm': r.p_holm, 'users': users,
                     'holm_multiplier': multiplier, 'mde_at_users': z * se, 'users_needed_for_observed': needed})


v7 = pd.read_csv(src / 'audit' / 'results_v7' / 'F4_unseen' / 'report' / 'paired_tests.csv')
for policy, frame in v7.groupby('policy'):
    add('v7 F4 test', policy, frame[frame.comparison.isin(['Hybrid - ALS+BPR', 'Hybrid - GNN', 'Hybrid - BPR', 'Hybrid - ALS'])],
        int(frame.users.iloc[0]))
for name in ('dev_results.json', 'dev_results_weighted.json'):
    result = json.loads((src / 'audit' / 'v8dev' / name).read_text(encoding='utf-8'))
    for policy, record in result['policies'].items():
        frame = pd.DataFrame(record['paired_d2']['Hybrid_v2 vs baselines and Hybrid_v1 (one Holm family)'])
        add(f'v8 dev D2 {result.get("design", "policy_truth")}', policy,
            frame[frame.comparison.isin(['Hybrid_v2 - ALS+BPR', 'Hybrid_v2 - GNN', 'Hybrid_v2 - Hybrid_v1'])],
            int(record['metrics_d2']['Hybrid_v2']['users']))
table = pd.DataFrame(rows)
out = Path(sys.argv[1])
table.to_csv(out, index=False)
print(table.round(4).to_string())
