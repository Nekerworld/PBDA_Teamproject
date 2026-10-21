"""Run the cross-root test-access guard against the real raw data and the recorded result roots.

For the v6 and v7 fold sets, asks which windows would be refused for the v6 plan, the v7 plan and a new
plan, with the current code. Also reports whether the raw files equal what each plan recorded. Nothing
is trained, no test file is read, and no access log is written.

The test-opening check compares actual windows (event timestamps). ``windows`` applies the split rule to the raw
events and compares it with each recorded window; ``boundary`` takes new folds whose nominal window only touches
F4's and reports what the start check (nominal shares) and the opening check (timestamps) say. ``decisions`` gives,
for each fold a record opened, whether its logged selection hash is the selection.json kept there and the decision
an exact reproduction would have to match (``rp.opened_decision``).

Usage: python report_tools/guard_real_data_check.py --raw ../../data --output audit/cycle7/guard_real_data.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402


def split_window(ts, fold):
    """[start, end) timestamps the pipeline's split rule gives on these raw events (cleaning removing none)."""
    n = len(ts)
    end = None if fold['end_fraction'] >= 1 else int(ts[int(n * fold['end_fraction'])])
    m = n if end is None else int(np.searchsorted(ts, end, 'left'))
    start = int(ts[min(max(int(m * (1 - fold['test_size'])), 1), m - 1)])
    return start, (end if end is not None else float('inf'))


def main(raw, output):
    hashes, forms = rp.raw_hashes(raw), rp.event_digests(raw)
    plans = {v: json.loads((HERE / 'audit' / f'results_{v}' / 'plan.json').read_text(encoding='utf-8'))
             for v in ('v6', 'v7')}
    result = {'raw_equals_plan_record': {v: p['raw_input_sha256'] == hashes for v, p in plans.items()},
              'plan_events_hash_in_event_forms': {v: p['raw_input_sha256']['events.csv'] in forms
                                                  for v, p in plans.items()},
              'records': sorted(str(p.parent.relative_to(HERE).as_posix()) for base in rp.RESULT_RECORDS
                                for p in Path(base).glob('*/plan.json')),
              'conflicts': {}}
    ts = np.sort(pd.read_csv(Path(raw) / 'events.csv', usecols=['timestamp']).timestamp.to_numpy())
    result['windows'] = {}
    for v, plan in plans.items():
        for fold in plan['plan']['folds']:
            recorded = rp.recorded_test_window(HERE / 'audit' / f'results_{v}' / fold['name'])
            result['windows'][f'{v}/{fold["name"]}'] = {'recorded': [str(x) for x in recorded],
                                                       'split_rule_on_raw': [str(x) for x in split_window(ts, fold)]}
    result['boundary'] = {}
    for end in (.39, .40, .43, .44, .45):
        fold = {'name': f'N{end}', 'end_fraction': end, 'test_size': 1 - .36 / end}
        own = split_window(ts, fold)
        start_check = rp.committed_test_conflicts(hashes, 'new', [fold], event_forms=forms)
        opening_check = rp.committed_test_conflicts(hashes, 'new', [fold], event_forms=forms,
                                                    selections={fold['name']: 'new'}, test_windows={fold['name']: own})
        result['boundary'][fold['name']] = {
            'window': [str(x) for x in own], 'events_shared_with_F4': int(max(0, np.searchsorted(ts, 1434817031045, 'left')
                                                                            - np.searchsorted(ts, own[0], 'left'))),
            'start_check_conflicts': len(start_check), 'opening_check_conflicts': len(opening_check)}
    result['decisions'] = {}
    for record in result['records']:
        log_path = HERE / record / 'test_access_log.jsonl'
        opened = {}
        for line in (log_path.read_text(encoding='utf-8').splitlines() if log_path.exists() else []):
            entry = json.loads(line)
            opened.setdefault(entry['fold'], set()).add(entry.get('selection_sha256'))
        for fold, hashes_opened in sorted(opened.items()):
            path = HERE / record / fold / 'selection.json'
            result['decisions'][f'{record}/{fold}'] = {
                'distinct_selection_hashes': len(hashes_opened), 'log_matches_selection_file': hashes_opened == {rp.sha256(path)},
                'opened_decision': rp.opened_decision(path.parent, hashes_opened)}
    for protocol in ('v6', 'v7'):
        rp.use_protocol(protocol)
        for name, plan_hash in (('v6_plan', plans['v6']['plan_sha256']), ('v7_plan', plans['v7']['plan_sha256']),
                                ('new_plan', 'new')):
            found = rp.committed_test_conflicts(hashes, plan_hash, rp.FOLDS, event_forms=forms)
            result['conflicts'][f'{protocol}_folds/{name}'] = [
                {**c, 'record': Path(c['record']).relative_to(HERE).as_posix()} for c in found]
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'conflicts'}, indent=2))
    for key, found in result['conflicts'].items():
        print(key, sorted({c['fold'] for c in found}), len(found))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.raw, args.output)
