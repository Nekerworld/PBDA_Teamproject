"""Run the cross-root test-access guard against the real raw data and the recorded result roots.

For the v6 and v7 fold sets, asks which windows would be refused for the v6 plan, the v7 plan and a new
plan, with the current code. Also reports whether the raw files equal what each plan recorded. Nothing
is trained, no test file is read, and no access log is written.

Usage: python report_tools/guard_real_data_check.py --raw ../../data --output audit/cycle7/guard_real_data.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402


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
