"""Re-run one v6 fold's preprocessing, leakage check and MF validation grid on this machine
and compare with the committed v6 summaries (audit/results_v6). Never reads the outer test.

Compared: fold_raw hashes, every leakage check value, validation populations, and every
ALS/BPR validation-grid row (repeat and novel metrics), plus the sealed MF selection.
"""
import os

# ALS output depends on the BLAS thread count; match rolling_protocol (forced, not defaulted)
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp  # noqa: E402
from audit_tools.leakage_check import main as leakage_main  # noqa: E402
from data_protocol import dump_json  # noqa: E402
from policy_evaluation import POLICIES, EvaluationContext  # noqa: E402

GRID_KEYS = ['family', 'factors', 'regularization', 'alpha', 'iterations', 'learning_rate', 'epochs']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_checks(new, old):
    rows = []
    for a, b in zip(new['checks'], old['checks']):
        same_key = (a['partition'], a['check']) == (b['partition'], b['check'])
        values_equal = {k: v for k, v in a.items()} == {k: v for k, v in b.items()}
        rows.append({'partition': a['partition'], 'check': a['check'], 'pass_new': a['pass'],
                     'pass_reference': b['pass'], 'identical': bool(same_key and values_equal)})
    return rows, len(new['checks']) == len(old['checks'])


def compare_grid(new, old):
    key = [k for k in GRID_KEYS if k in new.columns and k in old.columns]
    new, old = new.copy(), old.copy()
    for frame in (new, old):
        for column in key[1:]:
            frame[column] = frame[column].astype(float)
    merged = new.merge(old, on=key, how='outer', suffixes=('_new', '_ref'), indicator=True)
    metrics = [c[:-4] for c in merged.columns if c.endswith('_new') and c[:-4] + '_ref' in merged.columns]
    diffs = {m: float(np.nanmax(np.abs(merged[m + '_new'] - merged[m + '_ref']))) for m in metrics}
    by_family = {}
    for family, part in merged.groupby('family'):
        by_family[family] = {
            'rows': int(len(part)), 'unmatched_rows': int((part._merge != 'both').sum()),
            'max_abs_diff_repeat_score': float(np.nanmax(np.abs(part.repeat_score_new - part.repeat_score_ref))),
            'rows_bit_identical_all_metrics': int(np.all(
                [(part[m + '_new'] == part[m + '_ref']).to_numpy() for m in metrics], axis=0).sum())}
    return {'max_abs_diff': diffs, 'by_family': by_family}


def main(reference, root, fold_name, raw, output):
    fold = next(f for f in rp.FOLDS if f['name'] == fold_name)
    fold_dir = Path(root) / fold_name
    prepared, fold_raw = rp.prepare_fold(fold, root, raw)
    ref = Path(reference) / fold_name
    result = {'fold': fold_name, 'host': platform.node(), 'environment': rp.environment(),
              'reference_environment': read_json(Path(reference) / 'plan.json')['environment'],
              'outer_test_read': False}
    ref_raw = read_json(ref / 'fold_raw' / 'fold_raw.json')
    result['fold_raw_identical'] = {k: fold_raw[k] == ref_raw[k] for k in ref_raw}
    leak_path = fold_dir / 'leakage_check.json'
    if not leak_path.exists():
        leakage_main(prepared, fold_dir / 'fold_raw', gnn=True, output=leak_path)
    rows, same_count = compare_checks(read_json(leak_path), read_json(ref / 'leakage_check.json'))
    result['leakage'] = {'checks': len(rows), 'same_check_list': same_count,
                         'all_pass': all(r['pass_new'] for r in rows),
                         'identical_values': sum(r['identical'] for r in rows),
                         'differing': [r for r in rows if not r['identical']]}
    contexts = {p: EvaluationContext(prepared / 'validation', p) for p in POLICIES}
    population = [c.population() for c in contexts.values()]
    result['validation_population_identical'] = population == read_json(ref / 'validation_population.json')
    selection = rp.mf_validation(fold_dir, prepared / 'validation', contexts)
    reference_selection = read_json(ref / 'selection_mf.json')
    result['mf_selection_identical'] = {k: selection[k] == reference_selection[k] for k in ('als', 'bpr', 'bpr_epochs')}
    result['mf_grid'] = compare_grid(pd.read_csv(fold_dir / 'validation_mf_grid.csv'),
                                     pd.read_csv(ref / 'validation_mf_grid.csv'))
    result['train_seconds'] = selection['train_seconds']
    Path(output).mkdir(parents=True, exist_ok=True)
    dump_json(Path(output) / f'cross_machine_{fold_name}.json', result)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=HERE / 'audit' / 'results_v6')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fold', default='F3_early')
    parser.add_argument('--raw', type=Path, default=HERE.parent.parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.root, args.fold, args.raw.resolve(), args.output)
