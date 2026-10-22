"""Read the recorded Hybrid validation grids back as a resumed run would and compare with the sealed selection.

For every fold of a result root: how many rows the default CSV float parser changes, whether round-trip parsing
re-serialises the file exactly apart from line endings (CRLF in the run folder), whether the score order differs, and how many float values of
selection.json are found exactly among the parsed grid values. Reads validation files only (no test data).

Usage: python report_tools/grid_cache_precision_check.py --root audit_runs/v7 --output audit_runs/repro/grid_cache_precision_v7.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def floats(value, out):
    if isinstance(value, dict):
        for item in value.values():
            floats(item, out)
    elif isinstance(value, list):
        for item in value:
            floats(item, out)
    elif isinstance(value, float):
        out.add(value)
    return out


def main(root, output):
    result = {}
    for fold_dir in sorted(p for p in Path(root).iterdir() if (p / 'selection.json').exists()):
        sealed = floats(json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8')), set())
        exact, default, files = set(), set(), []
        for path in sorted(fold_dir.glob('validation_hybrid_*.csv')):
            a, b = pd.read_csv(path), pd.read_csv(path, float_precision='round_trip')
            numeric = b.select_dtypes('number').columns
            exact |= set(b[numeric].to_numpy().ravel().tolist())
            default |= set(a[numeric].to_numpy().ravel().tolist())
            order = lambda f: f.sort_values('score', ascending=False, kind='stable').index.tolist()
            files.append({'file': path.name, 'rows': len(b),
                          'rows_changed_by_default_parser': int((a[numeric].to_numpy() != b[numeric].to_numpy()).any(axis=1).sum()),
                          'round_trip_reserialises_exactly': b.to_csv(index=False, lineterminator='\n')
                          == path.read_text(encoding='utf-8').replace('\r\n', '\n'),
                          'score_order_differs': order(a) != order(b)})
        result[fold_dir.name] = {'grids': files, 'selection_floats': len(sealed),
                                 'found_with_round_trip': len(sealed & exact), 'found_with_default': len(sealed & default)}
    text = json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
