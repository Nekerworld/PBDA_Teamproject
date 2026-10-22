"""Compare re-generated outputs (written under the git-ignored audit_runs/) with the committed evidence they reproduce.

Each pair is NEW=COMMITTED; a directory pair compares every file of COMMITTED (except .gitattributes, which is not
generated) with the same relative path in NEW. NEW=COMMITTED#k1,k2 compares only those top-level JSON keys, for
outputs that are meant to change over time (verify_plan_code lists the files added since the plan). For every file the output records both sha256 digests, whether the bytes are equal apart from line endings, and for
JSON files whether the parsed values are equal (same types too: 1, 1.0 and true differ; NaN equals NaN) and which
top-level keys differ. A file that only NEW has, a comparison of zero files (in any pair), or #keys on a non-JSON
file, a non-object JSON or a key one of the files lacks makes the result unequal or an error, so a wrong path or key
cannot pass. Nothing is re-run here.

Usage: python report_tools/compare_outputs.py --output audit_runs/repro/compare.json \
           audit_runs/repro/stamp_code_v7.json=audit/cycle9/stamp_code_v7.json audit_runs/repro/figures_v7=audit/figures_v7
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def same(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    if isinstance(x, float) and math.isnan(x):
        return math.isnan(y)
    return x == y


def compare_file(new, committed, keys=None):
    row = {'new': new.as_posix(), 'committed': committed.as_posix(), 'new_exists': new.exists()}
    if not new.exists():
        return {**row, 'equal': False}
    a, b = new.read_bytes(), committed.read_bytes()
    row.update(new_sha256=hashlib.sha256(a).hexdigest(), committed_sha256=hashlib.sha256(b).hexdigest(),
               bytes_equal_apart_from_line_endings=a.replace(b'\r\n', b'\n') == b.replace(b'\r\n', b'\n'))
    row['equal'] = row['bytes_equal_apart_from_line_endings']
    if committed.suffix == '.json':
        x, y = json.loads(a), json.loads(b)
        if keys:
            if not (isinstance(x, dict) and isinstance(y, dict)):
                raise ValueError(f'#keys needs a JSON object: {committed}')
            missing = [k for k in keys if k not in x or k not in y]
            if missing:
                raise ValueError(f'#keys not in both files ({committed}): {missing}')
            row['compared_keys'] = keys
            x, y = {k: x[k] for k in keys}, {k: y[k] for k in keys}
        row['json_equal'] = same(x, y)
        if isinstance(x, dict) and isinstance(y, dict):
            row['differing_top_level_keys'] = sorted(k for k in set(x) | set(y)
                                                     if k not in x or k not in y or not same(x[k], y[k]))
        row['equal'] = row['json_equal']
    elif keys:
        raise ValueError(f'#keys needs a JSON file: {committed}')
    return row


def listed(folder):
    return {f.relative_to(folder) for f in folder.rglob('*') if f.is_file() and f.name != '.gitattributes'}


def main(pairs, output):
    rows, empty_pairs = [], []
    for pair in pairs:
        new, target = pair.split('=', 1)
        committed, _, keys = target.partition('#')
        new, committed, keys = Path(new), Path(committed), [k for k in keys.split(',') if k]
        if not committed.exists():
            raise FileNotFoundError(f'Committed evidence not found: {committed}')
        if committed.is_dir():
            if keys:
                raise ValueError(f'#keys needs a JSON file, not a folder: {committed}')
            if not listed(committed):
                empty_pairs.append(pair)
            rows += [compare_file(new / name, committed / name) for name in sorted(listed(committed))]
            rows += [{'new': (new / name).as_posix(), 'committed': (committed / name).as_posix(), 'new_exists': True,
                      'committed_exists': False, 'equal': False}
                     for name in sorted(listed(new) - listed(committed))] if new.is_dir() else []
        else:
            rows.append(compare_file(new, committed, keys))
    result = {'compared_utc': datetime.now(timezone.utc).isoformat(),
              'all_equal': bool(rows) and not empty_pairs and all(r['equal'] for r in rows), 'files': len(rows),
              'pairs_without_committed_files': empty_pairs, 'rows': rows}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    for r in rows:
        print('equal' if r['equal'] else 'DIFFERENT', r['committed'], r.get('differing_top_level_keys', ''))
    for pair in empty_pairs:
        print('NO COMMITTED FILES', pair)
    print('all_equal', result['all_equal'], 'files', len(rows))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pairs', nargs='+', help='NEW=COMMITTED (files or directories)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.pairs, args.output)
