"""Compare re-generated outputs (written under the git-ignored audit_runs/) with the committed evidence they reproduce.

Each pair is NEW=COMMITTED; a directory pair compares every file of COMMITTED (except .gitattributes, which is not
generated) with the same relative path in NEW. NEW=COMMITTED#k1,k2 compares only those top-level JSON keys, for
outputs that are meant to change over time (verify_plan_code lists the files added since the plan). For every file the output records both sha256 digests, whether the bytes are equal apart from line endings, and for
JSON files whether the parsed values are equal and which top-level keys differ. Nothing is re-run here.

Usage: python report_tools/compare_outputs.py --output audit_runs/repro/compare.json \
           audit_runs/repro/stamp_code_v7.json=audit/cycle9/stamp_code_v7.json audit_runs/repro/figures_v7=audit/figures_v7
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path



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
            row['compared_keys'] = keys
            x, y = {k: x.get(k) for k in keys}, {k: y.get(k) for k in keys}
        row['json_equal'] = x == y
        if isinstance(x, dict) and isinstance(y, dict):
            row['differing_top_level_keys'] = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
        row['equal'] = row['json_equal']
    return row


def main(pairs, output):
    rows = []
    for pair in pairs:
        new, target = pair.split('=', 1)
        committed, _, keys = target.partition('#')
        new, committed, keys = Path(new), Path(committed), [k for k in keys.split(',') if k]
        if committed.is_dir():
            rows += [compare_file(new / f.relative_to(committed), f) for f in sorted(committed.rglob('*'))
                     if f.is_file() and f.name != '.gitattributes']
        else:
            rows.append(compare_file(new, committed, keys))
    result = {'compared_utc': datetime.now(timezone.utc).isoformat(), 'all_equal': all(r['equal'] for r in rows),
              'files': len(rows), 'rows': rows}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    for r in rows:
        print('equal' if r['equal'] else 'DIFFERENT', r['committed'], r.get('differing_top_level_keys', ''))
    print('all_equal', result['all_equal'], 'files', len(rows))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pairs', nargs='+', help='NEW=COMMITTED (files or directories)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.pairs, args.output)
