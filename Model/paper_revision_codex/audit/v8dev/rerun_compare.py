"""Rerun of hybrid_ltr_dev.py (both designs) with the committed code against the saved development results.

Only the timing fields (started/finished time, Hybrid v1 grid seconds) may differ; every other value must be equal.
Usage: python rerun_compare.py <rerun root> <output>
"""
import json, sys
from pathlib import Path

saved_dir = Path(__file__).resolve().parent
rerun_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
TIMING = {'started_utc', 'finished_utc', 'hybrid_v1_grid_seconds'}


def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        found = [f'{path}: keys {sorted(set(a) ^ set(b))}'] if set(a) != set(b) else []
        for key in sorted(set(a) & set(b)):
            if key not in TIMING:
                found += diff(a[key], b[key], f'{path}/{key}')
        return found
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f'{path}[{i}]')]
    if isinstance(a, float) and isinstance(b, float) and abs(a - b) <= 1e-12:
        return [] if a == b else [('rounding', path, abs(a - b))]
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


lines = []
for name in ('dev_results.json', 'dev_results_weighted.json'):
    saved = json.loads((saved_dir / name).read_text(encoding='utf-8'))
    rerun = json.loads((rerun_dir / name).read_text(encoding='utf-8'))
    found = diff(saved, rerun)
    rounding = [d for d in found if isinstance(d, tuple)]
    other = [d for d in found if not isinstance(d, tuple)]
    lines.append(f'{name}: saved {saved["started_utc"]}, rerun {rerun["started_utc"]} - {rerun["finished_utc"]}')
    lines.append(f'  float differences <= 1e-12: {len(rounding)}'
                 + (f' (max {max(d[2] for d in rounding):.1e}; fields: {sorted({d[1].rsplit("/", 1)[-1] for d in rounding})};'
                    f' sections: {sorted({d[1].split("/")[3] for d in rounding})})' if rounding else ''))
    lines.append(f'  other differences outside the timing fields: {len(other)}')
    lines += [f'    {d}' for d in other[:50]]
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.read_text(encoding='utf-8'))
