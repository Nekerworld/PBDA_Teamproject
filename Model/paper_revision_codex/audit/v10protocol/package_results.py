"""Copy the v10 run record (per fold: the results_v8 file set, the sealed re-rankers and the sealed ladder rankers) to
audit/results_v10 with SHA256SUMS.

Usage: python audit/v10protocol/package_results.py audit_runs/v10_rr audit/results_v10"""
import hashlib
import shutil
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
folds = ['F1_hist', 'F2_mid', 'F3_early', 'F4_unseen']
if out.exists():
    raise SystemExit(f'{out} exists; refusing to overwrite a committed record')
files = ['plan.json', 'progress.jsonl', 'test_access_log.jsonl', 'report_all_folds_summary.csv',
         'report_paired_tests_all_folds.csv', 'report_temporal_variation.csv']
for fold in folds:
    f = root / fold
    files += [f'{fold}/{p.relative_to(f).as_posix()}' for p in sorted(f.glob('*'))
              if p.is_file() and p.suffix in ('.json', '.csv') and p.name != 'test_metrics.csv']
    files += [f'{fold}/fold_raw/fold_raw.json', f'{fold}/prepared/dataset_statistics.csv']
    files += [f'{fold}/report/{p.name}' for p in sorted((f / 'report').glob('*.csv'))]
    for folder in ('rerankers', 'ladder'):
        files += [f'{fold}/{folder}/{p.name}' for p in sorted((f / folder).glob('*')) if p.is_file()]
out.mkdir(parents=True)
(out / '.gitattributes').write_text('* -text\n', encoding='utf-8', newline='\n')
sums = []
for name in files:
    source = root / name
    target = out / (name.replace('/prepared/', '/') if '/prepared/' in name else name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    sums.append(f'{hashlib.sha256(target.read_bytes()).hexdigest()} *{target.relative_to(out).as_posix()}')
(out / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n', encoding='utf-8', newline='\n')
print(len(sums), 'files')
