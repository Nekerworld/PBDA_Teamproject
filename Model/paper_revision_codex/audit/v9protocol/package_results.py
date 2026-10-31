"""Copy the v9 sealed-run record (the results_v8 file set with the sealed ladder rankers) to audit/results_v9 with SHA256SUMS.

Usage: python audit/v9protocol/package_results.py audit_runs/v9_taobao audit/results_v9"""
import hashlib
import shutil
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
fold = 'TB2_full'
if out.exists():
    raise SystemExit(f'{out} exists; refusing to overwrite a committed record')
files = ['plan.json', 'progress.jsonl', 'test_access_log.jsonl', 'report_all_folds_summary.csv',
         'report_paired_tests_all_folds.csv', 'report_temporal_variation.csv']
f = root / fold
files += [f'{fold}/{p.relative_to(f).as_posix()}' for p in sorted(f.glob('*'))
          if p.is_file() and p.suffix in ('.json', '.csv') and p.name != 'test_metrics.csv']
files += [f'{fold}/fold_raw/fold_raw.json', f'{fold}/prepared/dataset_statistics.csv']
files += [f'{fold}/report/{p.name}' for p in sorted((f / 'report').glob('*.csv'))]
files += [f'{fold}/ladder/{p.name}' for p in sorted((f / 'ladder').glob('*')) if p.is_file()]
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
