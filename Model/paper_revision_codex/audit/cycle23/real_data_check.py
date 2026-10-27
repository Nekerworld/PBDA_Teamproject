"""Cycle 23 guards on the real v7 root: the recorded GNN refit histories against the strengthened refit check, and the
cache audit with the MF fit data check (read only: manifests, job results and histories, no training, no labels)."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'report_tools'))
import check_cached_runs

root = HERE / 'audit_runs' / 'v7'
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}'] + [
    f'{name} sha256 (working copy bytes) {hashlib.sha256((HERE / name).read_bytes()).hexdigest()}'
    for name in ('rolling_protocol.py', 'report_tools/check_cached_runs.py')]
for job in sorted(root.glob('*/runs/*gnn_seed_*')):
    if not job.is_dir():
        continue
    spec = json.loads((job / 'job_result.json').read_text(encoding='utf-8'))['spec']
    trained = json.loads((job / 'gnn_training_history.json').read_text(encoding='utf-8'))
    rows = trained.get('refit_history') or []
    ok = (trained.get('mode') == 'fixed_epochs' and trained.get('chosen_epochs') == spec['epochs']
          and [row.get('epoch') for row in rows] == list(range(1, spec['epochs'] + 1))
          and not any('validation_original_ndcg' in row for row in rows))
    lines.append(f'refit {job.relative_to(root)}: mode={trained.get("mode")} chosen={trained.get("chosen_epochs")} '
                 f'spec_epochs={spec["epochs"]} epoch_rows={len(rows)} accepted={ok}')
summary = check_cached_runs.main(root, Path(sys.argv[1]).with_name('cache_identity_v7.json'))
lines.append('cache audit ' + json.dumps({k: v for k, v in summary.items() if k not in ('gnn', 'mf')}))
lines += [f'mf {row["fit"]}: differs={row["differs"]}' for row in summary['mf']]
out = Path(sys.argv[1])
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.read_text(encoding='utf-8'))
