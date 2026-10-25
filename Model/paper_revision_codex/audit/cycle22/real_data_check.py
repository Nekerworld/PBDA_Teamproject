"""Cycle 22 guards on the real v7 root (read only: manifests and recorded GNN refit histories, no training)."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp

fold = HERE / 'audit_runs' / 'v7' / 'F4_unseen'
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}',
         f'rolling_protocol.py sha256 (working copy bytes) {hashlib.sha256((HERE / "rolling_protocol.py").read_bytes()).hexdigest()}']
for source in (fold / 'prepared' / 'validation', fold / 'val_mf', fold / 'prepared'):
    try:
        rp.require_inner_validation(fold, source, 'MF selection')
        lines.append(f'accepted {source.relative_to(HERE)}')
    except RuntimeError as error:
        lines.append(f'refused  {source.relative_to(HERE)}: {error}')
for job in sorted((fold / 'runs').glob('*gnn_seed_*')):
    if not job.is_dir():
        continue
    spec = json.loads((job / 'job_result.json').read_text(encoding='utf-8'))['spec']
    trained = json.loads((job / 'gnn_training_history.json').read_text(encoding='utf-8'))
    ok = trained.get('mode') == 'fixed_epochs' and trained.get('chosen_epochs') == spec['epochs']
    lines.append(f'refit {job.name}: mode={trained.get("mode")} chosen={trained.get("chosen_epochs")} '
                 f'spec_epochs={spec["epochs"]} accepted={ok}')
selection = json.loads((fold / 'selection.json').read_text(encoding='utf-8'))
lines.append(f'selection gnn={selection["gnn"]["name"]} hybrid_backbone={selection["hybrid_backbone"]["name"]}')
out = Path(sys.argv[1])
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.read_text(encoding='utf-8'))
