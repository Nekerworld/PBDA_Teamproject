"""Mean of the BPR pair-loss weights before the mean-1 normalisation in train_weighted_bpr, on the real v7 F4 training
data of each seed's refit (the data are the same split for every seed, so the value does not depend on the seed).
Reads only events_train_clean.csv; no labels, no training."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import rolling_protocol as rp

fold = HERE / 'audit_runs' / 'v7' / 'F4_unseen'
lines = [f'started_utc {datetime.now(timezone.utc).isoformat()}', f'MF_COMMON {rp.MF_COMMON}']
for directory in [fold / 'prepared'] + sorted((fold / 'runs').glob('seed_*')):
    matrix, users, items = rp.interaction_matrix(directory)
    raw = matrix.tocoo().data.astype(np.float32)
    lines.append(f'{directory.relative_to(fold)}: pairs={len(raw)} users={len(users)} items={len(items)} '
                 f'raw_weight_mean={raw.mean():.6f} min={raw.min():.6f} max={raw.max():.6f}')
out = Path(sys.argv[1])
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.read_text(encoding='utf-8'))
