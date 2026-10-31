"""The 'a02' design of the current code rebuilds the a02 D2 feature table (cached by the code of commit 7fb813a)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import hybrid_v9 as v9  # noqa: E402
import hybrid_v9_dev as dev  # noqa: E402


def main(policy='repeat'):
    old = pd.read_pickle(dev.ROOT / 'attempts' / 'a02' / f'd2_{policy}.pkl')
    context, scores, tables = dev.load(policy, *dev.bases()['d2'])
    signals = v9.Signals(context)
    new = v9.features(context, scores, signals, v9.candidates(context, scores, signals, v9.DESIGNS['a02']['pool']),
                      *tables)
    columns = list(old.columns)
    same = {'rows': len(old) == len(new), 'missing_columns': sorted(set(columns) - set(new.columns))}
    if same['rows'] and not same['missing_columns']:
        diff = (new[columns].to_numpy(dtype=np.float64) - old[columns].to_numpy(dtype=np.float64))
        same['max_abs_diff'] = float(np.abs(diff).max())
    same['equal'] = bool(same['rows'] and not same['missing_columns'] and same['max_abs_diff'] == 0)
    print(json.dumps(same))
    (Path(__file__).parent / f'check_variant_a02_{policy}.json').write_text(json.dumps(same, indent=2), encoding='utf-8')
    if not same['equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main(*sys.argv[1:])
