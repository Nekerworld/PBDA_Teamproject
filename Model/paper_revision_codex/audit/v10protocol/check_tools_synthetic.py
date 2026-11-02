"""Before the real v10 run: the v10 protocol on synthetic data over all four folds (tiny grids, as test_v10_protocol),
then the result tools on its output: package_results, verify_results, primary_family and recompute_test_metrics.

Usage: python audit/v10protocol/check_tools_synthetic.py <work dir>"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import hybrid_ltr as ltr  # noqa: E402
import hybrid_v9 as v9  # noqa: E402
import rolling_protocol as rp  # noqa: E402
from test_v6_smoke import make_raw  # noqa: E402
from test_v9_protocol import SMALL  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=False)
rp.use_protocol('v10')
for key, value in dict(SEEDS=[42, 43], ALS_GRID=[dict(factors=4, regularization=.02, alpha=2., iterations=3)],
                       BPR_GRID=[dict(factors=4, learning_rate=1e-2, regularization=.03)], BPR_EPOCH_CHECKPOINTS=(1, 2),
                       GNN_GRID=[dict(layers=1, embedding_dim=8, learning_rate=1e-2),
                                 dict(layers=2, embedding_dim=8, learning_rate=1e-2)],
                       GNN_FIXED={**rp.GNN_FIXED, 'batch_size': 64, 'n_negatives': 2, 'validation_interval': 2,
                                  'min_epochs': 1, 'early_stopping_patience': 4},
                       GNN_MAX_EPOCHS=4, GNN_CAP_EXTENSION_EPOCHS=6, MF_ALPHAS=[0., .5, 1.], HYBRID_BACKBONES=2,
                       HYBRID_GRID=[dict(alpha=.5, strategy=s, candidate_k=c, fusion='minmax', omega=w, gamma=1.)
                                    for c, s in ((3, 'blend'), (3, 'balanced_union'), ('all', 'blend')) for w in (.3, .7)],
                       RERANKER_CV_FOLDS=2, LADDER_NEGATIVES=5,
                       git_state=lambda: {'commit': 'synthetic', 'branch': 'synthetic', 'dirty_files': []},
                       gnn_device=lambda: 'cpu').items():
    setattr(rp, key, value)
ltr.PARAM_GRID = [dict(learning_rate=.1, max_iter=5, max_leaf_nodes=4), dict(learning_rate=.1, max_iter=10, max_leaf_nodes=4)]
ltr.POOL = dict(per_model=6, history=6, popular=4)
v9.DESIGNS['a02'] = dict(v9.DESIGNS['a02'], pool=SMALL)

raw, root = work / 'raw', work / 'run'
make_raw(raw, users=150, events=30000)
rp.run(root, raw)
print('run completed:', json.loads((root / 'progress.jsonl').read_text(encoding='utf-8').splitlines()[-1])['stage'])

tools = HERE / 'audit' / 'v10protocol'
py = sys.executable
for command in ([py, tools / 'package_results.py', root, work / 'results'],
                [py, tools / 'verify_results.py', work / 'results'],
                [py, tools / 'primary_family.py', work / 'results', work / 'primary']):
    out = subprocess.run([str(c) for c in command], capture_output=True, encoding='utf-8', errors='replace')
    print(f'$ {" ".join(Path(str(c)).name for c in command)} -> rc {out.returncode}')
    print(out.stdout[-4000:], out.stderr[-2000:])
for fold in [f['name'] for f in rp.FOLDS]:
    out = subprocess.run([py, str(HERE / 'report_tools' / 'recompute_test_metrics.py'), '--root', str(root), '--fold', fold,
                          '--raw', str(raw), '--output', str(work / f'recompute_{fold}.json')],
                         capture_output=True, encoding='utf-8', errors='replace')
    result = json.loads((work / f'recompute_{fold}.json').read_text(encoding='utf-8')) if out.returncode == 0 else out.stderr[-2000:]
    print(f'recompute {fold}: rc {out.returncode}', result)
