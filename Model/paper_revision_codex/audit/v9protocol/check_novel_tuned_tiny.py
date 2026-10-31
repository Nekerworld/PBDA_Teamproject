"""Checks novel_tuned_baselines.py on the tiny synthetic v9 run of test_v9_protocol (no real data):
picking by repeat scores must reproduce the protocol's own novel baselines exactly; the analysis refuses a root whose
sealed test is not complete; the opening is logged once with the root's plan and selection.

Usage: python audit/v9protocol/check_novel_tuned_tiny.py <empty work folder>"""
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import rolling_protocol as rp  # noqa: E402
import novel_tuned_baselines as ntb  # noqa: E402
from test_v6_smoke import make_raw  # noqa: E402
from test_v9_protocol import tiny_v9  # noqa: E402

work = Path(sys.argv[1])
work.mkdir(parents=True)
with pytest.MonkeyPatch.context() as mp:
    tiny_v9(mp)
    raw, root = work / 'raw', work / 'run'
    make_raw(raw, users=60, events=9000)
    rp.run(root, raw)
    fold_dir = root / 'TB2_full'

    early = work / 'early'
    shutil.copytree(root, early, ignore=shutil.ignore_patterns('test_access_log.jsonl'))
    try:
        ntb.run(early)
        raise SystemExit('FAIL: ran without a sealed test opening')
    except RuntimeError as error:
        print('refuses before the sealed test:', error)

    same = ntb.run(root, pick_by='repeat', out_name='sens_repeat_check')
    print('repeat-picked choices', same['choices']['als'], same['choices']['bpr'], same['choices']['bpr_epochs'],
          same['choices']['gnn']['name'], 'alpha', same['mf_alpha'], 'gnn reused', same['gnn_reused_from_protocol'])
    selection = json.loads((fold_dir / 'selection.json').read_text(encoding='utf-8'))
    assert {k: selection['mf'][k] for k in ('als', 'bpr', 'bpr_epochs')} == {k: same['choices'][k] for k in ('als', 'bpr', 'bpr_epochs')}
    assert same['mf_alpha'] == selection['policies']['novel']['mf_alpha'] and same['gnn_reused_from_protocol']
    mine = pd.read_csv(fold_dir / 'sens_repeat_check' / 'test_metrics.csv')
    theirs = pd.read_csv(fold_dir / 'test_metrics.csv')
    theirs = theirs.loc[theirs.policy == 'novel']
    worst = 0.
    for base, new in ntb.TUNED.items():
        a = mine.loc[mine.model == new].set_index(['seed', 'k'])[['ndcg', 'recall', 'precision', 'hit_rate']].sort_index()
        b = theirs.loc[theirs.model == base].set_index(['seed', 'k'])[['ndcg', 'recall', 'precision', 'hit_rate']].sort_index()
        worst = max(worst, float((a - b).abs().to_numpy().max()))
    print('max |rebuilt - protocol| over ALS, BPR, ALS+BPR, GNN (novel, 3 seeds, 5 K):', worst)
    assert worst == 0.

    result = ntb.run(root)
    print('novel-picked choices', result['choices']['als'], result['choices']['bpr'], result['choices']['bpr_epochs'],
          result['choices']['gnn']['name'], 'gnn reused', result['gnn_reused_from_protocol'])
    print('ladder with tuned baselines', result['ladder_with_tuned_baselines'])
    access = (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines()
    print('access log purposes', [json.loads(line)['purpose'] for line in access])
    assert [json.loads(line)['purpose'] for line in access] == [
        'sealed_test', 'sensitivity_repeat_tuned_baselines', 'sensitivity_novel_tuned_baselines']
    ntb.run(root)
    assert (root / 'test_access_log.jsonl').read_text(encoding='utf-8').splitlines() == access
    print('rerun reads the stored result and logs nothing: ok')

    # the GNN refit path: force the other validation GNN
    original = ntb.choices
    def other_gnn(fold_dir, pick_by):
        picked = original(fold_dir, pick_by)
        rows = json.loads((fold_dir / 'validation_gnn.json').read_text(encoding='utf-8'))
        row = next(r for r in rows if r['name'] != picked['gnn']['name'])
        picked['gnn'] = {k: row[k] for k in ('name', 'config', 'epochs', 'at_cap', 'job_dir')} | {'extended': bool(row.get('extended'))}
        return picked
    mp.setattr(ntb, 'choices', other_gnn)
    forced = ntb.run(root, out_name='sens_other_gnn_check')
    assert not forced['gnn_reused_from_protocol']
    for seed in rp.SEEDS:
        job = json.loads((fold_dir / 'sens_other_gnn_check' / f'gnn_seed_{seed}' / 'job_result.json').read_text(encoding='utf-8'))
        assert job['spec']['mode'] == 'refit' and job['spec']['config'] == forced['choices']['gnn']['config']
        assert job['chosen_epochs'] == forced['choices']['gnn']['epochs'] and job['spec']['seed'] == seed
    print('GNN refit path:', forced['choices']['gnn']['name'], 'epochs', forced['choices']['gnn']['epochs'],
          'GNN_n ndcg', forced['seed_mean_ndcg50']['GNN_n'], 'protocol GNN', forced['seed_mean_ndcg50']['GNN'])
print('ALL CHECKS PASSED')
