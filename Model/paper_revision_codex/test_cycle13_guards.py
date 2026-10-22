"""compare_outputs cannot pass on a wrong key or an empty committed folder, the score-precision check refuses an
outer-test folder, use_protocol('v6') undoes v7, and the Hybrid grid-end flags cover alpha and the
per-strategy candidate ends."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'report_tools'))
import rolling_protocol as rp
import check_score_precision
import compare_outputs as co


def test_compare_outputs_keys_must_exist_and_pairs_must_compare_files(tmp_path):
    new, old, empty = tmp_path / 'new', tmp_path / 'old', tmp_path / 'empty'
    for d in (new, old, empty):
        d.mkdir()
    (new / 'x.json').write_text(json.dumps({'a': 999, 'b': 2}), encoding='utf-8')
    (old / 'x.json').write_text(json.dumps({'a': 1, 'b': 2}), encoding='utf-8')
    (new / 'l.json').write_text(json.dumps([1]), encoding='utf-8')
    (old / 'l.json').write_text(json.dumps([1]), encoding='utf-8')
    (empty / '.gitattributes').write_text('* -text\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not in both files'):
        co.main([f'{new / "x.json"}={old / "x.json"}#typo'], None)
    with pytest.raises(ValueError, match='JSON object'):
        co.main([f'{new / "l.json"}={old / "l.json"}#a'], None)
    assert co.main([f'{new / "x.json"}={old / "x.json"}#b'], None)['all_equal']
    result = co.main([f'{tmp_path / "nothing"}={empty}', f'{new / "x.json"}={old / "x.json"}#b'], None)
    assert result['files'] == 1 and not result['all_equal']
    assert result['pairs_without_committed_files'] == [f'{tmp_path / "nothing"}={empty}']


def test_score_precision_check_refuses_an_outer_test_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(check_score_precision, 'require_protocol', lambda directory: {'role': 'test'})

    def must_not_load(*args, **kwargs):
        raise AssertionError('the folder was read before its role was checked')

    monkeypatch.setattr(check_score_precision, 'EvaluationContext', must_not_load)
    with pytest.raises(ValueError, match='inner validation only'):
        check_score_precision.main(tmp_path, None)


def test_use_protocol_v6_undoes_v7(monkeypatch):
    for name in rp.V7:
        monkeypatch.setattr(rp, name, getattr(rp, name))
    v6 = {name: getattr(rp, name) for name in rp.V7}
    rp.use_protocol('v7')
    assert rp.PROTOCOL == 'rolling-origin-v7'
    rp.use_protocol('v6')
    assert {name: getattr(rp, name) for name in rp.V7} == v6 and rp.PROTOCOL == 'rolling-origin-v6'


def test_hybrid_grid_end_flags_cover_alpha_and_strategy_candidate_ends():
    def flags(alpha, strategy, candidate_k, omega=.5, gamma=1.25):
        return rp.hybrid_at_grid_boundary(dict(alpha=alpha, strategy=strategy, candidate_k=candidate_k,
                                               fusion='rrf', omega=omega, gamma=gamma))

    assert flags(.75, 'balanced_union', 20000) == {'alpha': True, 'candidate_k': True, 'omega': False, 'gamma': False}
    assert flags(.5, 'blend', 20000) == {'alpha': False, 'candidate_k': False, 'omega': False, 'gamma': False}
    assert flags(.25, 'blend', 'all', .9, 1.)['candidate_k'] and flags(.25, 'blend', 'all', .9, 1.)['alpha']
    assert flags(.5, 'balanced_union', 500, .1, 1.5) == {'alpha': False, 'candidate_k': True, 'omega': True,
                                                         'gamma': True}
