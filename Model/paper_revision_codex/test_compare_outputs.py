"""compare_outputs.py: line endings are ignored, #keys limits a JSON comparison, .gitattributes is skipped and a
missing re-generated file counts as different."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'report_tools'))
import compare_outputs as co


def test_compare_outputs(tmp_path):
    new, old = tmp_path / 'new', tmp_path / 'old'
    for d in (new, old):
        d.mkdir()
    (old / 'a.csv').write_bytes(b'x,y\r\n1,2\r\n')
    (new / 'a.csv').write_bytes(b'x,y\n1,2\n')
    (old / '.gitattributes').write_text('* -text\n', encoding='utf-8')
    (old / 'p.json').write_text(json.dumps({'commit': 'c', 'added': ['t1']}), encoding='utf-8')
    (new / 'p.json').write_text(json.dumps({'commit': 'c', 'added': ['t1', 't2']}), encoding='utf-8')
    assert co.main([f'{new}={old}'], None)['files'] == 2  # a.csv, p.json; .gitattributes skipped
    assert not co.main([f'{new / "p.json"}={old / "p.json"}'], None)['all_equal']
    result = co.main([f'{new / "a.csv"}={old / "a.csv"}', f'{new / "p.json"}={old / "p.json"}#commit'], None)
    assert result['all_equal'] and result['rows'][1]['compared_keys'] == ['commit']
    assert not co.main([f'{new / "missing.csv"}={old / "a.csv"}'], None)['all_equal']


def test_compare_outputs_cannot_pass_by_mistake(tmp_path):
    new, old, empty = tmp_path / 'new', tmp_path / 'old', tmp_path / 'empty'
    for d in (new, old, empty):
        d.mkdir()
    (empty / '.gitattributes').write_text('* -text\n', encoding='utf-8')
    (new / 'a.csv').write_text('x\n', encoding='utf-8')
    (old / 'a.csv').write_text('x\n', encoding='utf-8')
    result = co.main([f'{new}={empty}'], None)
    assert result['files'] == 1 and not result['all_equal']  # a.csv exists only in NEW
    assert not co.main([f'{tmp_path / "nothing"}={empty}'], None)['all_equal']  # zero files compared
    with pytest.raises(FileNotFoundError):
        co.main([f'{new}={tmp_path / "typo"}'], None)
    with pytest.raises(ValueError, match='JSON'):
        co.main([f'{new / "a.csv"}={old / "a.csv"}#commit'], None)
    for a, b, equal in [({'v': 1}, {'v': 1.0}, False), ({'v': 1}, {'v': True}, False),
                        ({'v': float('nan')}, {'v': float('nan')}, True), ({'v': [1, 2]}, {'v': [1, 2]}, True)]:
        (new / 'p.json').write_text(json.dumps(a), encoding='utf-8')
        (old / 'p.json').write_text(json.dumps(b), encoding='utf-8')
        row = co.main([f'{new / "p.json"}={old / "p.json"}'], None)['rows'][0]
        assert row['equal'] is equal and (row['differing_top_level_keys'] == []) is equal
