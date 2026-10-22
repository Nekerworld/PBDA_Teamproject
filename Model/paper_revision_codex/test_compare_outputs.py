"""compare_outputs.py: line endings are ignored, #keys limits a JSON comparison, .gitattributes is skipped and a
missing re-generated file counts as different."""
import json
import sys
from pathlib import Path

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
