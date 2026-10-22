"""A GNN selection job must use the inner validation split of its own fold, not that of another fold."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolling_protocol as rp


def manifest(folder, **fields):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'manifest.json').write_text(json.dumps(fields), encoding='utf-8')
    return folder


def test_selection_spec_refuses_the_validation_split_of_another_fold(tmp_path):
    own = manifest(tmp_path / 'F2_mid' / 'prepared' / 'validation', role='validation')
    other = manifest(tmp_path / 'F1_hist' / 'prepared' / 'validation', role='validation')
    job = tmp_path / 'F2_mid' / 'gnn_val' / 'gnn_x'
    assert rp.gnn_spec(job, own, {}, 42, 'select', 4)['source'] == str(own.resolve())
    with pytest.raises(RuntimeError, match='its own fold'):
        rp.gnn_spec(job, other, {}, 42, 'select', 4)


def test_selection_spec_refuses_its_own_fold_folder_without_the_validation_role(tmp_path):
    own = manifest(tmp_path / 'F2_mid' / 'prepared' / 'validation', role='test')
    with pytest.raises(RuntimeError, match='its own fold'):
        rp.gnn_spec(tmp_path / 'F2_mid' / 'gnn_val' / 'gnn_x', own, {}, 42, 'select', 4)
