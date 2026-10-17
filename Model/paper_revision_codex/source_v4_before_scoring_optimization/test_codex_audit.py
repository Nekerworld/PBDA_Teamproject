"""Regression checks for evidence integrity and sparsity; requires pandas only."""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
import pandas as pd
from audit_revision import locked_config, sha256


class RevisionChecks(unittest.TestCase):
    def test_sparsity_table_uses_unique_pairs_with_repeated_events(self):
        path = Path(__file__).parent/'experiments.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'write_dataset_statistics_tables')
        namespace = {'pd':pd, 'Path':Path, 'sys':sys}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        stages = ['raw_valid','train_before_filter','train_after_if','train_final','test_before_filter']
        rows = [dict(stage=s, users=2, items=2, events=8, unique_user_item_pairs=2,
                     view_events=6, addtocart_events=1, transaction_events=1, positive_events=2) for s in stages]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp)
            (root/'if_events').mkdir()
            pd.DataFrame(rows).to_csv(root/'if_events'/'dataset_statistics.csv',index=False)
            namespace['write_dataset_statistics_tables'](root, root, 'if_events')
            table=pd.read_csv(root/'table1_dataset_statistics.csv').set_index('Metric')
            self.assertEqual(table.loc['Sparsity (%)','4. Final Train Set'], '50.0000%')
            self.assertEqual(str(table.loc['Avg. Inter./User','4. Final Train Set']), '4.00')

    def test_config_inspection_never_imports_ml_module(self):
        self.assertEqual(locked_config(Path(__file__).parent/'experiments.py'),
                         {'primary':'if_events','test_size':0.1})

    def test_hash_detects_changed_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/'evidence.csv'
            path.write_bytes(b'original')
            self.assertEqual(sha256(path),hashlib.sha256(b'original').hexdigest())
            previous=sha256(path)
            path.write_bytes(b'changed')
            self.assertNotEqual(previous,sha256(path))


if __name__=='__main__':
    unittest.main()
