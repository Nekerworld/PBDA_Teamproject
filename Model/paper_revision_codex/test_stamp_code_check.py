"""stamp_code_check compares raw-byte stamps with an LF-normalised plan through the plan's git commit (needs git)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'report_tools'))
import stamp_code_check as sc


def test_lf_plan_accepts_either_line_ending_and_flags_other_code(tmp_path):
    git = lambda *a: subprocess.run(['git', *a], cwd=HERE, capture_output=True, check=True).stdout
    commit, prefix = git('rev-parse', 'HEAD').decode().strip(), git('rev-parse', '--show-prefix').decode().strip()
    names = ['policy_evaluation.py', 'data_protocol.py']
    content = {n: git('show', f'{commit}:{prefix}{n}').replace(b'\r\n', b'\n') for n in names}
    digest = lambda data: hashlib.sha256(data).hexdigest()
    (tmp_path / 'plan.json').write_text(json.dumps({'code_hash_scheme': 'sha256-lf', 'git': {'commit': commit},
                                                    'code_sha256': {n: digest(c) for n, c in content.items()}}),
                                        encoding='utf-8')
    stamps = {'policy_evaluation.py': digest(content['policy_evaluation.py'].replace(b'\n', b'\r\n')),
              'data_protocol.py': digest(content['data_protocol.py'])}
    (tmp_path / 'als_artifacts.json').write_text(json.dumps({'source_hashes': stamps}), encoding='utf-8')
    assert sc.main(tmp_path, None)['mismatch_count'] == 0
    stamps['data_protocol.py'] = digest(content['data_protocol.py'] + b'# edited\n')
    (tmp_path / 'als_artifacts.json').write_text(json.dumps({'source_hashes': stamps}), encoding='utf-8')
    result = sc.main(tmp_path, None)
    assert result['mismatch_count'] == 1 and result['mismatches'][0]['file'] == 'data_protocol.py'
