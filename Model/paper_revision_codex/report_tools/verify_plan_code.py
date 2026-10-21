"""Check a plan's code_sha256 against git, independent of how the checkout converted line endings.

For every hashed file: does the plan hash equal the file at the plan's commit (as stored in git, with LF
or with CRLF line endings)? And does the current working file still have that content (line endings
ignored)? Plans before the 'sha256-lf' scheme hashed raw working-tree bytes, so the same commit can
hash differently on two checkouts; this tells "only line endings differ" apart from "code changed".

Usage: python report_tools/verify_plan_code.py --plan audit/results_v6/plan.json --output <json>
Exit status 1 means a plan hash matches no form of the file at the plan commit. Files changed since the
plan are only listed (changed_since_plan) unless --require-unchanged is given.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(['git', *args], cwd=HERE, capture_output=True, check=True).stdout


def lf(data):
    return data.replace(b'\r\n', b'\n')


def main(plan_path, output):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    commit = plan['git']['commit']
    prefix = git('rev-parse', '--show-prefix').decode().strip()
    scheme = plan.get('code_hash_scheme', 'sha256-raw')
    rows = []
    for name, stored in sorted(plan['code_sha256'].items()):
        blob = lf(git('show', f'{commit}:{prefix}{name}'))
        forms = {'lf': blob} if scheme == 'sha256-lf' else {'lf': blob, 'crlf': blob.replace(b'\n', b'\r\n')}
        matched = [form for form, data in forms.items() if hashlib.sha256(data).hexdigest() == stored]
        current = HERE / name
        rows.append({'file': name, 'plan_hash_matches_commit': bool(matched), 'form': matched[0] if matched else None,
                     'working_file_unchanged': current.exists() and lf(current.read_bytes()) == blob})
    hashed = set(plan['code_sha256'])
    added = sorted(p.relative_to(HERE).as_posix() for p in list(HERE.glob('*.py')) + list(HERE.glob('audit_tools/*.py'))
                   if p.relative_to(HERE).as_posix() not in hashed)
    result = {'plan': Path(plan_path).as_posix(), 'commit': commit, 'scheme': scheme, 'files': len(rows),
              'all_plan_hashes_match_commit': all(r['plan_hash_matches_commit'] for r in rows),
              'changed_since_plan': [r['file'] for r in rows if not r['working_file_unchanged']],
              'added_since_plan': added, 'rows': rows}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-unchanged', action='store_true',
                        help='also fail when a hashed file differs from the plan commit (default: report only)')
    args = parser.parse_args()
    result = main(args.plan, args.output)
    ok = result['all_plan_hashes_match_commit'] and not (args.require_unchanged and result['changed_since_plan'])
    raise SystemExit(0 if ok else 1)
