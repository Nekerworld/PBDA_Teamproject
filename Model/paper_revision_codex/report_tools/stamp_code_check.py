"""Compare the code hashes stamped into every *_artifacts.json of a result root with its plan's code hashes.

Each training stage stamps the hashes of the top-level non-test .py files it ran with (data_protocol.stamp_artifacts),
so a match shows each stage ran with the plan's code at the moment it was stamped (an edit reverted before
the stamp would not show).

Stamps hash the raw bytes on disk. A plan with code_hash_scheme 'sha256-raw' (v6, v7) holds the same kind of hash,
so they are compared directly. A 'sha256-lf' plan hashes line-ending-normalised bytes, which a raw stamp cannot be
compared with; for it each file is read from the plan's git commit, checked against the plan hash, and the stamp
must equal that content with LF or CRLF line endings.

Usage: python report_tools/stamp_code_check.py --root audit_runs/v7 --output audit/cycle8/stamp_code_v7.json
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def accepted_stamps(plan):
    """name -> set of raw-byte hashes a stamp may have under this plan."""
    code, scheme = plan['code_sha256'], plan.get('code_hash_scheme', 'sha256-raw')
    if scheme == 'sha256-raw':
        return {name: {digest} for name, digest in code.items()}
    if scheme != 'sha256-lf':
        raise ValueError(f'Unknown code hash scheme {scheme}')
    git = lambda *a: subprocess.run(['git', *a], cwd=PROJECT, capture_output=True, check=True).stdout
    prefix, commit = git('rev-parse', '--show-prefix').decode().strip(), plan['git']['commit']
    accepted = {}
    for name, digest in code.items():
        data = git('show', f'{commit}:{prefix}{name}').replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'{name} at {commit} does not match the plan hash')
        accepted[name] = {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest()}
    return accepted


def main(root, output):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    accepted = accepted_stamps(plan)
    files = sorted(root.rglob('*_artifacts.json'))
    compared, mismatches = 0, []
    for path in files:
        stamped = json.loads(path.read_text(encoding='utf-8')).get('source_hashes') or {}
        for name, digest in stamped.items():
            compared += 1
            if digest not in accepted.get(name, ()):
                mismatches.append({'artifact': path.relative_to(root).as_posix(), 'file': name})
    result = {'root': root.as_posix(), 'code_hash_scheme': plan.get('code_hash_scheme', 'sha256-raw'),
              'artifact_files': len(files), 'hashes_compared': compared,
              'mismatch_count': len(mismatches), 'mismatches': mismatches}
    text = json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding='utf-8')
    print(text)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
