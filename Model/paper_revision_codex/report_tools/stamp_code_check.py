"""Compare the code hashes stamped into every *_artifacts.json of a result root with its plan's code hashes.

Each training stage stamps the hashes of the top-level non-test .py files it ran with (data_protocol.stamp_artifacts),
so a match shows the run trained with the plan's code even if the code on disk was edited later.

Usage: python report_tools/stamp_code_check.py --root audit_runs/v7 --output audit/cycle8/stamp_code_v7.json
"""
import argparse
import json
from pathlib import Path


def main(root, output):
    root = Path(root)
    code = json.loads((root / 'plan.json').read_text(encoding='utf-8'))['code_sha256']
    files = sorted(root.rglob('*_artifacts.json'))
    compared, mismatches = 0, []
    for path in files:
        stamped = json.loads(path.read_text(encoding='utf-8')).get('source_hashes') or {}
        for name, digest in stamped.items():
            compared += 1
            if code.get(name) != digest:
                mismatches.append({'artifact': path.relative_to(root).as_posix(), 'file': name})
    result = {'root': root.as_posix(), 'artifact_files': len(files), 'hashes_compared': compared,
              'mismatch_count': len(mismatches), 'mismatches': mismatches}
    text = json.dumps(result, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
