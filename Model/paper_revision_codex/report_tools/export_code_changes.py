"""Export the actual code diff without altering any training source."""
import argparse,difflib,hashlib,json
from pathlib import Path


def main(root):
    root=Path(root);before=root/'source_before_hybrid_20261017';parts=[];manifest={}
    for current in sorted(root.glob('*.py')):
        prior=before/current.name
        old=prior.read_text(encoding='utf-8-sig').splitlines(keepends=True) if prior.exists() else []
        new=current.read_text(encoding='utf-8-sig').splitlines(keepends=True)
        if old==new:continue
        parts.extend(difflib.unified_diff(old,new,fromfile='before/'+current.name if prior.exists() else '/dev/null',tofile='after/'+current.name))
        manifest[current.name]={'before_sha256':hashlib.sha256(prior.read_bytes()).hexdigest() if prior.exists() else None,
            'after_sha256':hashlib.sha256(current.read_bytes()).hexdigest()}
    (root/'CODE_CHANGES.patch').write_text(''.join(parts),encoding='utf-8')
    (root/'CODE_CHANGES_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Changed/new Python files:',', '.join(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
