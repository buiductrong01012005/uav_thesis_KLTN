"""Export lightweight review and complete episode-boundary restart bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def export(runs, output):
    runs,output = Path(runs),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    statuses = {}
    for path in sorted(runs.rglob('status.json')):
        status = json.loads(path.read_text())
        if status.get('network_version')=='bs5_network_v3_1':
            statuses[str(path.parent.relative_to(runs))] = status
    files = sorted(p for p in runs.rglob('*') if p.is_file() and p.suffix not in {'.lock','.tmp','.pyc'}
                   and '__pycache__' not in p.parts and p.name!='.DS_Store')
    artifacts = {}
    for name,include_resume in [('NETV3_TCF_REVIEW.zip',False),('NETV3_TCF_RESUME.zip',True)]:
        dest = output/name
        temp = dest.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                if include_resume or path.name!='resume.pt':
                    archive.write(path,path.relative_to(runs))
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise IOError('Corrupt export')
        temp.replace(dest)
        artifacts[name] = dict(bytes=dest.stat().st_size,sha256=file_sha(dest))
    index = dict(version='bs5_network_v3_export_1',status=statuses,files=artifacts,
        primary_selection='best.pt by fixed-seed mean TCF score; utility supplementary',
        resume='Keep NETV3_TCF_RESUME.zip plus the same source Dataset. Old utility and TCF-v1 resumes are incompatible with this new network.',
        completion='Read status for each preset/arm; paused_at_episode_boundary is not a completed 120k run.')
    dest = output/'NETV3_TCF_INDEX.json'
    dest.with_suffix('.tmp').write_text(json.dumps(index,indent=2)+'\n')
    dest.with_suffix('.tmp').replace(dest)
    print(json.dumps(index,indent=2),flush=True)
    return index


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    export(args.runs,args.output)
