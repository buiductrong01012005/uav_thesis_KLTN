"""Merge the extra heuristic with original reviews using verified base provenance."""
from pathlib import Path
import argparse,io,json,stat,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis.report_bs5_heuristic_ho import generate,validate_protocols,LABELS

def matching_prefix(old,new):
    so,archive_o,prefix_o=old;sn,archive_n,prefix_n=new
    with zipfile.ZipFile(archive_o) as a,zipfile.ZipFile(archive_n) as b:
        with np.load(io.BytesIO(a.read(prefix_o+'trace.npz'))) as x:aa={k:x[k] for k in x.files}
        with np.load(io.BytesIO(b.read(prefix_n+'trace.npz'))) as x:bb={k:x[k] for k in x.files}
    if aa.keys()!=bb.keys():raise ValueError('Conflicting duplicate trace columns')
    n=min(so['slots'],sn['slots'])
    if not all(np.array_equal(aa[k][:n],bb[k][:n]) for k in aa):raise ValueError('Conflicting duplicate trajectory prefix')

def merge(archives,output,phase='test',allow_diagnostic=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh merge output folder; existing results are preserved')
    choices={};protocols={}
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                q=Path(info.filename)
                if q.is_absolute() or '..' in q.parts or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive member')
            for name in z.namelist():
                path=Path(name)
                if path.name!='summary.json' or len(path.parts)!=3 or not path.parts[1].startswith('seed') or path.parts[0] not in LABELS:continue
                label=path.parts[0];p=json.loads(z.read(label+'/protocol.json'))
                if p['phase']!=phase:continue
                if p['diagnostic'] and not allow_diagnostic:raise ValueError('Diagnostic run is not a production test')
                if label in protocols and protocols[label]!=p:raise ValueError('Conflicting label protocol')
                protocols[label]=p;s=json.loads(z.read(name))
                if s['label']!=label or s['experiment_fingerprint']!=p['experiment_fingerprint']:raise ValueError('Summary identity mismatch')
                key=label,s['seed'];candidate=(s,Path(archive),path.parent.as_posix()+'/')
                if key in choices:
                    matching_prefix(choices[key],candidate)
                    if s['slots']<=choices[key][0]['slots']:continue
                choices[key]=candidate
    if 'HEURISTIC_HO' not in protocols:raise ValueError('Attach at least one HEURISTIC_HO review')
    validate_protocols(protocols)
    output.mkdir(parents=True,exist_ok=True)
    for label,p in protocols.items():
        folder=output/label;folder.mkdir();(folder/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    for s,archive,prefix in choices.values():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith(prefix) and not name.endswith('/') and Path(name).name!='resume.pkl':
                    p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    generate(output);dest=output.parent/('MH_'+phase.upper()+'_WITH_HEURISTIC_HO.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output))
    with zipfile.ZipFile(dest) as z:
        if z.testzip():raise ValueError('Corrupt merged output')
    print(dest,flush=True);return dest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['pilot','test'],default='test');p.add_argument('--allow-diagnostic',action='store_true');a=p.parse_args()
    archives=[p for b in a.input for p in ([b] if b.is_file() else b.rglob('*_BENCH_REVIEW*.zip'))];merge(archives,a.output,a.phase,a.allow_diagnostic)
