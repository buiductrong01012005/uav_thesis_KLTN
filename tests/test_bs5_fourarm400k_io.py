"""Offline notebook bootstrap/resume and validation overlay integrity tests."""
from pathlib import Path
import csv,hashlib,io,json,tempfile,unittest,zipfile
from copy import deepcopy
from fourarm400k_fixture import ROOT,load_api,atomic_json
from scripts.notebooks.build_fourarm400k import BOOTSTRAP,RESUME,build
from scripts.analysis.compare_bs5_fourarm400k_training import collect,compare,digest
API=load_api()
def execute_functions(source,namespace):
    import ast
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'notebook-functions','exec'),namespace)
def csv_bytes(rows):
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);return stream.getvalue().encode()
class NotebookIOTests(unittest.TestCase):
 def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.input=self.root/'input';self.work=self.root/'work';self.input.mkdir();self.work.mkdir()
    import shutil
    self.ns=dict(Path=Path,json=json,hashlib=hashlib,zipfile=zipfile,shutil=shutil,tempfile=tempfile,WORKING=self.work)
    execute_functions(BOOTSTRAP,self.ns);execute_functions(RESUME,self.ns)
 def tearDown(self):self.tmp.cleanup()
 def dataset(self):
    names=['scripts/training/run_bs5_fourarm400k.py','configs/bs5_fourarm400k.json','scripts/analysis/compare_bs5_fourarm400k_training.py']
    files={'uav_thesis/'+n:(ROOT/n).read_bytes() for n in names}
    manifest=dict(version='drl_fourarm400k_dataset_v1',files_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    files['DRL400K_DATASET_MANIFEST.json']=json.dumps(manifest).encode();return files
 def zip(self,path,files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
 def test_dataset_zip_directory_and_both(self):
    files=self.dataset();z=self.input/'UAV_DRL400K_DATASET.zip';self.zip(z,files)
    source=self.ns['load_source'](self.input,self.work);self.assertTrue((source/'configs/bs5_fourarm400k.json').is_file())
    with zipfile.ZipFile(z) as archive:archive.extractall(self.input/'auto_extracted')
    self.assertEqual(source,self.ns['load_source'](self.input,self.work))
    z.unlink();self.assertEqual(source,self.ns['load_source'](self.input,self.work))
 def test_missing_dataset_and_tamper(self):
    with self.assertRaises(FileNotFoundError):self.ns['load_source'](self.input,self.work)
    files=self.dataset();files['uav_thesis/configs/bs5_fourarm400k.json']=b'changed';self.zip(self.input/'UAV_DRL400K_DATASET.zip',files)
    with self.assertRaises(RuntimeError):self.ns['load_source'](self.input,self.work)
 def test_duplicate_dataset_versions(self):
    files=self.dataset();self.zip(self.input/'UAV_DRL400K_DATASET.zip',files)
    folder=self.input/'second';folder.mkdir();m=json.loads(files['DRL400K_DATASET_MANIFEST.json']);m['changed']=True;(folder/'DRL400K_DATASET_MANIFEST.json').write_text(json.dumps(m))
    with self.assertRaises(RuntimeError):self.ns['load_source'](self.input,self.work)
 def test_traversal_symlink_and_duplicate_zip_rejected(self):
    path=self.input/'bad.zip'
    for name in ['../escape','/absolute','bad\\name']:
        with zipfile.ZipFile(path,'w') as z:
            # Preserve the invalid member name; Windows ZipInfo normalizes backslashes.
            info=zipfile.ZipInfo();info.filename=name;z.writestr(info,b'x')
        with self.assertRaises(ValueError):self.ns['extract_safe'](path,self.work)
    with zipfile.ZipFile(path,'w') as z:
        info=zipfile.ZipInfo('link');info.external_attr=(0o120777<<16);z.writestr(info,b'/tmp/out')
    with self.assertRaises(ValueError):self.ns['extract_safe'](path,self.work)
 def resume_files(self,arm='C-R',version='bs5_fourarm400k_v1',include=True):
    files={'run/protocol.json':json.dumps(dict(version=version,arm=arm,training_seed=551,total_steps=399600,diagnostic=False,fair_config=dict(mode='instant'))).encode()}
    if include:files['run/resume.pt']=b'fixture,not a model'
    return files
 def test_resume_zip_and_directory(self):
    files=self.resume_files();z=self.input/'RESUME.zip';self.zip(z,files);out=self.work/'restored'
    self.ns['restore_training'](z,out,'C-R',551,399600);self.assertTrue((out/'resume.pt').exists())
    with zipfile.ZipFile(z) as archive:archive.extractall(self.input/'extracted')
    self.ns['restore_training'](self.input/'extracted',self.work/'restored2','C-R',551,399600)
 def test_resume_wrong_arm_old_version_review_and_existing_rejected(self):
    for i,files in enumerate([self.resume_files('C-N'),self.resume_files(version='bs5_temporal_fair_v1'),self.resume_files(include=False)]):
        path=self.input/f'{i}.zip';self.zip(path,files)
        with self.assertRaises(ValueError):self.ns['restore_training'](path,self.work/f'out{i}','C-R',551,399600)
    existing=self.work/'existing';existing.mkdir()
    with self.assertRaises(RuntimeError):self.ns['restore_training'](self.input/'0.zip',existing,'C-R',551,399600)
 def test_five_notebooks_compile_cpu_offline(self):
    build(self.work/'notebooks');paths=list((self.work/'notebooks').glob('*.ipynb'));self.assertEqual(len(paths),5)
    for path in paths:
        nb=json.loads(path.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');compile(code,str(path),'exec')
        self.assertNotIn('pip install',code);self.assertNotIn('cuda',code)
        if 'COMPARE' not in path.name:self.assertIn('TOTAL_STEPS=399600',code);self.assertIn('RUN_PREFLIGHT=True',code)
 def output_files(self,arm,complete=True):
    p=API['load_protocol'](API['parser']().parse_args(['--arm',arm,'--output','fixture']))
    step=399600 if complete else 12000
    status=dict(training_env_version=p['version'],arm=arm,status='complete' if complete else 'paused_at_episode_boundary',committed_training_steps=step,target_steps=399600,best_by_metric={'reward_total':dict(step=12000,score=.4)})
    rows=[dict(step=s,seed=seed,reward_total=.4,throughput_mbps=120,qos_ues=25,service_coverage_ues=40,jain_time_average=.8,p05_ue_mean_rate_mbps=.4,rolling_unique_ge1_5min=80,longest_zero_gap_slots_max=100,joules_per_mbit=18,blocked_per_policy_move=.07,effective_unavailable_fraction=.15,zero_service_slot_fraction=.03) for s in (p['evaluation_steps'] if complete else [0,12000]) for seed in p['validation_seeds']]
    return {'protocol.json':json.dumps(p).encode(),'status.json':json.dumps(status).encode(),'training.csv':csv_bytes([dict(step=12000,mean_loss=.1)]),'validation.csv':csv_bytes(rows)}
 def test_overlay_zip_directory_duplicate_and_partial(self):
    files=self.output_files('C-R',False);self.zip(self.input/'CR_REVIEW.zip',{'run/'+k:v for k,v in files.items()})
    with zipfile.ZipFile(self.input/'CR_REVIEW.zip') as z:z.extractall(self.input/'auto')
    runs,_=collect(self.input);self.assertEqual(list(runs),['C-R'])
    meta=compare(self.input,self.work/'comparison');self.assertEqual(meta['status'],'partial_training');self.assertEqual(len(meta['missing']),3)
 def test_complete_without_all_validation_rows_rejected(self):
    files=self.output_files('C-R');rows=files['validation.csv'].splitlines(keepends=True);files['validation.csv']=b''.join(rows[:-1]);self.zip(self.input/'CR.zip',{'run/'+k:v for k,v in files.items()})
    with self.assertRaises(ValueError):collect(self.input)
 def test_overlay_full_and_conflicting_protocol_rejected(self):
    for arm in ['C-N','C-R','C-H','H-only']:self.zip(self.input/f'{arm}.zip',{'run/'+k:v for k,v in self.output_files(arm).items()})
    meta=compare(self.input,self.work/'comparison');self.assertEqual(meta['status'],'all_training_complete')
    files=self.output_files('C-N');p=json.loads(files['protocol.json']);p['training_seed']=999;common={k:v for k,v in p.items() if k not in {'arm','policy_actions','runtime','torch_version','comparison_fingerprint'}};p['comparison_fingerprint']=digest(common);files['protocol.json']=json.dumps(p).encode();self.zip(self.input/'C-N.zip',{'run/'+k:v for k,v in files.items()})
    with self.assertRaises(ValueError):collect(self.input)
if __name__=='__main__':unittest.main(verbosity=2)
