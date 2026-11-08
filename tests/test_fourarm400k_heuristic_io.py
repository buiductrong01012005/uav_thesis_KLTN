"""Notebook dataset and resume paths, with real CPU benchmark evidence."""
from pathlib import Path
import ast,hashlib,json,shutil,tempfile,unittest,zipfile
from scripts.evaluation import run_fourarm400k_heuristics as run
from scripts.analysis.report_fourarm400k_heuristics import scan_inputs
from scripts.notebooks.build_fourarm400k_heuristics import BOOTSTRAP,RESUME
ROOT=Path(__file__).resolve().parents[1]
class NotebookIOTests(unittest.TestCase):
 def setUp(self):
    self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.input=self.root/'input';self.work=self.root/'work';self.input.mkdir();self.work.mkdir()
    self.ns=dict(Path=Path,json=json,hashlib=hashlib,zipfile=zipfile,shutil=shutil,tempfile=tempfile,WORKING=self.work)
    for source in [BOOTSTRAP,RESUME]:
        tree=ast.parse(source);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'notebook-IO','exec'),self.ns)
 def tearDown(self):self.temp.cleanup()
 def zip(self,path,files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for k,v in files.items():z.writestr(k,v)
 def dataset(self):
    names=['scripts/evaluation/run_fourarm400k_heuristics.py','configs/bs5_fourarm400k.json','scripts/analysis/report_fourarm400k_heuristics.py']
    files={'uav_thesis/'+n:(ROOT/n).read_bytes() for n in names};m=dict(version='heur_fourarm400k_dataset_v1',files_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()});files['HEUR400K_DATASET_MANIFEST.json']=json.dumps(m).encode();return files
 def test_zip_extracted_and_both_with_checksum(self):
    z=self.input/'UAV_HEUR400K_BENCH_DATASET.zip';self.zip(z,self.dataset());src=self.ns['load_source'](self.input,self.work)
    with zipfile.ZipFile(z) as a:a.extractall(self.input/'auto')
    self.assertEqual(src,self.ns['load_source'](self.input,self.work));z.unlink();self.assertEqual(src,self.ns['load_source'](self.input,self.work))
    (src/'configs/bs5_fourarm400k.json').write_text('tampered')
    with self.assertRaises(RuntimeError):self.ns['load_source'](self.input,self.work)
 def test_dataset_missing_and_different_versions(self):
    with self.assertRaises(FileNotFoundError):self.ns['load_source'](self.input,self.work)
    files=self.dataset();self.zip(self.input/'UAV_HEUR400K_BENCH_DATASET.zip',files);m=json.loads(files['HEUR400K_DATASET_MANIFEST.json']);m['changed']=True
    (self.input/'HEUR400K_DATASET_MANIFEST.json').write_text(json.dumps(m))
    with self.assertRaises(RuntimeError):self.ns['load_source'](self.input,self.work)
 def test_unsafe_zip_rejected(self):
    for name in ['../escape','/absolute','bad\\path']:
        z=self.input/'bad.zip'
        with zipfile.ZipFile(z,'w') as archive:
            # Keep the intentionally invalid name: Windows ZipInfo otherwise
            # normalizes the backslash before the extraction guard sees it.
            info=zipfile.ZipInfo();info.filename=name;archive.writestr(info,b'x')
        with self.assertRaises(ValueError):self.ns['extract_safe'](z,self.work)
 def production_resume(self):
    p=run.make_protocol('RANDOM',1);return {'worker/protocol.json':json.dumps(p).encode(),'worker/seed930201/summary.json':b'{"complete":false}','worker/seed930201/progress.json':b'{"fixture_only":true}'}
 def test_resume_zip_directory_and_wrong_worker(self):
    z=self.input/'RESUME.zip';self.zip(z,self.production_resume());self.ns['restore_benchmark'](z,self.work/'out','RANDOM',1,None)
    with zipfile.ZipFile(z) as a:a.extractall(self.input/'auto')
    self.ns['restore_benchmark'](self.input/'auto',self.work/'out2','RANDOM',1,None)
    for label,part in [('DYNAMIC_AR',1),('RANDOM',2)]:
        with self.assertRaises(ValueError):self.ns['restore_benchmark'](z,self.work/f'{label}{part}',label,part,None)
 def test_partial_review_not_resumable(self):
    files=self.production_resume();del files['worker/seed930201/progress.json'];z=self.input/'REVIEW.zip';self.zip(z,files)
    with self.assertRaises(ValueError):self.ns['restore_benchmark'](z,self.work/'out','RANDOM',1,None)
 def test_real_review_zip_and_extracted_duplicate(self):
    p=run.make_protocol('RANDOM',1,True);p['runtime']=run.runtime();root=self.input/'worker';root.mkdir();run.atomic_json(root/'protocol.json',p);state=run.start_state(p,940201)
    for _ in range(96):run.advance(state,p)
    folder=root/'seed940201';folder.mkdir();run.save_seed(folder,p,state,True);run.export(root)
    rows,_,_=scan_inputs(self.input,True);self.assertEqual(len(rows),1)
    summary=json.loads((folder/'summary.json').read_text());summary['throughput_mbps']+=1;run.atomic_json(folder/'summary.json',summary)
    with self.assertRaises(ValueError):scan_inputs(self.input,True)
if __name__=='__main__':unittest.main(verbosity=2)
