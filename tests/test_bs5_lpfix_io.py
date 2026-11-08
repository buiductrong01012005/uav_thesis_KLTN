"""LPFIX notebook IO and raw-audited legacy/guarded diagnostic merge."""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import ast,hashlib,json,shutil,tempfile,unittest,zipfile,sys
from scripts.notebooks.build_fourarm400k_lpfix import build
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.analysis.fourarm400k_lp_compat import manifest,digest
from scripts.analysis.report_fourarm400k_multiseed import scan
from test_bs5_lp_roundoff import legacy,radio
class LPFixIOTests(unittest.TestCase):
 def test_fresh_seed552_and_later_notebooks_compile_offline(self):
  with tempfile.TemporaryDirectory() as tmp:
   build(tmp);files=list(Path(tmp).rglob('*.ipynb'));self.assertEqual(len(files),29)
   for path in files:
    nb=json.loads(path.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');compile(code,str(path),'exec')
    self.assertIn('UAV_MULTI400K_LPFIX_CPU_DATASET.zip',code);self.assertNotIn('pip install',code)
    if path.name.startswith('F01'):
     self.assertIn('TRAINING_SEED=552',code);self.assertIn('TOTAL_STEPS=399600',code);self.assertIn('DRL_H_ONLY_S552_400K_LPFIX',code);self.assertIn('RESUME_PATH=""',code);self.assertIn('test_bs5_lp_roundoff.py',code)
     run_cell=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'command=[sys.executable' in ''.join(c['source']))
     self.assertTrue(run_cell.startswith('os.environ["UAV_LP_DIAGNOSTIC_DIR"]'))
 def test_guarded_dataset_zip_and_auto_extracted(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);build(root/'nb');nb=json.loads((root/'nb/F01_H_ONLY_S552_400K_LPFIX_CPU.ipynb').read_text());bootstrap=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'def load_source' in ''.join(c['source']))
   ns=dict(Path=Path,json=json,hashlib=hashlib,zipfile=zipfile,shutil=shutil,tempfile=tempfile)
   tree=ast.parse(bootstrap);exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'bootstrap','exec'),ns)
   inp=root/'input';work=root/'work';inp.mkdir();work.mkdir()
   files={'uav_thesis/'+n:b'fixture' for n in ['scripts/training/run_bs5_fourarm400k.py','configs/bs5_fourarm400k.json','scripts/analysis/compare_bs5_fourarm400k_training.py']}
   m=dict(version='fourarm400k_lpfix_dataset_v1',files_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()});files['LPFIX400K_DATASET_MANIFEST.json']=json.dumps(m).encode();z=inp/'UAV_MULTI400K_LPFIX_CPU_DATASET.zip'
   with zipfile.ZipFile(z,'w') as a:
    for k,v in files.items():a.writestr(k,v)
   source=ns['load_source'](inp,work)
   with zipfile.ZipFile(z) as a:a.extractall(inp/'auto')
   self.assertEqual(source,ns['load_source'](inp,work));z.unlink();self.assertEqual(source,ns['load_source'](inp,work))
 def test_raw_merge_accepts_legacy_and_guarded_but_rejects_unknown_source(self):
  rel=next(iter(manifest()['files']));record=manifest()['files'][rel]
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for label,is_old in [('DYNAMIC_AR',True),('HEURISTIC_HO',False)]:
    p=core.make_protocol(label,1,True);p['runtime']=core.runtime()
    if is_old:
     p['source_sha256'][rel]=record['legacy_sha256'];p['environment_contract']['physics_source_sha256'][rel]=record['legacy_sha256'];p['environment_fingerprint']=digest(p['environment_contract'])
     keys=['version','config','training_protocol','source_sha256','environment_contract','environment_fingerprint'];p['comparison_fingerprint']=digest({k:p[k] for k in keys})
    folder=root/label;folder.mkdir();core.atomic_json(folder/'protocol.json',p)
    with ExitStack() as stack:
     if is_old:
      guarded_allocator=radio.allocate_flows
      for module in list(sys.modules.values()):
       if module is not None and getattr(module,'__name__','').startswith('src.') and getattr(module,'allocate_flows',None) is guarded_allocator:stack.enter_context(patch.object(module,'allocate_flows',legacy.allocate_flows))
     state=core.start_state(p,940201)
     for _ in range(96):core.advance(state,p)
    dest=folder/'seed940201';dest.mkdir();core.save_seed(dest,p,state,True)
   rows,protocols,provenance=scan(root,True);self.assertEqual(len(rows),2);self.assertEqual({r['numerical_variant'] for r in provenance},{'legacy','guarded'})
   self.assertEqual(len({p['environment_fingerprint'] for p in protocols}),2)
   path=root/'DYNAMIC_AR/protocol.json';p=json.loads(path.read_text());p['source_sha256']['src/envs/bs5_tcf_env.py']='unknown'
   keys=['version','config','training_protocol','source_sha256','environment_contract','environment_fingerprint'];p['comparison_fingerprint']=digest({k:p[k] for k in keys});core.atomic_json(path,p)
   with self.assertRaises(ValueError):scan(root,True)
if __name__=='__main__':unittest.main(verbosity=2)
