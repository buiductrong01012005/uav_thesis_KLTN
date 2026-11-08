"""Physical parity, unchanged controllers, exact resume, raw audit and held-out guards."""
from pathlib import Path
from copy import deepcopy
import json,tempfile,unittest,zipfile
import numpy as np
from scripts.evaluation import run_fourarm400k_heuristics as run
from scripts.analysis.report_fourarm400k_heuristics import scan_inputs,report
from scripts.verification.verify_fourarm400k_heuristics import run as preflight
from scripts.notebooks.build_fourarm400k_heuristics import build
from fourarm400k_fixture import load_api
class HeuristicTests(unittest.TestCase):
 def test_protocol_matches_drl_environment_and_seed_partitions(self):
    api=load_api();fingerprints=[]
    for label,arm in run.LABELS.items():
        p=run.make_protocol(label,1);q=run.make_protocol(label,2)
        self.assertEqual(p['target_seeds']+q['target_seeds'],list(range(930201,930211)));fingerprints.extend([p['comparison_fingerprint'],q['comparison_fingerprint']])
        t=api['load_protocol'](api['parser']().parse_args(['--arm',arm,'--output','fixture']))
        for key in ['service','reward_weights','coverage_floor_mbps','energy_config','collision_config','fair_config','training_env_version','obs_dim','state_dim']:self.assertEqual(p['training_protocol'][key],t[key],key)
    self.assertEqual(len(set(fingerprints)),1)
    with self.assertRaises(ValueError):run.make_protocol('RANDOM',1,seeds=[930206])
    with self.assertRaises(ValueError):run.make_protocol('RANDOM',1,seeds=[930201,930201])
 def test_real_environment_parity_with_drl_factory(self):
    api=load_api()
    for label,arm in run.LABELS.items():
        p=run.make_protocol(label,1,True);t=api['load_protocol'](api['parser']().parse_args(['--arm',arm,'--output','fixture']))
        a=run.make_env(p);b=api['make_env'](t,horizon=96);x=a.reset(seed=940201,assignment_seed=940201);y=b.reset(seed=940201,assignment_seed=940201)
        for _ in range(40):
            np.testing.assert_array_equal(x['obs'],y['obs']);np.testing.assert_array_equal(x['action_mask'],y['action_mask'])
            acts=[int(np.flatnonzero(m)[-1]) for m in x['action_mask']];x=a.step(acts);y=b.step(acts)
            np.testing.assert_array_equal(a.energy_j,b.energy_j);self.assertEqual(x['reward'],y['reward'])
 def test_frozen_controller_actions_and_history_map_unused(self):
    for label in run.LABELS:
        p=run.make_protocol(label,1,True);state=run.start_state(p,940201);original=deepcopy(state['controller'].inner)
        other=deepcopy(state['controller']);modified=state['current']['obs'].copy();modified[:,148:]=.9
        self.assertEqual(state['controller'].act(state['current']['obs'],state['current']['action_mask']),other.act(modified,state['current']['action_mask']))
        state=run.start_state(p,940201)
        for _ in range(25):
            obs,mask=state['current']['obs'],state['current']['action_mask'];a=state['controller'].act(obs,mask);b=original.act(obs[:,:148],mask)
            self.assertEqual(a,b);state['current']=state['env'].step(a)
 def test_resume_pickle_integrity_and_protocol_guard(self):
    p=run.make_protocol('RANDOM',1,True);state=run.start_state(p,940201);run.advance(state,p)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);run.save_state(folder,p,state);loaded=run.load_state(folder,p);self.assertEqual(len(loaded['rows']),1)
        wrong=deepcopy(p);wrong['label']='DYNAMIC_AR'
        with self.assertRaises(ValueError):run.load_state(folder,wrong)
        progress=json.loads((folder/'progress.json').read_text());f=folder/progress['resume_file'];f.write_bytes(f.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):run.load_state(folder,p)
 def test_notebooks_compile_offline_without_torch(self):
    with tempfile.TemporaryDirectory() as tmp:
        build(tmp);paths=list(Path(tmp).glob('*.ipynb'));self.assertEqual(len(paths),7)
        for path in paths:
            nb=json.loads(path.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code');compile(code,str(path),'exec')
            self.assertIn('REQUIRE_TORCH=False',code);self.assertNotIn('pip install',code)
            if 'MERGE' not in path.name:self.assertIn('RUN_PREFLIGHT=True',code)
 def test_actual_three_controllers_resume_raw_pairing_and_merge(self):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);traces=[]
        for label in run.LABELS:
            out=root/label;preflight(label,out)
            folder=out/'resume';p=run.make_protocol(label,1,True);p['runtime']=run.runtime();run.atomic_json(out/'protocol.json',p)
            folder.rename(out/'seed940201');traces.append(json.loads((out/'seed940201/summary.json').read_text())['ue_trace_sha256'])
        self.assertEqual(len(set(traces)),1)
        with self.assertRaises(ValueError):scan_inputs(root)
        results,p,_=scan_inputs(root,allow_diagnostic=True);self.assertEqual(len(results),3)
        meta=report(root,root/'diagnostic_report',True);self.assertEqual(meta['status'],'diagnostic_only')
        folder=root/'RANDOM/seed940201';(folder/'slots.csv').write_text((folder/'slots.csv').read_text()+'corrupt\n')
        with self.assertRaises(ValueError):scan_inputs(root,True)
 def test_main_run_partial_export_resume_and_scanner(self):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'worker'
        def args(stop):return run.parser().parse_args(['--label','RANDOM','--part','1','--output',str(root),'--diagnostic','--save-every','7','--log-every','96','--stop-after-slots',str(stop)])
        run.run(args(9));self.assertEqual(json.loads((root/'status.json').read_text())['status'],'paused')
        with zipfile.ZipFile(root.parent/(root.name+'_REVIEW.zip')) as z:self.assertFalse(any('resume_' in n and n.endswith('.pkl') for n in z.namelist()))
        run.run(args(0));self.assertEqual(json.loads((root/'status.json').read_text())['status'],'complete')
        self.assertFalse((root/'seed940201/progress.json').exists());run.run(args(0))
        results,_,_=scan_inputs(root,True);self.assertEqual(len(results),1)
if __name__=='__main__':unittest.main(verbosity=2)
