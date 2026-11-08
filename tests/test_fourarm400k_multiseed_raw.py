"""Real env/controller diagnostic IO. The injected DRL controller is a named test fixture."""
from pathlib import Path
import json,tempfile,unittest
import numpy as np
from fourarm400k_fixture import load_api
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.evaluation import run_fourarm400k_drl_benchmark as drl
from scripts.analysis.report_fourarm400k_multiseed import scan,report
class FixtureActor:
 """Only verifies recording/restart, never represents a trained neural policy."""
 @property
 def last_stats(self):return {}
 def act(self,obs,mask):return [int(np.flatnonzero(m)[-1]) for m in mask]
class RawIOTests(unittest.TestCase):
 def test_real_two_heuristics_and_drl_adapter_partial_resume_and_strict_merge(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for label in ['DYNAMIC_AR','HEURISTIC_HO']:
    args=core.parser().parse_args(['--label',label,'--part','1','--output',str(root/label),'--diagnostic','--log-every','96'])
    core.run(args)
   meta=report(root,root/'heur_report',True,True);self.assertEqual(meta['status'],'diagnostic_only');self.assertEqual(meta['completed_runs'],2)
   self.assertTrue((root/'heur_report/main_metrics.png').is_file())
   with self.assertRaises(ValueError):scan(root)
   api=load_api();tp=api['load_protocol'](api['parser']().parse_args(['--arm','C-R','--training-seed','551','--output','fixture']))
   rec=dict(selected_step=399600,validation_score=.1,checkpoint_sha256='fixture-only-not-a-trained-model')
   def factory(label,part,diagnostic=False,seeds=None):return drl.make_protocol('C-R',551,part,rec,tp,'fixture-only-bundle',diagnostic,seeds)
   def starter(p,seed):return core.start_state(p,seed,controller=FixtureActor())
   def args(stop):
    a=core.parser().parse_args(['--label','RANDOM','--part','1','--output',str(root/'drl_fixture'),'--diagnostic','--log-every','96','--stop-after-slots',str(stop)]);a.label='DRL_C-R_S551';return a
   core.run(args(13),protocol_factory=factory,state_factory=starter)
   self.assertEqual(json.loads((root/'drl_fixture/status.json').read_text())['status'],'paused')
   core.run(args(0),protocol_factory=factory,state_factory=starter)
   results,_,_=scan(root,True);self.assertEqual(len(results),3)
   meta=report(root,root/'full_partial_report',False,True);self.assertEqual(meta['status'],'partial');self.assertEqual(meta['expected_runs'],14)
   self.assertFalse((root/'full_partial_report/aggregate.csv').exists())
   path=root/'DYNAMIC_AR/seed940201/slots.csv';path.write_text(path.read_text()+'corrupt\n')
   with self.assertRaises(ValueError):scan(root,True)
if __name__=='__main__':unittest.main(verbosity=2)
