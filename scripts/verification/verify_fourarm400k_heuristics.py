"""CPU-only diagnostic: real controller/environment, exact mid-seed resume and raw audit."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation import run_fourarm400k_heuristics as runner
from scripts.analysis.report_fourarm400k_heuristics import validate_raw

def run(label,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh diagnostic output')
    output.mkdir(parents=True)
    p=runner.make_protocol(label,1,True);p['runtime']=runner.runtime();seed=p['target_seeds'][0]
    full=runner.start_state(p,seed);split=runner.start_state(p,seed)
    for _ in range(p['horizon']):runner.advance(full,p)
    folder=output/'resume';folder.mkdir()
    for _ in range(37):runner.advance(split,p)
    runner.save_state(folder,p,split);split=runner.load_state(folder,p)
    for _ in range(p['horizon']-37):runner.advance(split,p)
    assert full['ue_trace_sha256']==split['ue_trace_sha256']
    for key in full['arrays']:np.testing.assert_array_equal(full['arrays'][key],split['arrays'][key],err_msg=key)
    for rows_a,rows_b in zip(full['rows'],split['rows']):
        for key in rows_a:
            if key not in {'actor_ms','radio_ms','actor_exceeds_slot'}:assert rows_a[key]==rows_b[key],key
    # Actor timing is a measurement, not deterministic state.
    runner.save_seed(folder,p,split,True)
    row=json.loads((folder/'summary.json').read_text())
    validate_raw(lambda name:(output/name).read_bytes(),'resume',row,p)
    assert row['slots']==96 and row['energy_balance_max_abs_j']<1e-6
    if label=='HEURISTIC_HO':
        assert not np.asarray(full['arrays']['roles'])[np.asarray(full['arrays']['status'])==0].tolist() or np.all(np.asarray(full['arrays']['roles'])[np.asarray(full['arrays']['status'])==0]==2)
    result=dict(status='passed',label=label,diagnostic_seed=seed,slots=96,resume_after=37,actions_ue_trace_energy_identical=True,raw_reward_energy_audit_verified=True,torch_required=False,scope='Diagnostic only; no held-out test seed executed')
    runner.atomic_json(output/'PREFLIGHT_PASSED.json',result);print(json.dumps(result,indent=2),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',choices=tuple(runner.LABELS),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.label,a.output)
