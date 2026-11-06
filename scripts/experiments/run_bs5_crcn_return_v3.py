"""One Kaggle pipeline: frozen C-R + fresh C-N60k + paired long evaluations.
No C-R training, no baseline search policies, no physical/controller changes.
"""
from pathlib import Path
from copy import deepcopy
import argparse,gzip,json,os,shutil,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.experiments.bs5_crcn_common import ASSETS,TraceBank,trace_spec,verify_assets,fixture_signal_note
from scripts.training import run_bs5_cn_h1200_return as cn
from scripts.analysis.bs5_bundle import atomic_json,canonical_hash,file_hash,read_json
from scripts.training.run_bs5_launch_ab import source_identity,freeze_source,experiment_lock,weights_hash
from src.envs.bs5_curriculum_env import condition_for
SPEC=ROOT/'configs/bs5_crcn_return_v3_pair_v1.json'

def prepare(root,diagnostic=False):
    root=Path(root).resolve();assert not root.exists(),'Use existing output with run/restore; do not overwrite'
    cn.setup_runtime('cpu',1);assets=verify_assets();root.mkdir(parents=True)
    p=read_json(SPEC);identity=source_identity();p.update(diagnostic=diagnostic,source_sha256=identity['sha256'],assets=assets,runtime=cn.runtime('cpu'))
    if diagnostic:p.update(evaluation_horizon=24,seen_scenarios=[],heldout_scenarios=[180101,180102],paired_evaluation_steps=96,cn_training_steps=192,cn_validation_steps=72)
    p['job_order']=[dict(arm=arm,scenario=scenario,group=group) for group,pool in [('seen',p['seen_scenarios']),('heldout',p['heldout_scenarios'])] for scenario in pool for arm in ['C-R','C-N']]
    atomic_json(root/'protocol.json',p);freeze_source(root,identity)
    atomic_json(root/'baseline_note.json',fixture_signal_note())
    cn.prepare(root/'cn','cpu',diagnostic=diagnostic)
    frozen=read_json(ASSETS/'protocol.json');new=read_json(root/'cn/protocol.json')
    if not diagnostic:
        assert new['game_schedule']==frozen['game_schedule'] and new['evaluation_steps']==frozen['evaluation_steps']
        for k in ['lr','gamma','batch_size','warmup','update_every','target_every','epsilon_end','epsilon_decay_steps','replay_capacity','stage_ends','early_horizon','full_horizon']:
            assert new[k]==frozen[k],('matched recipe changed',k)
    atomic_json(root/'status.json',dict(status='prepared',phase='cn_training'))
    return p

def assert_frozen(root):
    p=read_json(root/'protocol.json');assert source_identity()['sha256']==p['source_sha256'],'Source changed'
    assert verify_assets()==p['assets'],'Frozen input assets changed'
    assert cn.runtime('cpu')==p['runtime'],'Runtime changed; keep exact resume runtime'
    return p

def load_checkpoint(env,path):
    learner=cn.BS5CurriculumQMIX(env,device='cpu')
    d=torch.load(path,map_location='cpu',weights_only=True)
    learner.agent.load_state_dict(d['agent'],strict=True);learner.mixer.load_state_dict(d['mixer'],strict=True)
    learner.agent.eval();learner.mixer.eval()
    for model in [learner.agent,learner.mixer]:
        for p in model.parameters():p.requires_grad_(False)
    return learner,int(d['step'])

def raw_game(root,relative):
    index=read_json(root/relative);rows=[]
    for chunk in index['chunks']:
        path=root/chunk['path'];assert file_hash(path)==chunk['sha256']
        with gzip.open(path,'rt') as f:rows.extend(json.load(f)['rows'])
    return index,rows

def compare_prefix(rows,arm,scenario,root,p):
    if arm=='C-R':
        with np.load(ASSETS/f'prefix_{scenario}.npz') as z:expected={k:z[k] for k in z.files}
    else:
        selected=read_json(root/'cn/checkpoint_selection.json')
        _,ref=raw_game(root/'cn',f"evaluations/{selected['best_step']:06d}/{scenario}/index.json")
        expected=prefix_arrays(ref)
    actual=prefix_arrays(rows[:1200]);errors={}
    for k in ['raw_actions','obs','state','action_mask','positions','status','roles','post_battery','post_status','ue_positions']:
        assert np.array_equal(actual[k],expected[k]),('prefix mismatch',arm,scenario,k)
    errors['rate_max_abs']=float(np.max(np.abs(actual['rates']-expected['rates'])))
    assert np.allclose(actual['rates'],expected['rates'],atol=p['prefix_rate_atol'],rtol=p['prefix_rate_rtol']),('prefix rates',arm,scenario,errors)
    atomic_json(root/'paired/prefix_checks'/f'{arm}_{scenario}.json',dict(status='passed',slots=1200,arm=arm,scenario=scenario,errors=errors))

def prefix_arrays(rows):
    return dict(raw_actions=np.array([r['raw_actions'] for r in rows]),obs=np.array([r['obs'] for r in rows],np.float32),
        state=np.array([r['state'] for r in rows],np.float32),action_mask=np.array([r['action_mask'] for r in rows],bool),
        ue_positions=np.array([r['service']['ue_positions'] for r in rows]),rates=np.array([r['service']['network']['ue_rate_mbps'] for r in rows]),
        positions=np.array([r['service']['uav_positions'] for r in rows]),status=np.array([r['service']['status'] for r in rows]),
        roles=np.array([r['service']['roles'] for r in rows]),post_battery=np.array([r['post']['battery'] for r in rows]),post_status=np.array([r['post']['status'] for r in rows]))

def prepare_pair(root,p):
    directory=root/'paired'
    if (directory/'protocol.json').exists():return read_json(directory/'protocol.json')
    assert read_json(root/'cn/status.json')['status']=='complete'
    selected=read_json(root/'cn/checkpoint_selection.json');directory.mkdir(exist_ok=True);(directory/'checkpoints').mkdir()
    shutil.copy2(ASSETS/'060000.pt',directory/'checkpoints/C-R.pt')
    shutil.copy2(root/'cn/checkpoints'/f"{selected['best_step']:06d}.pt",directory/'checkpoints/C-N.pt')
    weights={arm:dict(path=f'checkpoints/{arm}.pt',sha256=file_hash(directory/'checkpoints'/f'{arm}.pt'),step=60000 if arm=='C-R' else selected['best_step']) for arm in ['C-R','C-N']}
    protocol=dict(version=p['version'],weights=weights,jobs=p['job_order'],horizon=p['evaluation_horizon'],
        selection='C-R fixed historical60k; C-N max validation1200 at the same milestones; no selection on10k',diagnostic=p['diagnostic'])
    atomic_json(directory/'protocol.json',protocol);atomic_json(directory/'commit.json',dict(completed=[]))
    for arm in ['C-R','C-N']:
        d=directory/arm;d.mkdir();atomic_json(d/'budget_ledger.json',dict(phases={'eval':dict(cap=len(p['job_order'])//2*p['evaluation_horizon'],charged=0,observed=0)}))
    return protocol

def paired_run(root,p,deadline=None,stop_after_games=None):
    pair=prepare_pair(root,p);directory=root/'paired';commit=read_json(directory/'commit.json')
    for arm,cp in pair['weights'].items():assert file_hash(directory/cp['path'])==cp['sha256']
    for arm in ['C-R','C-N']:
        b=cn.Budget(directory/arm);expected=sum(p['evaluation_horizon'] for item in commit['completed'] if item['arm']==arm)
        assert b.data['phases']['eval']['charged']==b.data['phases']['eval']['observed']==expected,'Uncommitted eval compute; preserve evidence, no hidden repeat'
    assert commit['completed']==[{**job,'index':f"{job['arm']}/evaluations/{pair['weights'][job['arm']]['step']:06d}/{job['scenario']}/index.json"} for job in pair['jobs'][:len(commit['completed'])]]
    cn.TRACE_HOME=root/'traces';base=read_json(root/'cn/protocol.json');initial_count=len(commit['completed'])
    learners={}
    for job in pair['jobs'][len(commit['completed']):]:
        if (deadline is not None and time.monotonic()>=deadline) or (stop_after_games is not None and len(commit['completed'])-initial_count>=stop_after_games):break
        arm,scenario=job['arm'],job['scenario'];q=deepcopy(base);q.update(arm=arm,arms=[arm],allow_relay=arm=='C-R',full_horizon=p['evaluation_horizon'])
        c=condition_for(q['stage_ends'][2],arm,q)
        if arm not in learners:learners[arm]=load_checkpoint(cn.new_env(q,c,p['evaluation_horizon'],30),directory/pair['weights'][arm]['path'])
        learner,step=learners[arm];before=weights_hash(learner);budget=cn.Budget(directory/arm)
        state=dict(steps=step,timers={k:0. for k in ['env_s','actor_s','record_s','raw_io_s']})
        spec=dict(condition=c,horizon=p['evaluation_horizon'],scenario_seed=scenario,assignment_seed=scenario,residual_battery=30.,trace_group=job['group'])
        callback=(lambda rows:compare_prefix(rows,arm,scenario,root,p)) if job['group']=='seen' and p['evaluation_horizon']>=1200 else None
        atomic_json(root/'status.json',dict(status='running',phase='paired_evaluation',job=job,completed=len(commit['completed'])))
        try:
            result,index=cn.rollout(directory/arm,learner,None,np.random.default_rng(0),q,budget,state,spec=spec,evaluation=True,prefix_validator=callback)
            assert weights_hash(learner)==before
            indexpath=directory/arm/index;data=read_json(indexpath);data.update(trace_group=job['group'],policy_checkpoint_sha256=pair['weights'][arm]['sha256']);atomic_json(indexpath,data)
        except BaseException as exc:
            budget.flush();atomic_json(root/'status.json',dict(status='partial_error',phase='paired_evaluation',job=job,error=repr(exc),instruction='Keep partial output; never reset the ledger to rerun an uncertain game'))
            raise
        commit['completed'].append(dict(**job,index=arm+'/'+index));atomic_json(directory/'commit.json',commit)
        print(json.dumps(dict(phase='paired',completed=len(commit['completed']),total=len(pair['jobs']),arm=arm,scenario=scenario,utility=result['metrics']['utility'])),flush=True)
    done=len(commit['completed'])==len(pair['jobs']);atomic_json(root/'status.json',dict(status='complete' if done else 'paused_at_game_boundary',phase='paired_evaluation',completed=len(commit['completed']),total=len(pair['jobs'])))
    return done

def run(root,max_seconds=None,stop_after_games=None):
    root=Path(root).resolve();cn.setup_runtime('cpu',1);p=assert_frozen(root)
    with experiment_lock(root):
        if (root/'SHA256SUMS.json').exists():
            for rel,digest in read_json(root/'SHA256SUMS.json').items():assert file_hash(root/rel)==digest,('changed exported evidence',rel)
            (root/'SHA256SUMS.json').unlink()
        deadline=time.monotonic()+max_seconds if max_seconds is not None else None
        try:
            if read_json(root/'cn/status.json')['status']!='complete':
                atomic_json(root/'status.json',dict(status='running',phase='cn_training'))
                cn.run(root/'cn',resume=(root/'cn/resume.pt').exists(),max_seconds=max(0,deadline-time.monotonic()) if deadline is not None else None)
                if read_json(root/'cn/status.json')['status']!='complete':
                    atomic_json(root/'status.json',dict(status='paused_at_game_boundary',phase='cn_training'));return
            if deadline is not None and time.monotonic()>=deadline:
                atomic_json(root/'status.json',dict(status='paused_at_game_boundary',phase='before_paired_evaluation'));return
            paired_run(root,p,deadline,stop_after_games)
        except BaseException as exc:
            previous=read_json(root/'status.json')
            atomic_json(root/'status.json',dict(**{k:v for k,v in previous.items() if k not in ['status','error','instruction']},
                status='partial_error',error=repr(exc),instruction='Preserve all output and ledgers; do not silently repeat uncertain compute'))
            raise

def export(root):
    from scripts.analysis.analyze_bs5_crcn_return_v3 import analyze
    root=Path(root).resolve();out=root.parent/(root.name+'_export');i=1
    while out.exists():i+=1;out=root.parent/(root.name+f'_export_{i}')
    out.mkdir();analysis=out/'analysis';analyze(root,analysis)
    # Copy standalone stdlib analysis helpers into the evidence root.
    for name in ['analyze_bs5_crcn_return_v3.py','bs5_bundle.py','bs5_continuous_common.py','analyze_bs5_curriculum_bundle.py','analyze_bs5_curriculum_extend.py','bs5_curve_plot.py','bs5_return_audit.py','analyze_bs5_cn_h1200_return.py']:
        shutil.copy2(ROOT/'scripts/analysis'/name,root/name)
    locked=root/'locked_cr';locked.mkdir(exist_ok=True)
    for name in ['060000.pt','protocol.json','checkpoint_selection.json','source_identity.json','ASSET_SHA256SUMS.json']:
        shutil.copy2(ASSETS/name,locked/name)
    excluded={'run.lock','.experiment.lock','SHA256SUMS.json'}
    members=[f for f in sorted(root.rglob('*')) if f.is_file() and f.name not in excluded and not f.name.endswith('.tmp') and not f.relative_to(root).as_posix().startswith('preflight/test_data/')]
    manifest={f.relative_to(root).as_posix():file_hash(f) for f in members};atomic_json(root/'SHA256SUMS.json',manifest);members.append(root/'SHA256SUMS.json')
    receipts=[]
    for category in ['audit_light','repro_full']:
        selected=[f for f in members if (f.suffix=='.pt')==(category=='repro_full')]
        number=0;archive=None;size=0
        for f in selected:
            if archive is None or size+f.stat().st_size>75*1024**2:
                if archive:archive.close();temp.replace(final)
                number+=1;size=0;final=out/f'{category}_part{number:03d}.zip';temp=final.with_suffix('.zip.tmp');archive=zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3)
            archive.write(f,f.relative_to(root).as_posix());size+=f.stat().st_size
        if archive:archive.close();temp.replace(final)
    for f in sorted(out.glob('*.zip')):
        with zipfile.ZipFile(f) as z:assert z.testzip() is None
        receipts.append(dict(path=f.name,kind='independent_zip',sha256=file_hash(f),bytes=f.stat().st_size))
    atomic_json(out/'EXPORT_INDEX.json',dict(version='bs5_crcn_transport_v1',status=read_json(root/'status.json'),files=receipts,instruction='Extract ALL companion ZIPs into one root; run analyze_bs5_crcn_return_v3.py --root MERGED --output ANALYSIS. Restore all files for exact continuation.'))
    with zipfile.ZipFile(out/'review_small.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(analysis.rglob('*')):
            if f.is_file():z.write(f,'analysis/'+f.relative_to(analysis).as_posix())
        for name in ['protocol.json','status.json','baseline_note.json']:
            if (root/name).exists():z.write(root/name,name)
        z.write(out/'EXPORT_INDEX.json','EXPORT_INDEX.json')
    print('EXPORT',out,flush=True);return out

def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run','export']);a.add_argument('--output',type=Path,required=True);a.add_argument('--diagnostic',action='store_true');a.add_argument('--max-seconds',type=float);a.add_argument('--stop-after-games',type=int);args=a.parse_args()
    if args.command=='prepare':prepare(args.output,args.diagnostic)
    elif args.command=='run':run(args.output,args.max_seconds,args.stop_after_games)
    else:export(args.output)
if __name__=='__main__':main()
