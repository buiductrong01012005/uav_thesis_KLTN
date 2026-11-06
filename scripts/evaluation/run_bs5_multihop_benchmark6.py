"""Frozen-policy / causal-heuristic benchmark with mid-seed resume and explanations."""
from pathlib import Path
import argparse,csv,hashlib,json,os,pickle,signal,sys,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from filelock import FileLock
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.bs5_service_v2 import ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights
from src.baselines.bs5_delayed_multihop import RandomAR,DelayedDynamicRole
from src.diagnostics.bs5_benchmark_explain import slot_sample,network_sample,temporal_summary
from scripts.analysis.report_bs5_multihop_benchmark6 import generate

VERSION='bs5_multihop_benchmark6_1'
STOP=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def atomic_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def write_csv(path,rs):
    if not rs:return
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
def stop_handler(*_):
    global STOP
    STOP=True


def protocol(args):
    cfg=json.loads((ROOT/'configs/bs5_multihop_benchmark6.json').read_text())
    seeds=cfg['diagnostic_seeds'] if args.diagnostic else cfg['pilot_seeds' if args.phase=='pilot' else 'test_seeds']
    import torch,scipy
    science={str(p.relative_to(ROOT)):sha(p) for folder in ['src','scripts/evaluation','scripts/analysis']
             for p in sorted((ROOT/folder).rglob('*.py'))}
    science['configs/bs5_multihop_benchmark6.json']=sha(ROOT/'configs/bs5_multihop_benchmark6.json')
    science['configs/bs5_multihop_train120k.json']=sha(ROOT/'configs/bs5_multihop_train120k.json')
    registry=json.loads((args.checkpoint_root/'selection_manifest.json').read_text())
    common=dict(version=VERSION,phase=args.phase,diagnostic=args.diagnostic,
        target_seeds=seeds,horizon=cfg['diagnostic_horizon'] if args.diagnostic else cfg['horizon'],
        reward_weights=cfg['reward_weights'],source_sha256=science,checkpoint_registry=registry,
        runtime=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__),device='cpu',threads=1),
        network_sample_every=16 if args.diagnostic else cfg['network_sample_every_slots'],
        counterfactual_bandwidth_probes=not args.no_bandwidth_probes,
        info_contract=cfg['information_contract'],explanation_contract=cfg['explanation_contract'],
        trace_hash_contract='SHA256 chain: H0=32 zero bytes; Ht=SHA256(Ht-1 || service-time UE positions float64 little endian C order)',
        scope=cfg['scope'])
    p=dict(**common,label=args.label,arm=cfg['labels'][args.label],experiment_fingerprint=digest(common))
    return p,cfg


def learner_for(p,checkpoint_root):
    if p['label'] in ['RANDOM','DYNAMIC']:return None
    import torch
    from src.rl.bs5_multihop_qmix import make_learner
    record=p['checkpoint_registry'][p['label']]
    path=checkpoint_root/record['file']
    if sha(path)!=record['sha256']:raise ValueError('Frozen checkpoint hash mismatch')
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if cp['selected_step']!=record['selected_step'] or cp['protocol_sha256']!=record['protocol_sha256']:
        raise ValueError('Frozen checkpoint selection mismatch')
    if cp['obs_dim']!=148 or cp['state_dim']!=66 or cp['selection_metric']!='tcf_score':
        raise ValueError('Wrong checkpoint input or selection criterion')
    if p['label']!='CN' and (cp.get('arm')!=p['arm'] or cp.get('preset')!='balanced' or cp.get('training_env_version')!='bs5_multihop_training_1'):
        raise ValueError('Wrong checkpoint arm/environment/reward preset')
    for rel,value in record.get('preserved_training_source',{}).items():
        if sha(ROOT/rel)!=value:raise ValueError('Original training science changed: '+rel)
    learner=make_learner(p['arm'],device='cpu')
    learner.agent.load_state_dict(cp['agent']);learner.mixer.load_state_dict(cp['mixer'])
    learner.agent.eval();learner.mixer.eval()
    return learner


def start_state(p,seed):
    config=json.loads((ROOT/'configs/bs5_multihop_train120k.json').read_text())
    env=BS5MultihopTrainEnv(arm=p['arm'],horizon=p['horizon'],service_config=ServiceV2Config(**config['service']),reward_weights=TCFWeights(**p['reward_weights']))
    current=env.reset(seed=seed,assignment_seed=seed)
    controller=RandomAR(np.random.SeedSequence([seed,0x52414E44])) if p['label']=='RANDOM' else DelayedDynamicRole() if p['label']=='DYNAMIC' else None
    return dict(seed=seed,env=env,current=current,controller=controller,rng=np.random.default_rng(seed),
        rows=[],arrays={},network=[],cycles=dict(rejoins=0,completed=[],pending={}),
        initial_battery=float(env.battery.sum()),ue_trace_hash='00'*32,live_events=[],wall_seconds=0.)


def update_cycles(state,before,served):
    env=state['env'];c=state['cycles'];pending=c['pending']
    for aid in range(5):
        if before['status'][aid]==2 and env.status[aid]==0:
            c['rejoins']+=1
            if aid in pending:pending[aid]['rejoin_slot']=int(env.t)
        if aid in pending and pending[aid]['rejoin_slot'] is not None and served[aid]:
            event=pending.pop(aid);event.update(served_again_slot=int(env.t),outcome='served_again');c['completed'].append(event)
        if before['status'][aid]==0 and env.status[aid]==1:
            if aid in pending:
                event=pending.pop(aid);event['outcome']='next_return_without_service';c['completed'].append(event)
            pending[aid]=dict(agent=aid,return_slot=int(env.t),rejoin_slot=None)


def save_state(folder,p,state):
    temp=folder/'resume.pkl.tmp'
    with temp.open('wb') as f:
        pickle.dump(dict(protocol_fingerprint=digest(p),state=state),f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
    temp.replace(folder/'resume.pkl')
    atomic_json(folder/'progress.json',dict(slot=len(state['rows']),seed=state['seed'],horizon=p['horizon'],resume_sha256=sha(folder/'resume.pkl')))


def save_seed(folder,p,state,complete):
    if not state['rows']:return
    arrays={k:np.asarray(v) for k,v in state['arrays'].items()}
    summary,ue=temporal_summary(arrays,state['rows'],state['cycles'],state['initial_battery'])
    summary.update(label=p['label'],arm=p['arm'],seed=state['seed'],phase=p['phase'],diagnostic=p['diagnostic'],
        complete=complete,target_slots=p['horizon'],experiment_fingerprint=p['experiment_fingerprint'],
        ue_trace_sha256=state['ue_trace_hash'],simulation_wall_seconds=state['wall_seconds'])
    write_csv(folder/'slots.csv',state['rows']);write_csv(folder/'ue_summary.csv',ue)
    np.savez_compressed(folder/'trace.npz',**arrays)
    atomic_json(folder/'summary.json',summary)
    atomic_json(folder/'cycles.json',dict(rejoins=state['cycles']['rejoins'],completed=state['cycles']['completed'],pending=list(state['cycles']['pending'].values())))
    for name,key in [('network_samples.jsonl','network'),('live_events.jsonl','live_events')]:
        (folder/name).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in state[key]))


def export(root,label):
    root=Path(root);generate(root)
    outputs=[]
    for kind,include_resume in [('REVIEW',False),('RESUME',True)]:
        path=root.parent/f'{label}_BENCH_{kind}.zip';temp=path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob('*')):
                if f.is_file() and f.suffix not in ['.lock','.tmp'] and (include_resume or f.name!='resume.pkl'):
                    z.write(f,f.relative_to(root))
        with zipfile.ZipFile(temp) as z:
            if z.testzip():raise ValueError('Corrupt export')
        temp.replace(path);outputs.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    atomic_json(root.parent/f'{label}_BENCH_INDEX.json',dict(version=VERSION,outputs=outputs))
    return outputs


def run(args):
    global STOP
    STOP=False
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p,cfg=protocol(args)
    chosen=args.seeds or p['target_seeds']
    if not set(chosen)<=set(p['target_seeds']):raise ValueError('Use only locked seeds for this phase')
    root=args.output;folder=root/p['label'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'protocol.json'
    if path.is_file():
        if json.loads(path.read_text())!=p:raise ValueError('Resume refused: source/runtime/checkpoint/phase/science changed')
    else:atomic_json(path,p)
    model=learner_for(p,args.checkpoint_root)
    started=time.perf_counter();processed=0
    with FileLock(str(folder)+'.lock',timeout=0):
        try:
            for seed in chosen:
                dest=folder/f'seed{seed}';dest.mkdir(exist_ok=True)
                summary=dest/'summary.json'
                if summary.is_file() and json.loads(summary.read_text()).get('complete'):
                    print('Reuse completed',p['label'],seed,flush=True);continue
                if (dest/'resume.pkl').is_file():
                    progress=json.loads((dest/'progress.json').read_text())
                    if sha(dest/'resume.pkl')!=progress['resume_sha256']:raise ValueError('Resume checksum failed')
                    with (dest/'resume.pkl').open('rb') as f:saved=pickle.load(f)
                    if saved['protocol_fingerprint']!=digest(p):raise ValueError('Resume protocol changed')
                    state=saved['state']
                    print('Resume',p['label'],seed,'slot',len(state['rows']),flush=True)
                else:state=start_state(p,seed)
                seed_started=time.perf_counter();previous_wall=state['wall_seconds']
                while len(state['rows'])<p['horizon']:
                    env=state['env'];current=state['current']
                    before=dict(status=env.status.copy(),battery=env.battery.copy(),positions=env.uav_positions.copy())
                    tic=time.perf_counter()
                    if model is None:
                        actions=state['controller'].act(current['obs'],current['action_mask'])
                        stats=state['controller'].last_stats.copy()
                    else:
                        actions=model.act(current['obs'],current['action_mask'],0.,state['rng']);stats={}
                    stats['decision_ms']=(time.perf_counter()-tic)*1000
                    if not all(current['action_mask'][a,int(x)] for a,x in enumerate(actions)):raise AssertionError('Illegal policy action')
                    state['current']=env.step(actions)
                    row,arrays,served=slot_sample(env,before,state['current']['info'],stats)
                    arrays['actions']=np.asarray(actions,dtype=np.int8)
                    state['rows'].append(row)
                    for k,v in arrays.items():state['arrays'].setdefault(k,[]).append(v)
                    data=np.asarray(env.last_service_snapshot['ue_positions'],dtype='<f8').tobytes(order='C')
                    state['ue_trace_hash']=hashlib.sha256(bytes.fromhex(state['ue_trace_hash'])+data).hexdigest()
                    update_cycles(state,before,served)
                    slot=len(state['rows']);processed+=1
                    if slot%p['network_sample_every']==0:
                        state['network'].append(network_sample(env,p['counterfactual_bandwidth_probes']))
                    if slot%args.log_every==0 or slot==p['horizon']:
                        rs=state['rows'];rates=np.asarray(state['arrays']['rates'])
                        event=dict(label=p['label'],seed=seed,slot=slot,target=p['horizon'],
                            mean_mbps=float(np.mean([r['throughput_mbps'] for r in rs])),
                            covered=float(np.mean([r['covered_ues'] for r in rs])),
                            reachable=float(np.mean([r['reachable_ues'] for r in rs])),
                            ue_ge1=float(np.mean([r['service_coverage_ues'] for r in rs])),
                            unique_ge1=int((rates>=1-1e-9).any(axis=0).sum()),
                            return_charge_pct=100*float(np.mean([r['unavailable_uavs']/5 for r in rs])),
                            actor_ms=float(np.mean([r['actor_ms'] for r in rs[-args.log_every:]])))
                        state['live_events'].append(event)
                        print(f"{p['label']} seed={seed} {slot}/{p['horizon']} | Mbps={event['mean_mbps']:.2f} | covered/BH/ge1={event['covered']:.1f}/{event['reachable']:.1f}/{event['ue_ge1']:.1f} | unique_ge1={event['unique_ge1']}/90 | Return+Charge={event['return_charge_pct']:.1f}% | actor={event['actor_ms']:.1f}ms",flush=True)
                    state['wall_seconds']=previous_wall+time.perf_counter()-seed_started
                    stop=(STOP or (args.max_seconds>0 and time.perf_counter()-started>=args.max_seconds)
                          or (args.stop_after_slots>0 and processed>=args.stop_after_slots))
                    if slot%args.save_every==0 or stop or slot==p['horizon']:
                        save_state(dest,p,state)
                    if stop:break
                complete=len(state['rows'])==p['horizon']
                save_seed(dest,p,state,complete)
                if not complete:
                    print('Paused safely at committed slot',len(state['rows']),flush=True);break
        finally:
            export(root,p['label']+'_'+p['phase'])
    return p


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',choices=['CN','CR','CH','HO','RANDOM','DYNAMIC'],required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','test'],default='pilot')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--no-bandwidth-probes',action='store_true')
    p.add_argument('--max-seconds',type=float,default=10*3600)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--stop-after-slots',type=int,default=0)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.save_every<1 or args.log_every<1:raise ValueError('Log/save interval must be positive')
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    run(args)
