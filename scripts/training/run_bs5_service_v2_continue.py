"""Versioned episode-boundary continuation of a completed v2 pilot.

The original pilot runner and all src files remain byte-for-byte unchanged.
Only the final S4 budget and validation cadence are extended. New diagnostics
observe validation trajectories; none are actor inputs or reward terms.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import signal
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import run_bs5_service_v2_pilot as pilot
import numpy as np
import scipy
import torch
from filelock import FileLock

CONFIG = ROOT / 'configs/bs5_service_v2_continue120k.json'
VERSION = 'bs5_service_v2_continue120k_1'


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def runtime():
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, threads=1)


def extension_lock():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), CONFIG]}


def check_source(protocol):
    if protocol['source_sha256'] != pilot.source_lock():
        raise RuntimeError('Parent simulator/learner/pilot source differs; continuation refused')
    if protocol.get('environment_version') != pilot.VERSION:
        raise RuntimeError('Environment version differs')


def check_payload(saved, arm):
    p = saved['protocol']
    if p['arm'] != arm:
        raise ValueError(f"Expected {arm}, checkpoint is {p['arm']}")
    if not saved['training'] or saved['training'][-1]['step'] != saved['step']:
        raise ValueError('Checkpoint and training history disagree')
    if saved['training'][-1]['updates'] != saved['updates']:
        raise ValueError('Update count and training history disagree')
    if len(saved['replay']) == 0 or saved['task_signature'] is None:
        raise ValueError('Missing replay or task signature')
    if set(saved['rng']) != {'python', 'numpy', 'torch', 'cuda', 'actions'}:
        raise ValueError('Incomplete RNG payload')
    for network in ['agent', 'mixer', 'target_agent', 'target_mixer']:
        if not all(torch.isfinite(v).all() for v in saved['model'][network].values()):
            raise ValueError('Nonfinite model tensor')
    if not saved['model']['optimizer'].get('state'):
        raise ValueError('Missing optimizer state')
    if saved['best_step'] not in saved['validated_steps']:
        raise ValueError('Best checkpoint is not a validated step')
    best = [r['utility'] for r in saved['validation'] if r['step'] == saved['best_step']]
    if len(best) != len(p['validation_seeds']) or not np.isclose(np.mean(best), saved['best'], atol=1e-12, rtol=0):
        raise ValueError('Best score and validation history disagree')


def make_protocol(saved, checkpoint, total_steps, evaluation_every, device, allow_runtime_change=False):
    parent = saved['protocol']
    check_payload(saved, parent['arm'])
    check_source(parent)
    if parent['version'] != 'bs5_service_v2_pilot24k_1':
        raise ValueError('Only a completed original v2 pilot can start this continuation')
    if saved['status'] != 'complete' or saved['step'] != parent['total_steps']:
        raise ValueError('Finish the parent pilot before extending its budget')
    expected = pilot.load_protocol(argparse.Namespace(
        diagnostic=parent['diagnostic'], reward=parent['service']['reward_mode'],
        backhaul=parent['service']['backhaul_total_mhz'], delay=parent['service']['report_delay_slots'],
        arm=parent['arm'], device=parent['device']))
    exempt = {'runtime', 'torch_version'}
    if {k:v for k,v in parent.items() if k not in exempt} != {k:v for k,v in expected.items() if k not in exempt}:
        raise ValueError('Parent protocol does not match the preserved pilot protocol')
    horizon = parent['full_horizon']
    if total_steps <= saved['step'] or (total_steps-saved['step']) % horizon:
        raise ValueError('New budget must add whole full-task episodes')
    if evaluation_every <= 0 or evaluation_every % horizon:
        raise ValueError('Validation cadence must be a positive multiple of the episode horizon')
    changed_runtime = parent['runtime'] != runtime() or parent['torch_version'] != str(torch.__version__) or parent['device'] != device
    if changed_runtime and not allow_runtime_change:
        raise RuntimeError(f"Runtime mismatch: parent={parent['runtime']}, torch={parent['torch_version']}, device={parent['device']}; "
                           f"current={runtime()}, torch={torch.__version__}, device={device}. "
                           'Use the parent runtime, or explicitly select --allow-runtime-change for a recorded migration.')
    p = deepcopy(parent)
    p.update(version=VERSION, total_steps=total_steps, runtime=runtime(),
             torch_version=str(torch.__version__), device=device,
             extension_source_sha256=extension_lock())
    p['stage_ends'][-1] = total_steps
    p['evaluation_steps'] = sorted(set(parent['evaluation_steps'] +
        list(range(saved['step']+evaluation_every, total_steps+1, evaluation_every)) + [total_steps]))
    p['continuation'] = dict(parent_step=saved['step'], parent_episode=saved['episode'],
        parent_updates=saved['updates'], parent_protocol_sha256=pilot.digest(parent),
        parent_checkpoint_sha256=file_hash(checkpoint), evaluation_every=evaluation_every,
        runtime_changed=changed_runtime, parent_runtime=parent['runtime'],
        parent_torch=parent['torch_version'], parent_device=parent['device'],
        changes=['extend final S4 budget', 'add fixed-cadence validation and observer diagnostics'],
        preserved=['physics', 'association', 'observation', 'reward', 'training seed schedule',
                   'model', 'targets', 'optimizer', 'replay', 'all RNG states', 'epsilon schedule', 'best selection'])
    p['scope'] = 'continuation of completed v2 pilot; unchanged task; not an independent training seed or final benchmark'
    return p


def validate(learner, p, step):
    rows = []
    for seed in p['validation_seeds']:
        env = pilot.BS5ServiceV2Env(pilot.FULL_CONDITION,
            service_config=pilot.ServiceV2Config(**p['service']), allow_relay=p['arm']=='C-R', horizon=p['full_horizon'])
        current = env.reset(seed=seed, assignment_seed=seed)
        rng = np.random.default_rng(seed)
        samples = []
        max_violation = 0.0
        for _ in range(p['full_horizon']):
            current = env.step(learner.act(current['obs'], current['action_mask'], 0, rng))
            snap, net = env.last_service_snapshot, env.last_v2_network
            active = np.flatnonzero(snap['status']==0)
            access = [a for a in active if snap['roles'][a]==0]
            relays = [a for a in active if snap['roles'][a]==1]
            covered = np.zeros(env.cfg.num_ues, dtype=bool)
            for a in access:
                covered |= np.linalg.norm(snap['ue_positions']-snap['uav_positions'][a],axis=1)*env.cfg.cell_size_m <= env.cfg.access_radius_cells*env.cfg.cell_size_m+1e-9
            paths = env.last_v2_radio_detail['paths']
            forwarding = {r for a,r in net['routes'].items() if r is not None and net['backhaul_caps_mbps'].get(a,0)>1e-9}
            positive = int(np.sum(net['ue_rate_mbps']>1e-9))
            info = current['info']
            samples.append(dict(utility=info['utility'], qos_fraction=info['qos_fraction'],
                ue_outage_fraction=info['ue_outage_fraction'], throughput_mbps=float(net['ue_rate_mbps'].sum()),
                covered_ues=int(covered.sum()), associated_ues=sum(map(len,net['association'].values())),
                reachable_ues=sum(len(net['association'][a]) for a,path in paths.items() if path),
                positive_ues=positive, qos_ues=info['qos_fraction']*env.cfg.num_ues,
                active_access=len(access), active_relay=len(relays), forwarding_relay=len(forwarding),
                access_offer_mbps=float(net['access_raw_ue_rates'].sum()),
                reachable_offer_mbps=sum(net['access_offer_mbps'][a] for a,path in paths.items() if path),
                schedule_airtime=net['schedule_time_sum'], zero_service_slot_fraction=float(positive==0),
                returning_uavs=int(np.sum(snap['status']==1)), charging_uavs=int(np.sum(snap['status']==2)),
                stranded_slot_fraction=float(bool(env.last_return_control['stranded_ids_post']))))
            max_violation = max(max_violation, net['radio_flow_violation_mbps'])
        row = dict(step=step, seed=seed, **{k:float(np.mean([r[k] for r in samples])) for k in samples[0]},
                   slots=p['full_horizon'], max_flow_violation_mbps=max_violation)
        rows.append(row)
    return rows


def plot_curves(output, saved):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
    for axis,field,label in zip(axes,['utility','qos_fraction'],['Utility','Fraction of UEs at 4 Mbps']):
        training = saved['training']
        axis.plot([r['step'] for r in training],[r[field] for r in training],alpha=.5,label='Training episode mean')
        steps = sorted({r['step'] for r in saved['validation']})
        axis.plot(steps,[np.mean([r[field] for r in saved['validation'] if r['step']==s]) for s in steps],marker='o',label='Fixed validation mean')
        axis.axvline(saved['protocol']['continuation']['parent_step'],color='gray',ls='--',lw=1,label='Continuation boundary')
        axis.set(xlabel='Cumulative training steps',ylabel=label,ylim=(0,1));axis.grid(alpha=.2);axis.legend(fontsize=7)
    p = saved['protocol']
    fig.suptitle(f"{p['arm']} / v2 / {'diagnostic only / ' if p['diagnostic'] else ''}continue to {p['total_steps']:,}")
    for extension in ['png','pdf']:
        fig.savefig(output/f'learning_curve.{extension}',dpi=160)
    plt.close(fig)


def run(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output)+'.lock',timeout=0):
        if args.resume:
            saved = torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
            check_payload(saved,args.arm)
            p = saved['protocol']
            check_source(p)
            if p['version'] != VERSION or p['extension_source_sha256'] != extension_lock():
                raise RuntimeError('Continuation source/config changed')
            if p['runtime'] != runtime() or p['torch_version'] != str(torch.__version__) or p['device'] != args.device:
                raise RuntimeError('Runtime changed during continuation; use the same continuation runtime')
            if p['total_steps'] != args.total_steps or p['continuation']['evaluation_every'] != args.evaluation_every:
                raise RuntimeError('Continuation budget/cadence changed; resume refused')
        else:
            if output.exists():
                raise ValueError('Output exists; select --resume or a new output')
            checkpoint = args.parent_checkpoint.resolve()
            saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
            check_payload(saved,args.arm)
            p = make_protocol(saved,checkpoint,args.total_steps,args.evaluation_every,args.device,args.allow_runtime_change)
            parent = deepcopy(saved['protocol'])
            if saved['best_step'] == saved['step']:
                best = dict(agent=saved['model']['agent'],mixer=saved['model']['mixer'],
                    selected_step=saved['best_step'],validation_utility=saved['best'],
                    protocol_sha256=pilot.digest(parent),obs_dim=148,state_dim=66,environment_version=pilot.VERSION)
            elif args.parent_best:
                best = torch.load(args.parent_best,map_location='cpu',weights_only=True)
                if best['selected_step'] != saved['best_step'] or best['protocol_sha256'] != pilot.digest(parent) or best['validation_utility'] != saved['best']:
                    raise ValueError('Parent best checkpoint lineage differs')
            else:
                raise ValueError('Earlier best requires --parent-best; never replace it with last weights')
            output.mkdir()
            pilot.atomic_json(output/'parent_protocol.json',parent)
            pilot.atomic_json(output/'continuation_receipt.json',p['continuation'])
            saved['protocol'] = p
            saved['status'] = 'prepared_continuation'
            pilot.atomic_json(output/'protocol.json',p)
            pilot.atomic_torch(output/'best.pt',best)
            pilot.atomic_torch(output/'resume.pt',saved)
            pilot.atomic_json(output/'status.json',dict(status=saved['status'],committed_training_steps=saved['step'],
                training_updates=saved['updates'],completed_episodes=saved['episode'],best_validation_utility=saved['best'],
                best_step=saved['best_step'],diagnostic=p['diagnostic'],environment_version=pilot.VERSION))
        original = {name:getattr(pilot,name) for name in ['load_protocol','validate','plot_curves','export_tables']}
        def export_tables(directory,state):
            original['export_tables'](directory,state)
            with zipfile.ZipFile(directory/'v2_pilot_review.zip','a',zipfile.ZIP_DEFLATED) as archive:
                for name in ['parent_protocol.json','continuation_receipt.json']:
                    if (directory/name).is_file():archive.write(directory/name,name)
        try:
            pilot.load_protocol = lambda _:deepcopy(p)
            pilot.validate = validate
            pilot.plot_curves = plot_curves
            pilot.export_tables = export_tables
            if args.prepare_only:
                export_tables(output,saved)
                return saved
            inner = argparse.Namespace(output=output,resume=True,max_seconds=args.max_seconds,
                                       stop_after_episodes=args.stop_after_episodes)
            return pilot._run_unlocked(inner)
        finally:
            for name,value in original.items():setattr(pilot,name,value)


def main():
    defaults = json.loads(CONFIG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['C-R','C-N'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--parent-checkpoint',type=Path)
    group.add_argument('--resume',action='store_true')
    parser.add_argument('--parent-best',type=Path)
    parser.add_argument('--total-steps',type=int,default=defaults['total_steps'])
    parser.add_argument('--evaluation-every',type=int,default=defaults['evaluation_every'])
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--allow-runtime-change',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=0)
    parser.add_argument('--stop-after-episodes',type=int,default=0)
    args = parser.parse_args()
    signal.signal(signal.SIGINT,pilot.stop_handler);signal.signal(signal.SIGTERM,pilot.stop_handler)
    run(args)


if __name__ == '__main__':
    main()
