"""Seven frozen methods, best within 120k, one shared scenario x 10k slots. CPU only.

This adapter delegates actual rollouts to the unchanged Return-v4 runner.
Each method has its own locked worker directory, ledger and immutable commits.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'

from scripts.experiments.bs5_v4_fresh_common import Budget, canonical_hash, setup, source_identity, file_hash, atomic_json, read_json, experiment_lock
from scripts.evaluation.run_bs5_v4_benchmark import save_commit, read_commit, run as core_run

ASSETS = ROOT / 'checkpoints/bs5_v4_best120k'
CONFIG = ROOT / 'configs/bs5_benchmark7_best120k.json'
DEFAULT_OUTPUT = ROOT / 'results/bs5_bench7_best120k_seed260001_updated'
RUNNING_CHILDREN = []


def model_asset_path(original_path):
    """Resolve original metadata paths into the repository checkpoint directory."""
    prefix = 'benchmark_assets/bs5_v4_best120k/'
    assert original_path.startswith(prefix), original_path
    path = (ASSETS / original_path[len(prefix):]).resolve()
    assert path.is_relative_to(ASSETS.resolve()), original_path
    return path


def check_core_source():
    manifest = ASSETS / 'FROZEN_CORE_SHA256SUMS.json'
    core = read_json(manifest)
    receipt_path = ROOT / 'configs/bs5_best120k_repo_import.json'
    adaptations = {}
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        assert file_hash(manifest) == receipt['original_core_manifest_sha256']
        adaptations = receipt['integration_files']
        assert set(adaptations) <= set(core)
    changed = []
    for name, digest in core.items():
        if name in adaptations:
            assert not name.startswith(('src/', 'configs/')), name
            item = adaptations[name]
            assert item['original_sha256'] == digest, name
            digest = item['integrated_sha256']
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        if not path.is_file() or file_hash(path) != digest:
            changed.append(name)
    if changed:
        raise RuntimeError('Frozen scientific source differs; do not silently benchmark another env: ' + ', '.join(changed[:12]))


def append_event(output, phase, **details):
    record=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**details)
    with (Path(output)/'compute_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')


def compute_environment():
    model=platform.processor()
    cpuinfo=Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        model=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith('model name')),model)
    try:affinity=len(os.sched_getaffinity(0))
    except AttributeError:affinity=os.cpu_count()
    quota=Path('/sys/fs/cgroup/cpu.max')
    return dict(platform=platform.platform(),cpu_model=model,logical_cpu_count=os.cpu_count(),affinity_cpu_count=affinity,
                cpu_quota=quota.read_text().strip() if quota.exists() else None,device='cpu',torch_threads_per_worker=1,
                timing_note='Decision time includes actor/heuristic previews. Wall times may include CPU contention; worker CPU time excludes other processes.')


def timed_worker(output,max_seconds):
    wall=time.perf_counter();cpu=time.process_time();outcome='error'
    try:
        core_run(output,max_seconds=max_seconds)
        outcome=read_json(output/'status.json')['status']
    finally:
        atomic_json(output/'worker_compute.json',dict(worker_wall_s=time.perf_counter()-wall,worker_cpu_s=time.process_time()-cpu,
                    outcome=outcome,scope='core runner verification, model loading, rollout and output; excludes Python/module import startup'))


def state_weights_hash(saved):
    digest = hashlib.sha256()
    for prefix in ['agent', 'mixer']:
        for name, tensor in sorted(saved[prefix].items()):
            digest.update((prefix + name).encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def check_inputs():
    """Integrity only: no environment step, training or benchmark rollout."""
    import torch
    check_core_source()
    manifest = read_json(ASSETS / 'ASSET_SHA256SUMS.json')
    for name, sha in manifest.items():
        assert file_hash(ASSETS / name) == sha, name
    cfg = read_json(CONFIG)
    assert cfg['horizon'] == 10000 and cfg['scenario_seeds'] == [260001]
    assert cfg['training_budget'] == 120000 and cfg['training_seed'] == 551
    assert cfg['baselines'] == ['Random-Relay', '5A-Recovery', '4A1R-Recovery', '3A2R-Recovery', 'Dynamic-Recovery']
    metadata = read_json(ASSETS / 'MODEL_SELECTION.json')
    assert [q['selected_step'] for q in metadata['models']] == [114000, 72000]
    protocols = []
    for q in metadata['models']:
        cp = model_asset_path(q['path'])
        assert file_hash(cp) == q['checkpoint_sha256']
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert state_weights_hash(saved) == q['weights_sha256']
        p = read_json(model_asset_path(q['training_protocol']))
        assert p['total_steps'] == 120000 and p['training_seeds'] == [551]
        assert not set(cfg['scenario_seeds']) & set(p['train_pool'] + p['validation_pool'])
        selection = read_json(ASSETS / ('CR_checkpoint_selection.json' if q['arm'] == 'C-R' else 'CN_checkpoint_selection.json'))
        best = max(selection['curve'], key=lambda x: (x['utility'], -x['step']))
        assert best['step'] == selection['best_step'] == q['selected_step']
        assert best['weights_sha256'] == q['weights_sha256']
        protocols.append(p)
    assert all(protocols[0][k] == protocols[1][k] for k in protocols[0] if k not in ['version', 'arms', 'study'])
    return cfg, metadata, protocols


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'{output} exists; use run/all to continue, never overwrite it.')
    cfg, metadata, protocols = check_inputs()
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    shutil.copytree(ASSETS, output / 'input_assets')
    identity = source_identity()
    atomic_json(output / 'source_identity.json', identity)
    policies = []
    for q in metadata['models']:
        dest = output / 'models' / (q['label'] + '.pt')
        shutil.copy2(model_asset_path(q['path']), dest)
        policies.append(dict(label=q['label'], display_name=q['display_name'], arm=q['arm'], training_seed=551,
                             training_budget=120000, step=q['selected_step'], path=dest.relative_to(output).as_posix(),
                             sha256=file_hash(dest), weights_sha256=q['weights_sha256'], privileged=False))
    policies += [dict(label=label, display_name=cfg['display_names'][label], arm=label, training_seed=None,
                      privileged=label != 'Random-Relay') for label in cfg['baselines']]
    jobs = [dict(scenario=s, policy=q['label']) for s in cfg['scenario_seeds'] for q in policies]
    protocol = dict(version=cfg['version'], diagnostic=False, training_protocol=protocols[0],
                    training_protocol_sha256=canonical_hash(protocols[0]), source_sha256=identity['sha256'],
                    original_scientific_source_sha256=metadata['original_scientific_source_sha256'],
                    plan=cfg, scenarios=cfg['scenario_seeds'], horizon=cfg['horizon'], policies=policies, jobs=jobs,
                    runtime=setup('cpu'), planned_steps=len(jobs)*cfg['horizon'], training_updates=0,
                    previews=cfg['greedy_information'], checkpoint_selection=metadata,
                    execution='Independent worker processes; timing includes possible shared-host contention.')
    atomic_json(output / 'protocol.json', protocol)
    atomic_json(output / 'commit.json', dict(completed=[]))
    atomic_json(output / 'status.json', dict(status='prepared', completed=0, total=7))
    atomic_json(output / 'COMPUTE_ENVIRONMENT.json',compute_environment())
    Budget(output / 'budget_ledger.json', {q['label']:10000 for q in policies})
    for q in policies:
        worker = output / 'workers' / q['label']
        worker.mkdir(parents=True)
        sub = deepcopy(protocol)
        sub.update(policies=[q], jobs=[dict(scenario=260001, policy=q['label'])], planned_steps=10000,
                   parent_protocol_sha256=canonical_hash(protocol))
        atomic_json(worker / 'protocol.json', sub)
        save_commit(worker, dict(completed=[]))
        Budget(worker / 'budget_ledger.json', {q['label']:10000})
        atomic_json(worker / 'status.json', dict(status='prepared', completed=0, total=1))
        if q.get('path'):
            (worker / 'models').mkdir()
            shutil.copy2(output / q['path'], worker / q['path'])
    collect(output)
    print(json.dumps(dict(phase='prepared', output=str(output), methods=[q['display_name'] for q in policies],
                          scenario=260001, horizon=10000, selected_steps={'C-R':114000,'C-N':72000}, training_updates=0)), flush=True)


def verify_protocol(output):
    p = read_json(output / 'protocol.json')
    assert p['source_sha256'] == source_identity()['sha256'], 'Source changed after prepare; preserve this output and use its original source.'
    assert p['runtime'] == setup('cpu'), 'Runtime changed after prepare; preserve the same Python/NumPy/Torch environment.'
    assert p['plan'] == read_json(CONFIG)
    for q in p['policies']:
        if q.get('path'):
            assert file_hash(output / q['path']) == q['sha256']
    return p


def collect(output):
    p = read_json(output / 'protocol.json')
    completed, live, ledgers, traces, initials = [], [], {}, set(), set()
    for q in p['policies']:
        label = q['label']; worker = output / 'workers' / label
        commit = read_commit(worker)
        ledger = read_json(worker / 'budget_ledger.json')[label]
        ledgers[label] = ledger
        if commit['completed']:
            assert len(commit['completed']) == 1
            entry = commit['completed'][0]
            idxpath = worker / entry['index']; idx = read_json(idxpath)
            assert idx['horizon'] == 10000 and idx['scenario'] == 260001
            assert ledger['cap'] == ledger['charged'] == ledger['observed'] == 10000
            if q.get('path'):
                assert idx['checkpoint_sha256'] == q['sha256'] and idx['weights_sha256'] == q['weights_sha256']
            traces.add(idx['trace_sha256']); initials.add(canonical_hash(idx['initial']))
            completed.append(dict(scenario=260001, policy=label, index=idxpath.relative_to(output).as_posix()))
        live.append(dict(method=q['display_name'], observed_slots=ledger['observed'], charged_slots=ledger['charged'],
                         committed=bool(commit['completed']), status=read_json(worker / 'status.json')['status']))
    assert len(traces) <= 1 and len(initials) <= 1, 'Methods did not share the same UE trace / initial state.'
    prior = read_json(output / 'commit.json')['completed']
    assert {x['policy'] for x in prior} <= {x['policy'] for x in completed}
    atomic_json(output / 'commit.json', dict(completed=completed))
    atomic_json(output / 'budget_ledger.json', ledgers)
    if completed:
        atomic_json(output / 'MATCHED_EVALUATION_RECEIPT.json', dict(status='passed', methods_checked=len(completed),
            trace_sha256=next(iter(traces)), initial_state_sha256=next(iter(initials)), training_updates=0))
    atomic_json(output / 'progress.json', dict(completed=len(completed), total=7, methods=live))
    return completed, live


def run_workers(output, workers=3, max_seconds=37800):
    p = verify_protocol(output)
    with experiment_lock(output):
        done, _ = collect(output)
        labels_done = {x['policy'] for x in done}
        pending = deque(q['label'] for q in p['policies'] if q['label'] not in labels_done)
        # Independent methods keep their own RNG/trace. After the quick actors,
        # start the costliest heuristics early to reduce idle time at the end.
        priority=['C-R_seed551','C-N_seed551','Random-Relay','Dynamic-Recovery','3A2R-Recovery','4A1R-Recovery','5A-Recovery']
        pending=deque(sorted(pending,key=priority.index))
        for label in pending:
            w = output / 'workers' / label
            row = read_json(w / 'budget_ledger.json')[label]
            if row['charged'] or row['observed'] or (w / 'games').exists():
                raise RuntimeError(f'{label}: interrupted uncommitted trajectory. Preserve evidence; no automatic reset or hidden replay. See README.md.')
        started = time.monotonic();cpu_started=time.process_time(); active = {}; failures = []; last_print = 0
        append_event(output,'invocation_started',workers=workers,queue=list(pending),completed_before=len(done),max_seconds=max_seconds)
        atomic_json(output / 'status.json', dict(status='running', completed=len(done), total=7))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        while pending or active:
            elapsed = time.monotonic() - started
            while pending and len(active) < workers and max_seconds - elapsed >= 600:
                label = pending.popleft(); w = output / 'workers' / label
                stream = (w / 'run.log').open('a', encoding='utf-8')
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker', '--output', str(w),
                                         '--max-seconds', str(max_seconds-elapsed)], stdout=stream, stderr=subprocess.STDOUT, env=env)
                RUNNING_CHILDREN.append(proc)
                active[label] = (proc, stream)
                append_event(output,'worker_started',policy=label,method=p['plan']['display_names'][label],elapsed_s=time.monotonic()-started)
                print(json.dumps(dict(phase='started', method=p['plan']['display_names'][label])), flush=True)
            for label, (proc, stream) in list(active.items()):
                if proc.poll() is not None:
                    stream.close(); del active[label]
                    append_event(output,'worker_finished',policy=label,returncode=proc.returncode,elapsed_s=time.monotonic()-started)
                    if proc.returncode:
                        failures.append(dict(policy=label, returncode=proc.returncode))
                    else:
                        print(json.dumps(dict(phase='worker_finished', method=p['plan']['display_names'][label])), flush=True)
            completed, live = collect(output)
            if elapsed-last_print >= 30 or not (pending or active):
                print(json.dumps(dict(phase='progress', elapsed_s=round(elapsed,1), completed=len(completed), total=7, methods=live)), flush=True)
                last_print=elapsed
            if not active and (failures or max_seconds-elapsed < 600):
                break
            if pending or active:
                time.sleep(2)
        completed, _ = collect(output)
        status = 'complete' if len(completed)==7 else ('partial_error' if failures else 'paused_at_method_boundary')
        atomic_json(output / 'status.json', dict(status=status, completed=len(completed), total=7, failures=failures,
                    workers=workers, invocation_elapsed_s=time.monotonic()-started))
        append_event(output,'invocation_finished',status=status,completed=len(completed),workers=workers,
                     wall_s=time.monotonic()-started,coordinator_cpu_s=time.process_time()-cpu_started)
        if failures:
            raise RuntimeError(f'Workers failed; inspect workers/<policy>/run.log. Evidence preserved: {failures}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['check', 'prepare', 'run', 'status', 'export', 'all', '_worker'])
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--workers', type=int, choices=range(1,8), default=3)
    ap.add_argument('--max-seconds', type=float, default=37800, help='Soft budget; never kills an in-progress 10k-slot method.')
    ap.add_argument('--deep-planner', action='store_true', help='Independent offline Return/BFS audit, no extra rollout.')
    args = ap.parse_args(); output = args.output.resolve()
    if args.command == 'check':
        cfg, metadata, _ = check_inputs()
        print(json.dumps(dict(status='passed', runtime=setup('cpu'), methods=list(cfg['display_names'].values()),
                              checkpoint_steps={q['arm']:q['selected_step'] for q in metadata['models']}, executed_slots=0, training_updates=0)), flush=True)
    elif args.command == '_worker':
        timed_worker(output,max_seconds=args.max_seconds)
    elif args.command == 'prepare':
        prepare(output)
    elif args.command == 'status':
        print(json.dumps(read_json(output/'status.json'),ensure_ascii=False,indent=2))
        print(json.dumps(read_json(output/'progress.json'),ensure_ascii=False,indent=2))
    elif args.command == 'export':
        from scripts.analysis.report_bs5_benchmark7_best120k import export_report
        print(export_report(output, deep=args.deep_planner))
    elif args.command in ['run','all']:
        if not output.exists():
            if args.command == 'run':
                raise FileNotFoundError('Use prepare or all first.')
            prepare(output)
        run_workers(output, args.workers, args.max_seconds)
        if args.command == 'all':
            from scripts.analysis.report_bs5_benchmark7_best120k import export_report
            print(export_report(output, deep=args.deep_planner))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        for proc in RUNNING_CHILDREN:
            if proc.poll() is None:
                proc.terminate()
        for proc in RUNNING_CHILDREN:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        print('Interrupted. Keep all output, ledgers and raw evidence. In-progress trajectories are not automatically replayed.',file=sys.stderr)
        raise SystemExit(130)
