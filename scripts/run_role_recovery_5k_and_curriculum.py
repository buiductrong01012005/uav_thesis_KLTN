"""Five-thousand-step comparison of recovery baselines and Cartesian QMIX curriculum.

Run from project root:
  python scripts/run_role_recovery_5k_and_curriculum.py --dry-run
  python scripts/run_role_recovery_5k_and_curriculum.py --steps 5000

One environment reset per policy; inference only. Existing 20k direct-QMIX
CSV traces are reused later by the plotting script, not rerun here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def complete(path: Path, csv: Path, *, steps: int, scenario_seed: int, checkpoint: Path | None):
    if not path.is_file() or not csv.is_file():
        return False
    d = json.loads(path.read_text(encoding='utf-8'))
    if d.get('steps') != steps or d.get('scenario_seed') != scenario_seed:
        return False
    if checkpoint is not None and Path(d.get('checkpoint','')).resolve() != checkpoint.resolve():
        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--scenario-seed', type=int, default=3101)
    p.add_argument('--rl-seeds', type=int, nargs='+', default=[91,92,93])
    p.add_argument('--checkpoint-root', default='results/thesis_full_curriculum/cartesian')
    p.add_argument('--output-root', default='results/thesis_continuous_eval')
    p.add_argument('--only', choices=('all','recovery','curriculum'), default='all')
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--zero-service-patience', type=int, default=3)
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.steps <= 0 or args.log_every <= 0 or args.zero_service_patience < 1:
        p.error('steps, log-every, and patience must be positive')
    outroot=Path(args.output_root)
    if not outroot.is_absolute(): outroot=ROOT/outroot
    ckroot=Path(args.checkpoint_root)
    if not ckroot.is_absolute(): ckroot=ROOT/ckroot
    recovery_out = outroot/f'recovery_{args.steps}'
    curriculum_out = outroot/f'curriculum_{args.steps}'

    if args.only in ('all','recovery'):
        baseline_script=ROOT/'scripts'/'run_role_recovery_diagnostic.py'
        if not baseline_script.exists() and not args.dry_run:
            p.error(f'Missing prerequisite source: {baseline_script}; unpack role_navigation_audit_patch.zip first')
        already=[]
        missing=[]
        for policy in ('4a1r','3a2r','dynamic'):
            file=recovery_out/f'recovery_{policy}_summary.json'
            csv=recovery_out/f'recovery_{policy}_steps.csv'
            if complete(file,csv,steps=args.steps,scenario_seed=args.scenario_seed,checkpoint=None) and not args.force:
                already.append(policy)
            else:
                missing.append(policy.upper() if policy!='dynamic' else 'dynamic')
        print('RECOVERY skip:',already,'need:',missing, flush=True)
        if missing:
            cmd=[sys.executable,str(baseline_script),'--steps',str(args.steps),
                 '--scenario-seed',str(args.scenario_seed),'--policies',*missing,
                 '--zero-service-patience',str(args.zero_service_patience),
                 '--log-every',str(args.log_every),'--outdir',str(recovery_out)]
            if args.force: cmd.append('--force')
            print('RUN:',' '.join(cmd),flush=True)
            if not args.dry_run: subprocess.run(cmd,check=True,cwd=ROOT)

    if args.only in ('all','curriculum'):
        expected={seed:ckroot/f'seed{seed}_best_stage3.pt' for seed in args.rl_seeds}
        if not args.dry_run:
            missing=[str(path) for path in expected.values() if not path.is_file()]
            if missing: p.error('Missing trained Cartesian curriculum checkpoints:\n'+'\n'.join(missing))
            if not (ROOT/'scripts'/'run_uav_continuous_eval.py').exists():
                p.error('Missing prerequisite scripts/run_uav_continuous_eval.py')
        for seed, checkpoint in expected.items():
            label=f'curriculum_rl_seed{seed}'
            sm=curriculum_out/f'{label}_summary.json'
            csv=curriculum_out/f'{label}_steps.csv'
            if not args.force and complete(sm,csv,steps=args.steps,scenario_seed=args.scenario_seed,
                                           checkpoint=checkpoint):
                print('SKIP existing:',label,flush=True)
                continue
            if not args.force and (sm.exists() or csv.exists()):
                p.error(f'Incomplete/conflicting files for {label}; specify --force or a new --output-root')
            print(f'CURRICULUM inference: {label} | checkpoint={checkpoint} | output={curriculum_out}',flush=True)
            if args.dry_run: continue
            import torch
            torch.set_num_threads(max(1,args.torch_threads))
            from scripts.run_uav_continuous_eval import run_one
            ns=SimpleNamespace(
                steps=args.steps,scenario_seed=args.scenario_seed,
                outdir=str(curriculum_out),device=args.device,
                log_every=args.log_every,
                # Avoid overwriting the existing direct-QMIX case study.
                case_training_seed=-9999,
            )
            run_one(ns,label,'rl',checkpoint)
    print('Finished. Existing direct QMIX 20k traces have not been modified.',flush=True)


if __name__=='__main__':
    main()
