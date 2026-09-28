from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'scripts'/'run_v4_a0_kaggle_long.py'

def aggregate(root: Path, seeds: list[int]):
    rows=[]
    for s in seeds:
        p=root/f'seed{s}'/f'seed{s}_summary.json'
        if not p.exists():
            continue
        d=json.loads(p.read_text())
        hist=d.get('history',[])
        best_row=max(hist,key=lambda x:x['mean_utility']) if hist else None
        rows.append({
            'seed':s,
            'best_validation':float(d['best_validation']),
            'final20':float(d['final_best_checkpoint_eval']['final20']),
            'best_step':int(best_row['steps']) if best_row else None,
            'final_eval_mean':float(d['final_best_checkpoint_eval']['mean_utility']),
        })
    out={'requested_seeds':seeds,'completed_seeds':[r['seed'] for r in rows],'rows':rows}
    if rows:
        for key in ['best_validation','final20','final_eval_mean']:
            a=np.asarray([r[key] for r in rows],dtype=float)
            out[key]={'mean':float(a.mean()),'std':float(a.std()),'n':int(len(a))}
    (root/'aggregate.json').write_text(json.dumps(out,indent=2))
    return out

def main():
    ap=argparse.ArgumentParser(description='Locked final A0 multi-training-seed runner')
    ap.add_argument('--seeds',type=int,nargs='+',default=[81,82,83])
    ap.add_argument('--total-transitions',type=int,default=100000)
    ap.add_argument('--eval-every',type=int,default=2000)
    ap.add_argument('--save-every',type=int,default=2000)
    ap.add_argument('--output-root',default='results/final_a0_multiseed')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--aggregate-only',action='store_true')
    args=ap.parse_args()
    root=Path(args.output_root)
    if not root.is_absolute(): root=(ROOT/root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if not args.aggregate_only:
        for s in args.seeds:
            outdir=root/f'seed{s}'
            cmd=[sys.executable,str(TRAIN),'--total-transitions',str(args.total_transitions),'--seed',str(s),
                 '--eval-every',str(args.eval_every),'--save-every',str(args.save_every),'--outdir',str(outdir)]
            latest=outdir/f'seed{s}_latest_full.pt'
            if args.resume and latest.exists(): cmd.append('--resume')
            print('RUN', ' '.join(cmd), flush=True)
            subprocess.run(cmd,check=True)
            print(json.dumps(aggregate(root,args.seeds),indent=2),flush=True)
    else:
        print(json.dumps(aggregate(root,args.seeds),indent=2))

if __name__=='__main__': main()
