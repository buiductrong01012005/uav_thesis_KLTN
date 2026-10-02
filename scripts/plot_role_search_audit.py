"""Compare observed short-horizon greedy collapse without mixing rollout lengths.

Uses the first `--steps` of existing full traces plus the shorter measured
exhaustive trace. No extrapolation. For 20k, first run `exhaustive` for 20k.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR = {
    "QMIX (3 trained seeds)": "#2563EB",
    "Dynamic exhaustive": "#D12D4C",
    "4A1R": "#E67E22",
    "3A2R": "#8655CA",
    "Random": "#697586",
    "Local 5A": "#169B62",
    "4A1R + recovery navigation": "#D4A500",
    "3A2R + recovery navigation": "#BB72CE",
    "Dynamic + recovery navigation": "#9B1A34",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces",default="results/thesis_continuous_eval")
    ap.add_argument("--dynamic-trace",default=None,
                    help="exhaustive_steps.csv (same scenario, same steps); required to include dynamic")
    ap.add_argument("--recovery-dir",default=None)
    ap.add_argument("--steps",type=int,default=650)
    ap.add_argument("--smooth",type=int,default=35)
    ap.add_argument("--outdir",default="results/thesis_continuous_eval/role_search_audit_figures")
    args=ap.parse_args()
    base=Path(args.traces)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    names={"Random":"random", "Local 5A":"local5a",
           "4A1R":"fixed_4a1r", "3A2R":"fixed_3a2r"}
    data={}
    qmix=[]
    for seed in (91,92,93):
        df=pd.read_csv(base/f"rl_seed{seed}_steps.csv")
        if len(df)<args.steps:
            raise ValueError(f"QMIX seed{seed} fewer than {args.steps} samples")
        qmix.append(df.iloc[:args.steps].copy())
    data["QMIX (3 trained seeds)"] = pd.DataFrame({
        "step":qmix[0].step.to_numpy()[:args.steps],
        **{metric:np.mean([d[metric].to_numpy() for d in qmix],axis=0)
           for metric in ("utility","served_ues","relay_served_ues")},
    })
    for label,name in names.items():
        df=pd.read_csv(base/f"{name}_steps.csv")
        if len(df)<args.steps: raise ValueError(f"{name} has fewer than {args.steps} steps")
        data[label]=df.iloc[:args.steps].copy()
    if args.dynamic_trace:
        dpath=Path(args.dynamic_trace)
        df=pd.read_csv(dpath)
        if len(df)<args.steps: raise ValueError("Dynamic trace too short for selected steps")
        data["Dynamic exhaustive"]=df.iloc[:args.steps].copy()
    if args.recovery_dir:
        rd=Path(args.recovery_dir)
        for label,stem in [("4A1R + recovery navigation","recovery_4a1r"),
                           ("3A2R + recovery navigation","recovery_3a2r"),
                           ("Dynamic + recovery navigation","recovery_dynamic")]:
            p=rd/f"{stem}_steps.csv"
            if not p.exists():continue
            df=pd.read_csv(p)
            if len(df)<args.steps:continue  # do not plot partial as if complete
            data[label]=df.iloc[:args.steps].copy()

    for label,df in data.items():
        if not np.array_equal(df.step.to_numpy(),np.arange(1,args.steps+1)):
            raise ValueError(f"Nonmatching step axis: {label}")

    keys=[("utility","Mean user satisfaction","Utility"),
          ("served_ues","UEs served per step","Served UEs"),
          ("relay_served_ues","UEs via Relay per step","Relay-served UEs")]
    for metric,ylab,title in keys:
        fig,ax=plt.subplots(figsize=(10.3,5.4))
        for label,df in data.items():
            raw=np.asarray(df[metric],dtype=float)
            smooth=pd.Series(raw).rolling(args.smooth,min_periods=1).mean().to_numpy()
            ax.plot(df.step,smooth,label=label,color=COLOR[label],lw=2.15)
        ax.set(xlabel="Evaluation step",ylabel=ylab,
               title=f"{title} — first {args.steps:,} measured steps (trailing {args.smooth}-step average)")
        ax.set_xlim(1,args.steps);ax.set_ylim(bottom=0)
        ax.grid(alpha=.22)
        ax.legend(loc="upper right",fontsize=8,ncol=2)
        fig.tight_layout()
        fig.savefig(out/f"audit_{metric}_{args.steps}.png",dpi=230)
        plt.close(fig)
    table=[]
    for label,df in data.items():
        row={"method":label,"steps":len(df),"mean_utility":df.utility.mean(),
             "last100_utility":df.utility.tail(100).mean(),
             "mean_served_ues":df.served_ues.mean(),
             "last100_served_ues":df.served_ues.tail(100).mean()}
        table.append(row)
    pd.DataFrame(table).to_csv(out/f"audit_summary_{args.steps}.csv",index=False)
    print(pd.DataFrame(table).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("Saved",out)

if __name__=="__main__":main()
