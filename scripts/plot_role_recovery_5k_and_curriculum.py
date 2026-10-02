"""Compare 5k evaluation of direct QMIX, curriculum QMIX and recovery baselines.

Every method is placed on the SAME chart, in a distinct color.
No simulation/training is done by this plotting script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ORDER=('QMIX (direct)','QMIX + Curriculum','Dynamic + Recovery',
       '4A1R + Recovery','3A2R + Recovery','Random','Local 5A')
COLORS={
    'QMIX (direct)':'#2563EB',
    'QMIX + Curriculum':'#F97316',
    'Dynamic + Recovery':'#DC2626',
    '4A1R + Recovery':'#7C3AED',
    '3A2R + Recovery':'#C026D3',
    'Random':'#64748B',
    'Local 5A':'#15803D',
}
METRICS={
    'utility':('Mean user satisfaction','Utility across evaluation steps'),
    'served_ues':('UEs served / step','UEs receiving throughput'),
    'relay_served_ues':('UEs served via Relay / step','Actually useful Relay traffic'),
    'relay':('Active Relay UAVs / step','Relay role selected by policy'),
}


def trailing(x,k):
    return pd.Series(x).rolling(k,min_periods=1).mean().to_numpy()


def check_csv(csv:Path,summary:Path,steps:int,scenario_seed:int):
    if not csv.is_file(): raise FileNotFoundError(f'Missing CSV: {csv}')
    if not summary.is_file(): raise FileNotFoundError(f'Missing summary: {summary}')
    df=pd.read_csv(csv)
    s=json.loads(summary.read_text(encoding='utf-8'))
    if len(df)<steps or not np.array_equal(df['step'].to_numpy()[:steps],np.arange(1,steps+1)):
        raise ValueError(f'Nonconsecutive/too short step sequence: {csv}')
    if s.get('scenario_seed')!=scenario_seed:
        raise ValueError(f'Scenario seed mismatch for {csv}: {s.get("scenario_seed")} != {scenario_seed}')
    if s.get('steps',steps)<steps:
        raise ValueError(f'Summary covers fewer than {steps} steps: {summary}')
    expected={'utility','served_ues','relay_served_ues','relay','decision_ms'}
    if not expected.issubset(df.columns):
        raise ValueError(f'Missing columns in {csv}: {expected-set(df.columns)}')
    df=df.iloc[:steps].reset_index(drop=True)
    if not np.isfinite(df[list(expected)].to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite metrics in {csv}')
    return df


def load_all(root:Path,steps:int,scenario_seed:int):
    recover=root/f'recovery_{steps}'
    curr=root/f'curriculum_{steps}'
    matrices={}
    for label,path,names in (
        ('QMIX (direct)',root,[f'rl_seed{s}' for s in (91,92,93)]),
        ('QMIX + Curriculum',curr,[f'curriculum_rl_seed{s}' for s in (91,92,93)]),
        ('Dynamic + Recovery',recover,['recovery_dynamic']),
        ('4A1R + Recovery',recover,['recovery_4a1r']),
        ('3A2R + Recovery',recover,['recovery_3a2r']),
        ('Random',root,['random']),
        ('Local 5A',root,['local5a']),
    ):
        matrices[label]=[
            check_csv(path/f'{name}_steps.csv',path/f'{name}_summary.json',steps,scenario_seed)
            for name in names
        ]
    return matrices


def plot_timeseries(data,steps:int,k:int,key:str,path:Path):
    x=np.arange(1,steps+1)
    fig,ax=plt.subplots(figsize=(10.0,5.35))
    for label in ORDER:
        series=[d[key].to_numpy(dtype=float) for d in data[label]]
        color=COLORS[label]
        if len(series)>1:
            for y in series:
                ax.plot(x,trailing(y,k),color=color,alpha=.11,lw=.8)
        average=np.mean([trailing(y,k) for y in series],axis=0)
        ax.plot(x,average,color=color,lw=2.1,label=label)
    ylab,title=METRICS[key]
    ax.set(xlabel='Evaluation Step',ylabel=ylab,title=title)
    ax.set_xlim(1,steps)
    ax.set_ylim(bottom=0)
    if key=='served_ues':ax.set_ylim(0,90)
    ax.grid(alpha=.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda xx,_:f'{xx:,.0f}'))
    ax.legend(loc='upper right',fontsize=8.5,ncol=2)
    fig.tight_layout()
    fig.savefig(path,dpi=240)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--scenario-seed',type=int,default=3101)
    p.add_argument('--input-root',default='results/thesis_continuous_eval')
    p.add_argument('--outdir',default='results/thesis_continuous_eval/figures_5k_audit')
    p.add_argument('--smooth-window',type=int,default=200)
    args=p.parse_args()
    if args.steps<=0 or args.smooth_window<=0:p.error('steps and smooth-window must be positive')
    data=load_all(Path(args.input_root),args.steps,args.scenario_seed)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    for key in METRICS:
        plot_timeseries(data,args.steps,args.smooth_window,key,out/f'{key}_all_5k.png')
    rows=[]
    for label in ORDER:
        traces=data[label]
        row={'method':label,'n_training_seeds':len(traces) if label.startswith('QMIX') else None,
             'evaluation_steps_per_trajectory':args.steps}
        for name in ('utility','served_ues','relay_served_ues','relay','decision_ms'):
            per_tr=[float(df[name].mean()) for df in traces]
            row[f'mean_{name}']=float(np.mean(per_tr))
            row[f'std_across_training_seeds_{name}']=(float(np.std(per_tr,ddof=1))
                      if len(per_tr)>1 else None)
        row['fraction_steps_with_effective_relay']=float(np.mean(
            [(df['relay_served_ues']>0).mean() for df in traces]))
        row['mean_utility_last_1000']=float(np.mean([
            df['utility'].iloc[-1000:].mean() for df in traces]))
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'all_methods_metrics_5k.csv',index=False)
    direct=summary[summary.method=='QMIX (direct)'].iloc[0]
    curr=summary[summary.method=='QMIX + Curriculum'].iloc[0]
    print(summary[['method','mean_utility','mean_served_ues','mean_relay_served_ues',
                   'mean_relay','mean_decision_ms','mean_utility_last_1000']].to_string(index=False))
    print('Curriculum minus Direct: UE served via Relay per step =',
          float(curr['mean_relay_served_ues']-direct['mean_relay_served_ues']))
    print('Curriculum minus Direct: active Relay UAV per step =',
          float(curr['mean_relay']-direct['mean_relay']))
    metadata={
        'scenario_seed':args.scenario_seed,'steps':args.steps,'smoothing':f'trailing mean {args.smooth_window} actual measurements',
        'rl_training_seeds':[91,92,93],
        'each_policy_simulation':'one long uninterrupted trajectory starting from same scenario seed',
        'direct_rl_traces_are_first_5k_steps_of_existing_20k_data':True,
        'curriculum_rl_requires_new_checkpoint_5k_rollouts':True,
        'comparability':'one scenario, different training seeds for RL; heuristic search uses exact simulator UE positions',
    }
    (out/'figure_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Wrote to:',out.resolve())

if __name__=='__main__': main()
