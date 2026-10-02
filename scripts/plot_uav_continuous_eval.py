"""Plot exact never-reset operational rollouts; DO NOT mix with episodic files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_uav_operational_eval import draw_actual_map

COLORS = {
    'Random': '#8A8F99',
    'Local 5A': '#1D9B76',
    'QMIX': '#2563EB',
    'Exhaustive dynamic': '#C24136',
}


def load_runs(path: Path) -> dict[str, list[pd.DataFrame]]:
    data = {}
    expected = [('random', 'Random'), ('local5a', 'Local 5A'), ('exhaustive', 'Exhaustive dynamic')]
    for filename, label in expected:
        p = path / f'{filename}_steps.csv'
        if p.exists():
            df = pd.read_csv(p)
            _validate(df, p)
            data[label] = [df]
    rls = []
    for p in sorted(path.glob('rl_seed*_steps.csv')):
        df = pd.read_csv(p)
        _validate(df, p)
        rls.append(df)
    if rls:
        data['QMIX'] = rls
    if not data:
        raise FileNotFoundError(f'No continuous *_steps.csv files found under {path}')
    lengths = [len(df) for frames in data.values() for df in frames]
    if len(set(lengths)) != 1:
        raise RuntimeError('Runs have unequal duration: do not compare partial and completed runs')
    return data


def _validate(df, path):
    if not np.array_equal(df.step.to_numpy(), np.arange(1, len(df) + 1)):
        raise RuntimeError(f'Non-monotonic or missing step IDs: {path}')


def smooth(x, k):
    return pd.Series(x).rolling(window=k, center=True, min_periods=1).mean().to_numpy()


def plot_curves(data, outdir, window):
    specs = [
        ('utility', 'User-satisfaction utility', 'utility_long_run.png'),
        ('served_ues', 'UEs served with nonzero rate', 'served_ues_long_run.png'),
        ('relay_served_ues', 'UEs actually served via Relay', 'relay_ues_long_run.png'),
        ('relay', 'Active Relay UAVs', 'relay_count_long_run.png'),
        ('mean_battery', 'Mean battery of UAV fleet', 'fleet_battery_long_run.png'),
    ]
    for key, ylab, fname in specs:
        fig, ax = plt.subplots(figsize=(10.0, 5.1))
        for name, frames in data.items():
            y = np.mean([df[key].to_numpy(dtype=float) for df in frames], axis=0)
            t = frames[0].step.to_numpy()
            ax.plot(t, y, color=COLORS[name], alpha=.11, lw=.5)
            ax.plot(t, smooth(y, window), color=COLORS[name], lw=2.0,
                    label=name + (f' ({len(frames)} training seeds)' if len(frames)>1 else ''))
        ax.set_xlabel('Simulation step (one continuous run; no resets)')
        ax.set_ylabel(ylab)
        ax.set_xlim(1, len(t))
        ax.set_title(ylab + ' during uninterrupted operation')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=250)
        plt.close(fig)

    if 'QMIX' in data:
        frames = data['QMIX']
        df = frames[0]
        t = df.step.to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.5), sharex=True)
        axes[0].plot(t, smooth(df.utility, window), color=COLORS['QMIX'])
        axes[0].set_ylabel('Utility')
        for key, color, name in [
            ('served_ues', '#1D9B76', 'UEs served'),
            ('fully_satisfied_ues', '#C24136', 'Demand satisfied'),
            ('relay_served_ues', '#9746B0', 'UEs via Relay'),
        ]:
            axes[1].plot(t, smooth(df[key], window), color=color, label=name)
        axes[1].set_ylabel('UEs')
        axes[1].legend(fontsize=7)
        for key, color, name in [
            ('relay', '#9746B0', 'Relay UAVs'),
            ('returning', '#BA7F20', 'Returning UAVs'),
            ('charging', '#C24136', 'Charging UAVs'),
        ]:
            axes[2].plot(t, smooth(df[key], window), color=color, label=name)
        axes[2].set_ylabel('UAV count')
        axes[2].legend(fontsize=7)
        axes[2].set_xlabel('Simulation step (one continuous run; no resets)')
        for ax in axes:
            ax.grid(alpha=.17)
            ax.set_xlim(1, len(t))
        fig.suptitle('One uninterrupted QMIX run: service, Relay and battery lifecycle')
        fig.tight_layout()
        fig.savefig(outdir / 'qmix_continuous_temporal_diagnostics.png', dpi=250)
        plt.close(fig)

    rows=[]
    for name, frames in data.items():
        means=[float(f.utility.mean()) for f in frames]
        rows.append({
            'method':name,
            'steps_per_policy':len(frames[0]),
            'mean_utility':float(np.mean(means)),
            'std_between_training_seeds':float(np.std(means,ddof=1)) if len(means)>1 else None,
            'mean_served_ues':float(np.mean([f.served_ues.mean() for f in frames])),
            'mean_ues_via_relay':float(np.mean([f.relay_served_ues.mean() for f in frames])),
            'mean_active_relay_uavs':float(np.mean([f.relay.mean() for f in frames])),
            'mean_decision_ms':float(np.mean([f.decision_ms.mean() for f in frames])),
            'mean_return_events_per_run':float(np.mean([f.return_events_this_slot.sum() for f in frames])),
            'mean_rejoin_events_per_run':float(np.mean([f.rejoin_events_this_slot.sum() for f in frames])),
        })
    result=pd.DataFrame(rows)
    result.to_csv(outdir / 'continuous_operational_summary.csv', index=False)
    print(result.to_string(index=False))


def plot_real_snapshots(parent, outdir):
    path=parent/'rl_continuous_case_study.json'
    if not path.exists():
        return
    trace=json.loads(path.read_text(encoding='utf-8'))
    frames=trace.get('case_frames',{})
    numerics=sorted(int(k) for k in frames if k.isdigit())
    if not numerics:
        return
    chosen=[0,20,150,500,1000,numerics[-1]]
    chosen=[k for k in dict.fromkeys(chosen) if str(k) in frames]
    while len(chosen)<6 and numerics:
        k=numerics.pop(len(numerics)//2)
        if k not in chosen:
            chosen.append(k)
    chosen=sorted(chosen)[:6]
    fig,axs=plt.subplots(2,3,figsize=(15.5,10.5))
    for a in axs.flat:
        a.set_axis_off()
    for i,k in enumerate(chosen):
        a=axs.flat[i]
        a.set_axis_on()
        draw_actual_map(a,frames[str(k)],trace, f'Continuous simulation t={k}',with_legend=(i==0))
    fig.suptitle('Real served UEs, coverage and executed routes without episode reset',fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir/'qmix_continuous_six_snapshots.png',dpi=190)
    plt.close(fig)

    snap=trace.get('best_observed_relay')
    if snap is not None and int(trace.get('best_relay_served_ues',0))>0:
        fig,ax=plt.subplots(figsize=(8.1,7.4))
        draw_actual_map(ax,snap,trace,
                        f'Measured Relay service at t={snap["step_global"]}',with_legend=True)
        fig.tight_layout()
        fig.savefig(outdir/'qmix_actual_relay_chain.png',dpi=250)
        plt.close(fig)
    else:
        (outdir/'NO_RELAY_IN_THIS_RUN.txt').write_text(
            'No UE was served via a Relay in the captured run; no fictional chain drawn.\n',
            encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default='results/thesis_continuous_eval')
    ap.add_argument('--outdir',default=None)
    ap.add_argument('--smooth-window',type=int,default=200)
    args=ap.parse_args()
    if args.smooth_window < 1:
        ap.error('--smooth-window must be positive')
    base=Path(args.input)
    if not base.is_absolute():base=ROOT/base
    dest=Path(args.outdir) if args.outdir else base/'figures'
    if not dest.is_absolute():dest=ROOT/dest
    dest.mkdir(parents=True,exist_ok=True)
    data=load_runs(base)
    plot_curves(data,dest,args.smooth_window)
    plot_real_snapshots(base,dest)
    print('Saved long-horizon figures to',dest)


if __name__=='__main__':main()
