"""Plot 4A1R / 3A2R vs existing continuous QMIX/Random/Local5A logs.

Do not confuse one 20k-step scenario with a multi-scenario significance test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

METHODS = {
    "rl_seed91": ("QMIX seed 91", "#2563EB"),
    "rl_seed92": ("QMIX seed 92", "#0284C7"),
    "rl_seed93": ("QMIX seed 93", "#0891B2"),
    "random": ("Random", "#737373"),
    "local5a": ("Local 5A", "#16A34A"),
    "fixed_4a1r": ("Fixed-role 4A1R", "#EA580C"),
    "fixed_3a2r": ("Fixed-role 3A2R", "#7C3AED"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/thesis_continuous_eval")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--smooth-window", type=int, default=400)
    args = ap.parse_args()
    if args.smooth_window <= 0:
        ap.error("smooth-window must be positive")
    base = Path(args.input)
    if not base.is_absolute():
        base = ROOT / base
    dest = Path(args.outdir) if args.outdir else base / "figures"
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    data = {}
    summaries = {}
    for stem, (label, color) in METHODS.items():
        csvfile = base / f"{stem}_steps.csv"
        summaryfile = base / f"{stem}_summary.json"
        if not csvfile.exists():
            continue
        if not summaryfile.exists():
            raise FileNotFoundError(f"{summaryfile} not found for {csvfile}")
        df = pd.read_csv(csvfile)
        summ = json.loads(summaryfile.read_text(encoding="utf-8"))
        if df.empty or not np.array_equal(df.step.to_numpy(), np.arange(1, len(df) + 1)):
            raise RuntimeError(f"Invalid time sequence in {csvfile}")
        if len(df) != int(summ["steps"]):
            raise RuntimeError(f"Step count mismatch in {csvfile}")
        if (
            summ.get("number_of_environment_resets") != 1
            or summ.get("evaluation_type") != "one continuous simulator run without episode reset"
        ):
            raise RuntimeError(f"Not a continuous, single-reset run: {summaryfile}")
        data[label] = (df, color)
        summaries[label] = summ

    required = ["Fixed-role 4A1R", "Fixed-role 3A2R"]
    if any(item not in data for item in required):
        ap.error("Please run both fixed 4A1R and 3A2R policies before plotting")
    durations = {len(df) for df, _ in data.values()}
    conditions = {(d["scenario_seed"], d["steps"]) for d in summaries.values()}
    if len(durations) != 1 or len(conditions) != 1:
        raise RuntimeError("Policy durations/scenarios do not match; cannot plot them together")

    specs = [
        (
            "utility",
            "Mean user-satisfaction utility",
            "fixed_mix_utility_continuous.png",
        ),
        (
            "served_ues",
            "UEs served with nonzero rate",
            "fixed_mix_served_ues_continuous.png",
        ),
        (
            "relay_served_ues",
            "UEs served via actual Relay route",
            "fixed_mix_relay_service_continuous.png",
        ),
        (
            "decision_ms",
            "Policy decision time (ms/slot)",
            "fixed_mix_decision_time_continuous.png",
        ),
    ]
    for key, ylab, fname in specs:
        fig, ax = plt.subplots(figsize=(10.6, 5.4))
        for name, (df, color) in data.items():
            x = df.step.to_numpy()
            y = df[key].to_numpy(dtype=float)
            smoothed = (
                pd.Series(y)
                .rolling(args.smooth_window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            ax.plot(x, smoothed, color=color, lw=2.0, label=name)
        ax.set_xlim(1, next(iter(durations)))
        ax.set_xlabel("Simulation step (one run; no episode resets)")
        ax.set_ylabel(ylab)
        ax.set_title(ylab + " | moving average " + str(args.smooth_window) + " slots")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(dest / fname, dpi=250)
        plt.close(fig)

    # Extra: show why the observed four-active 3A1R phase is not a separate
    # constant-composition policy for a five-UAV two-role simulator.
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for name in required:
        df, color = data[name]
        x = df.step.to_numpy()
        y = (df["selected_composition"] == "3A1R").astype(float).to_numpy()
        yy = pd.Series(y).rolling(args.smooth_window, center=True, min_periods=1).mean().to_numpy()
        ax.plot(x, yy, color=color, label=name, lw=2)
    ax.set_xlabel("Simulation step (no reset)")
    ax.set_ylabel("Fraction of nearby slots with 3A1R")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(1, next(iter(durations)))
    ax.set_title("Observed 3A1R windows (one UAV unavailable owing to battery)")
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "fixed_mix_3a1r_windows.png", dpi=250)
    plt.close(fig)

    rows = []
    for name, (df, color) in data.items():
        rows.append(
            {
                "method": name,
                "scenario_seed": summaries[name]["scenario_seed"],
                "continuous_steps": len(df),
                "mean_utility": float(df.utility.mean()),
                "mean_served_ues": float(df.served_ues.mean()),
                "mean_ues_via_relay": float(df.relay_served_ues.mean()),
                "mean_decision_ms": float(df.decision_ms.mean()),
                "fraction_slots_effective_relay": float((df.relay_served_ues > 0).mean()),
                "four_active_3A1R_steps": int((df.selected_composition == "3A1R").sum())
                if "selected_composition" in df.columns
                else None,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(dest / "fixed_mix_vs_existing_summary.csv", index=False)
    print(table.to_string(index=False))
    print("Created plots in", dest)


if __name__ == "__main__":
    main()
