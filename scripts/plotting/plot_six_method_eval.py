"""Six-method operational comparison, with one common evaluation timeline.

Usage:
    python scripts/plotting/plot_six_method_eval.py --steps 5000

Inputs from measured simulator rollouts; never creates missing metrics or
replaces missing 5A+Recovery with the old, non-comparable Local 5A trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ORDER = (
    "QMIX",
    "Dynamic + Recovery",
    "5A + Recovery",
    "4A1R + Recovery",
    "3A2R + Recovery",
    "Random",
)
COLORS = {
    "QMIX": "#2563EB",
    "Dynamic + Recovery": "#DC2626",
    "5A + Recovery": "#16A34A",
    "4A1R + Recovery": "#F59E0B",
    "3A2R + Recovery": "#8B5CF6",
    "Random": "#687587",
}
METRICS = {
    "utility": ("Mean user satisfaction", "Utility of the six methods"),
    "served_ues": ("UEs served / step", "UEs receiving throughput"),
    "relay_served_ues": (
        "UEs served via Relay / step",
        "Actual traffic carried by Relay",
    ),
    "relay": ("Active Relay UAVs / step", "Relay roles selected"),
}


def read_trace(csv_path: Path, summary_path: Path, steps: int, scenario_seed: int) -> pd.DataFrame:
    if not csv_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Required measured trace missing: {csv_path} or {summary_path}. "
            "Run the new 5A + Recovery baseline first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("scenario_seed") != scenario_seed or summary.get("steps", 0) < steps:
        raise ValueError(f"Incompatible steps/scenario in {summary_path}")
    if summary.get("number_of_environment_resets") != 1:
        raise ValueError(f"Expected one continuous rollout in {summary_path}")
    df = pd.read_csv(csv_path)
    if len(df) < steps or not np.array_equal(
        df["step"].iloc[:steps].to_numpy(), np.arange(1, steps + 1)
    ):
        raise ValueError(f"Invalid Evaluation Step sequence in {csv_path}")
    required = ["utility", "served_ues", "relay_served_ues", "relay", "decision_ms"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing metrics in {csv_path}: {set(required) - set(df.columns)}")
    df = df.iloc[:steps].reset_index(drop=True)
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError(f"NaN/Inf metrics in {csv_path}")
    if (
        summary["steps"] == steps
        and abs(float(df["utility"].mean()) - float(summary["mean_utility"])) > 1e-8
    ):
        raise ValueError(f"Summary/CSV utility mismatch: {csv_path}")
    return df


def load_data(
    root: Path,
    recovery_dir: Path,
    steps: int,
    scenario_seed: int,
    rl_seeds: tuple[int, ...],
):
    locations = {
        "QMIX": [(root, f"rl_seed{i}") for i in rl_seeds],
        "Dynamic + Recovery": [(recovery_dir, "recovery_dynamic")],
        "5A + Recovery": [(recovery_dir, "recovery_5a")],
        "4A1R + Recovery": [(recovery_dir, "recovery_4a1r")],
        "3A2R + Recovery": [(recovery_dir, "recovery_3a2r")],
        "Random": [(root, "random")],
    }
    data = {
        method: [
            read_trace(
                directory / f"{stem}_steps.csv",
                directory / f"{stem}_summary.json",
                steps,
                scenario_seed,
            )
            for directory, stem in paths
        ]
        for method, paths in locations.items()
    }
    only_access = data["5A + Recovery"][0]
    if not ((only_access["relay"] == 0) & (only_access["relay_served_ues"] == 0)).all():
        raise ValueError("5A + Recovery must never select Relay or serve traffic via Relay")
    for name in (
        "Dynamic + Recovery",
        "5A + Recovery",
        "4A1R + Recovery",
        "3A2R + Recovery",
    ):
        p = (
            recovery_dir
            / f"recovery_{ {'Dynamic + Recovery': 'dynamic', '5A + Recovery': '5a', '4A1R + Recovery': '4a1r', '3A2R + Recovery': '3a2r'}[name] }_summary.json"
        )
        summary = json.loads(p.read_text(encoding="utf-8"))
        if summary.get("zero_service_patience") != 3:
            raise ValueError(f"Require identical 3-step recovery patience: {p}")
    return data


def trailing_mean(y: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(y).rolling(window=n, min_periods=1).mean().to_numpy()


def plot_series(data, steps, window, key, dest: Path, early: int | None = None):
    fig, ax = plt.subplots(figsize=(10.3, 5.5))
    x = np.arange(1, steps + 1)
    upto = min(early or steps, steps)
    for method in ORDER:
        raw = np.stack([df[key].to_numpy(dtype=float) for df in data[method]])
        color = COLORS[method]
        if len(raw) > 1:
            for row in raw:
                ax.plot(
                    x[:upto],
                    trailing_mean(row, window)[:upto],
                    lw=0.85,
                    alpha=0.2,
                    color=color,
                )
        averaged = np.mean(raw, axis=0)
        ax.plot(
            x[:upto],
            trailing_mean(averaged, window)[:upto],
            lw=2.25,
            color=color,
            label=method,
        )
    ylabel, title = METRICS[key]
    if early is not None:
        title += f" — first {upto:,} steps"
    ax.set(xlabel="Evaluation Step", ylabel=ylabel, title=title, xlim=(1, upto))
    ax.set_ylim(bottom=0)
    if key == "served_ues":
        ax.set_ylim(top=90)
    ax.grid(alpha=0.19)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda xx, _: f"{xx:,.0f}"))
    ax.legend(ncol=2, fontsize=8.8, framealpha=0.95, loc="upper right")
    fig.tight_layout()
    fig.savefig(dest, dpi=250)
    plt.close(fig)


def make_table(data, steps: int) -> pd.DataFrame:
    rows = []
    for method in ORDER:
        traces = data[method]
        row = {
            "method": method,
            "evaluation_steps_per_trajectory": steps,
            "training_seeds": len(traces) if method == "QMIX" else None,
            "mean_utility": float(np.mean([df["utility"].mean() for df in traces])),
            "mean_utility_last_1000": float(
                np.mean([df["utility"].iloc[-min(steps, 1000) :].mean() for df in traces])
            ),
            "mean_served_ues": float(np.mean([df["served_ues"].mean() for df in traces])),
            "mean_relay_served_ues": float(
                np.mean([df["relay_served_ues"].mean() for df in traces])
            ),
            "mean_active_relays": float(np.mean([df["relay"].mean() for df in traces])),
            "fraction_relay_serves_ues": float(
                np.mean([(df["relay_served_ues"] > 0).mean() for df in traces])
            ),
            "mean_decision_ms": float(np.mean([df["decision_ms"].mean() for df in traces])),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def bar_chart(df: pd.DataFrame, metric: str, xlabel: str, ylabel: str, path: Path, log_y=False):
    y = df[metric].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    xx = np.arange(len(ORDER))
    bars = ax.bar(
        xx,
        y,
        width=0.67,
        color=[COLORS[label] for label in ORDER],
        edgecolor="#333333",
        linewidth=0.5,
    )
    for bar, val in zip(bars, y):
        ax.annotate(
            f"{val:.3f}" if not log_y else f"{val:.2f}",
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8.9,
        )
    if log_y:
        if (y <= 0).any():
            raise ValueError("Log-scale latency requires strictly positive observations")
        ax.set_yscale("log")
        ax.set_ylim(max(0.01, min(y) * 0.45), max(y) * 3)
    else:
        ax.set_ylim(0, max(y) * 1.2)
    ax.set_xticks(xx, [s.replace(" + ", "\n+ ") for s in ORDER], fontsize=8.4)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--scenario-seed", type=int, default=3101)
    ap.add_argument("--rl-seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--input-root", default="results/thesis_continuous_eval")
    ap.add_argument("--recovery-dir", default=None)
    ap.add_argument("--outdir", default="results/thesis_continuous_eval/final_six_figures")
    ap.add_argument("--smooth-window", type=int, default=200)
    ap.add_argument("--early-steps", type=int, default=1200)
    args = ap.parse_args()
    if args.steps <= 0 or args.smooth_window <= 0 or args.early_steps <= 0:
        ap.error("steps, smooth-window and early-steps must be positive")
    root = Path(args.input_root)
    recover = Path(args.recovery_dir) if args.recovery_dir else root / f"recovery_{args.steps}"
    outdir = Path(args.outdir)
    data = load_data(root, recover, args.steps, args.scenario_seed, tuple(args.rl_seeds))
    outdir.mkdir(parents=True, exist_ok=True)
    for key in METRICS:
        plot_series(data, args.steps, args.smooth_window, key, outdir / f"{key}_all_six.png")
    plot_series(
        data,
        args.steps,
        min(50, args.smooth_window),
        "utility",
        outdir / "utility_early_all_six.png",
        early=args.early_steps,
    )
    table = make_table(data, args.steps)
    table.to_csv(outdir / "all_six_metrics.csv", index=False)
    bar_chart(
        table,
        "mean_utility",
        "",
        "Mean user satisfaction",
        outdir / "mean_utility_all_six.png",
    )
    bar_chart(
        table,
        "mean_decision_ms",
        "",
        "Policy decision time (ms), log scale",
        outdir / "decision_time_all_six.png",
        log_y=True,
    )
    metadata = {
        "scenario_seed": args.scenario_seed,
        "steps": args.steps,
        "method_names": list(ORDER),
        "no_curriculum_method_plotted": True,
        "old_local_5a_trace_used": False,
        "shared_recovery_logic": "same _ServiceRecoveryMixin after 3 zero-service steps",
        "qmix_aggregation": f"mean of trained model seeds {args.rl_seeds}",
        "other_policies": "one measured trajectory per policy on same scenario seed",
        "smoothing": f"trailing average {args.smooth_window} steps (no fabricated samples)",
        "search_baselines_privileged_current_state_information": True,
    }
    (outdir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print(f"Figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
