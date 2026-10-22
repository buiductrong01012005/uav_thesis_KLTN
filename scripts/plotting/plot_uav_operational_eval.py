"""Figures from measured operational simulator rollouts, never synthetic links."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from src.envs.v4_fixed_mobile_battery_env import ACTIVE, RETURNING, CHARGING
from src.envs.v0_env import ACCESS, RELAY

COLORS = {
    "Random": "#8A8F99",
    "Local 5A heuristic": "#1D9B76",
    "QMIX + spatial encoder": "#2563EB",
    "Exhaustive dynamic": "#C24136",
}


def frame_center(cell, m):
    r, c = map(float, cell)
    return np.array([(c + 0.5) * m, (r + 0.5) * m], dtype=float)


def load_rollouts(base):
    files = sorted(base.glob("*_steps.csv"))
    if not files:
        raise FileNotFoundError(f"No *_steps.csv files in {base}; run the evaluation first")
    data = {p.stem.removesuffix("_steps"): pd.read_csv(p) for p in files}
    for k, df in data.items():
        if not np.array_equal(df.step.to_numpy(), np.arange(1, len(df) + 1)):
            raise RuntimeError(f"Invalid step sequence: {k}")
    return data


def moving(y, w):
    return pd.Series(y).rolling(window=w, min_periods=1, center=True).mean().to_numpy()


def pooled(data):
    out = {}
    rl_keys = sorted(k for k in data if k.startswith("rl_seed"))
    if rl_keys:
        x = data[rl_keys[0]].step.to_numpy()
        if any(not np.array_equal(x, data[k].step.to_numpy()) for k in rl_keys[1:]):
            raise RuntimeError("Training-seed rollouts have different step IDs")
        out["QMIX + spatial encoder"] = [data[k] for k in rl_keys]
    for key, label in [
        ("random", "Random"),
        ("local5a", "Local 5A heuristic"),
        ("exhaustive", "Exhaustive dynamic"),
    ]:
        if key in data:
            out[label] = [data[key]]
    return out


def plot_operational(data, dest, w):
    groups = pooled(data)
    metrics = [
        ("utility", "Instantaneous utility", "utility_vs_simulation_step.png"),
        (
            "served_ues",
            "UEs receiving nonzero rate",
            "served_ues_vs_simulation_step.png",
        ),
        ("relay", "Active Relay UAVs", "relay_count_vs_simulation_step.png"),
    ]
    for key, yl, name in metrics:
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for label, frames in groups.items():
            if any(len(z) != len(frames[0]) for z in frames):
                raise RuntimeError("Policies have different evaluation lengths")
            yy = np.vstack([f[key].to_numpy(dtype=float) for f in frames])
            mean = np.mean(yy, axis=0)
            x = frames[0].step.to_numpy(dtype=int)
            c = COLORS[label]
            ax.plot(x, mean, color=c, alpha=0.12, lw=0.45)
            ax.plot(
                x,
                moving(mean, w),
                color=c,
                lw=2.0,
                label=label + (f" ({len(frames)} training seeds)" if len(frames) > 1 else ""),
            )
        ax.set(
            xlabel="Cumulative simulation step (environment resets every 150 slots)",
            ylabel=yl,
            title=f"Operational evaluation: {yl.lower()}",
            xlim=(1, int(x[-1])),
        )
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(dest / name, dpi=300)
        plt.close(fig)

    # Detailed first case, RL model used for the snapshots.
    case_keys = sorted(k for k in data if k.startswith("rl_seed"))
    if case_keys:
        case_key = case_keys[0]
        tracepath = dest.parent / "rl_case_study.json"
        if tracepath.exists():
            case_seed = json.loads(tracepath.read_text(encoding="utf-8"))["case_seed"]
            case_df = data[case_key][data[case_key].scenario_seed == case_seed]
            fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True)
            axes[0].plot(
                case_df.episode_step,
                case_df.utility,
                color=COLORS["QMIX + spatial encoder"],
            )
            axes[0].set_ylabel("Utility")
            axes[1].plot(
                case_df.episode_step,
                case_df.served_ues,
                color="#1D9B76",
                label="Any nonzero rate",
            )
            axes[1].plot(
                case_df.episode_step,
                case_df.fully_satisfied_ues,
                color="#C24136",
                label="Full demand met",
            )
            axes[1].plot(
                case_df.episode_step,
                case_df.relay_served_ues,
                color="#9746B0",
                label="Served via Relay",
            )
            axes[1].set_ylabel("UE count")
            axes[1].legend(ncol=3, fontsize=7.8)
            axes[2].plot(
                case_df.episode_step,
                case_df.relay,
                color="#9746B0",
                label="Active Relay UAVs",
            )
            axes[2].plot(
                case_df.episode_step,
                case_df.returning,
                color="#BA7F20",
                label="Returning",
            )
            axes[2].plot(
                case_df.episode_step,
                case_df.charging,
                color="#C24136",
                label="Charging",
            )
            axes[2].set_ylabel("UAV count")
            axes[2].legend(fontsize=8)
            axes[2].set_xlabel("Episode step (150-slot disaster-recovery case)")
            for a in axes:
                a.grid(alpha=0.2)
            fig.suptitle(f"Measured operation of {case_key} on scenario {case_seed}")
            fig.tight_layout()
            fig.savefig(dest / "rl_single_episode_temporal.png", dpi=300)
            plt.close(fig)

    summ_rows = []
    for label, frames in groups.items():
        # Each policy's scenario variability uses *the same scenario episodes*.
        # RL scenario score first averages multiple trained policies per scenario.
        seed_rows = []
        step_means = []
        for d in frames:
            m = d.groupby("scenario_seed", as_index=False)["utility"].mean()
            m = m.rename(columns={"utility": "episode_mean"})
            seed_rows.append(m)
            step_means.append(float(d.utility.mean()))
        grid = pd.concat([m.assign(run=i) for i, m in enumerate(seed_rows)])
        scenario_mean = grid.groupby("scenario_seed").episode_mean.mean()
        summ_rows.append(
            {
                "method": label,
                "simulation_steps_per_training_seed": int(len(frames[0])),
                "scenario_count": int(len(scenario_mean)),
                "mean_utility_per_step": float(np.mean(step_means)),
                "std_across_scenario_means": float(scenario_mean.std(ddof=1)),
                "std_across_training_seed_means": float(np.std(step_means, ddof=1))
                if len(frames) > 1
                else np.nan,
                "mean_served_ues": float(np.mean([f.served_ues.mean() for f in frames])),
                "mean_fully_satisfied_ues": float(
                    np.mean([f.fully_satisfied_ues.mean() for f in frames])
                ),
                "mean_effective_relay_routes": float(
                    np.mean([f.effective_relay_routes.mean() for f in frames])
                ),
                "mean_decision_ms": float(np.mean([f.decision_ms.mean() for f in frames])),
            }
        )
    table = pd.DataFrame(summ_rows)
    table.to_csv(dest / "operational_summary_comparable.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = table.method.tolist()
    bars = ax.bar(
        labels,
        table.mean_utility_per_step.to_numpy(),
        color=[COLORS[label] for label in labels],
        edgecolor="#333333",
        linewidth=0.6,
    )
    for b, v in zip(bars, table.mean_utility_per_step):
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(v) + 0.01,
            f"{float(v):.3f}",
            ha="center",
        )
    ax.set(
        ylabel="Mean utility across all evaluated simulation steps",
        title="Operational evaluation: same scenario sequence",
    )
    ax.set_ylim(0, max(table.mean_utility_per_step) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(dest / "operational_comparison_mean.png", dpi=300)
    plt.close(fig)
    print(table.to_string(index=False))
    return table


def draw_actual_map(ax, snap, trace, title, with_legend=False):
    """All routes and UE service derived from captured executed network result."""
    m = float(trace["cell_size_m"])
    region = np.asarray(trace["affected_mask"], dtype=int)
    ax.imshow(
        np.ma.masked_where(region == 0, region),
        origin="lower",
        extent=(0, 20 * m, 0, 20 * m),
        cmap="Purples",
        alpha=0.18,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    bs = np.asarray(trace["bs_xy_m"], dtype=float)
    failed = int(trace["failed_bs_idx"])
    bidx = [i for i in range(len(bs)) if i != failed]
    ax.scatter(bs[bidx, 0], bs[bidx, 1], s=40, marker="s", color="#0C806F", zorder=6)
    ax.scatter([bs[failed, 0]], [bs[failed, 1]], s=90, marker="X", color="#CC3030", zorder=6)
    for i in bidx:
        ax.annotate(f"BS{i}", bs[i], xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    stations = np.asarray([frame_center(c, m) for c in trace["charging_station_cells"]])
    ax.scatter(stations[:, 0], stations[:, 1], s=70, marker="^", color="#986C18", zorder=6)

    roles = np.asarray(snap["uav_roles"], dtype=int)
    status = np.asarray(snap["uav_status"], dtype=int)
    pos = np.asarray([frame_center(c, m) for c in snap["uav_cells"]])
    ues = np.asarray([frame_center(c, m) for c in snap["ue_cells"]])
    rates = np.asarray(snap["ue_rate_mbps"], dtype=float)
    user_class = np.zeros(len(ues), dtype=int)
    for a, users in snap["association"].items():
        if snap["routes"].get(a) is not None:
            for uid in users:
                if rates[uid] > 1e-9:
                    user_class[int(uid)] = 2
        else:
            for uid in users:
                if rates[uid] > 1e-9:
                    user_class[int(uid)] = 1
    for cl, color, name in [
        (0, "#989EA6", "UE unserved"),
        (1, "#16A085", "UE via direct"),
        (2, "#E16A3D", "UE via Relay"),
    ]:
        ids = user_class == cl
        if ids.any():
            ax.scatter(ues[ids, 0], ues[ids, 1], s=9, color=color, alpha=0.65, zorder=3)

    # Access coverage is a geometric feasibility radius, not a promise of service.
    for i in range(len(pos)):
        if status[i] == ACTIVE and roles[i] == ACCESS:
            ax.add_patch(
                Circle(
                    pos[i],
                    float(trace["access_radius_m"]),
                    edgecolor="#2264C7",
                    facecolor="#246BD0",
                    alpha=0.07,
                    linewidth=1.0,
                    zorder=2,
                )
            )
    active_links = set()
    for a, rid in snap["routes"].items():
        aid = int(a)
        served = any(rates[int(u)] > 1e-9 for u in snap["association"].get(a, []))
        if not served or snap["backhaul_caps_mbps"].get(a, 0) <= 1e-9:
            continue
        who = aid if rid is None else int(rid)
        px = pos[who]
        feasible = [
            (j, float(np.linalg.norm(bs[j] - px)))
            for j in bidx
            if float(np.linalg.norm(bs[j] - px)) <= float(trace["backhaul_radius_m"]) + 1e-8
        ]
        if not feasible:
            raise RuntimeError(f"Served Access {a} has no feasible route to a gateway")
        # The model chooses maximum Shannon capacity; at equal PHY params and
        # fixed heights it is equivalent to choosing the closest feasible BS.
        best_bs = min(feasible, key=lambda item: item[1])[0]
        if rid is not None:
            ax.plot(
                [pos[aid, 0], pos[who, 0]],
                [pos[aid, 1], pos[who, 1]],
                color="#D45A25",
                lw=2.1,
                zorder=4,
            )
            if (who, best_bs) not in active_links:
                ax.plot(
                    [px[0], bs[best_bs, 0]],
                    [px[1], bs[best_bs, 1]],
                    color="#D45A25",
                    lw=1.8,
                    linestyle="--",
                    zorder=4,
                )
                active_links.add((who, best_bs))
        else:
            ax.plot(
                [px[0], bs[best_bs, 0]],
                [px[1], bs[best_bs, 1]],
                color="#0C806F",
                lw=1.9,
                zorder=4,
            )

    for i, p in enumerate(pos):
        if status[i] == RETURNING:
            marker, color, suffix = "v", "#B78728", "Return"
        elif status[i] == CHARGING:
            marker, color, suffix = "P", "#B78728", "Charge"
        elif roles[i] == RELAY:
            marker, color, suffix = "D", "#A83B9E", "Relay"
        else:
            marker, color, suffix = "o", "#246BD0", "Access"
        ax.scatter(
            [p[0]],
            [p[1]],
            marker=marker,
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=8,
        )
        ax.annotate(
            f"U{i} {suffix}\nE={snap['battery_at_service'][i]:.0f}/100",
            p,
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.5,
            zorder=9,
        )
    nserv = int((rates > 1e-9).sum())
    nrelay = int((user_class == 2).sum())
    ax.set_title(
        f"{title}\nUtility {snap['utility']:.3f} | served {nserv}/{len(ues)} | via Relay {nrelay}",
        fontsize=9,
    )
    ax.set_xlim(0, 20 * m)
    ax.set_ylim(0, 20 * m)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.14)
    if with_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                markerfacecolor="#246BD0",
                color="black",
                label="Access UAV",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                ls="",
                markerfacecolor="#A83B9E",
                color="black",
                label="Relay UAV",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                ls="",
                markerfacecolor="#CC3030",
                color="black",
                label="Failed BS",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                ls="",
                markerfacecolor="#0C806F",
                color="black",
                label="Surviving BS",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                ls="",
                markerfacecolor="#986C18",
                color="black",
                label="Charging station",
            ),
            Line2D([0], [0], color="#0C806F", lw=2, label="Active direct backhaul"),
            Line2D([0], [0], color="#D45A25", lw=2, label="Active relay backhaul"),
            Line2D([0], [0], marker=".", color="#989EA6", ls="", label="UE unserved"),
            Line2D([0], [0], marker=".", color="#16A085", ls="", label="UE served direct"),
            Line2D(
                [0],
                [0],
                marker=".",
                color="#E16A3D",
                ls="",
                label="UE served via Relay",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.90, ncol=2)


def plot_case(base, dest):
    p = base / "rl_case_study.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    case = d["case_frames"]
    steps = [s for s in ["0", "20", "50", "75", "100", "150"] if s in case]
    fig, axes = plt.subplots(2, 3, figsize=(14.1, 9.1))
    for j, ax in enumerate(axes.flat):
        if j < len(steps):
            s = steps[j]
            ttl = (
                f"Scenario {d['case_seed']}, t={s}"
                if s != "0"
                else f"Scenario {d['case_seed']}, initial state"
            )
            draw_actual_map(ax, case[s], d, ttl)
        else:
            ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", ls="", color="#246BD0", label="Access"),
        Line2D([0], [0], marker="D", ls="", color="#A83B9E", label="Relay"),
        Line2D([0], [0], color="#D45A25", lw=2, label="Used Relay route"),
        Line2D([0], [0], color="#0C806F", lw=2, label="Used direct route"),
        Patch(
            edgecolor="#246BD0",
            facecolor="#246BD0",
            alpha=0.15,
            label="Access range 200 m",
        ),
        Line2D([0], [0], marker=".", ls="", color="#E16A3D", label="UE served via Relay"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(
        "Executed UAV configurations; links/UE rates captured inside each service slot",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(dest / "rl_six_snapshots_real_routing.png", dpi=260)
    plt.close(fig)
    best = d.get("best_observed_relay")
    if best and d.get("best_relay_served_ues", 0) > 0:
        fig, ax = plt.subplots(figsize=(7.5, 7.1))
        draw_actual_map(
            ax,
            best,
            d,
            f"Observed Relay-serving state | seed {best['scenario_seed']}, t={best['episode_step']}",
            True,
        )
        fig.tight_layout()
        fig.savefig(dest / "rl_real_relay_chain_highlight.png", dpi=300)
        plt.close(fig)
        print(f"Observed relay-serving snapshot: {d['best_relay_served_ues']} UE(s)")
    else:
        print("No actual served-via-Relay UE was observed; no relay-chain figure fabricated.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/thesis_operational_eval")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--smooth-window", type=int, default=151)
    args = ap.parse_args()
    base = Path(args.input)
    if not base.is_absolute():
        base = ROOT / base
    dest = Path(args.outdir) if args.outdir else base / "figures"
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)
    data = load_rollouts(base)
    plot_operational(data, dest, max(1, args.smooth_window))
    plot_case(base, dest)
    print("Wrote measured charts to", dest)


if __name__ == "__main__":
    main()
