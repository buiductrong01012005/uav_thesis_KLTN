"""Reproducible illustration of the simulated Access/Relay network.

From the project root (PowerShell):
    python scripts/plotting/render_uav_relay_map.py
    python scripts/plotting/render_uav_relay_map.py --scenario-seed 3101 --step 61

Uses the current 3A2R + Recovery policy and an actual in-slot network capture.
No ImageGen, no pre-drawn or invented links. Saves PNG and a JSON snapshot.

NOTE: the simulation snapshot records Access -> Relay assignments, but not the
winning BS ID. For Access/Relay -> BS arrows we reproduce the simulator's
max-capacity gateway selection by the same radio-capacity function. The purple
shaded cells are the affected BS-failure region, *not* surviving-BS coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_uav_continuous_eval import make_continuous_env
from scripts.evaluation.run_uav_operational_eval import counts_from_exact_snapshot
from src.baselines.recovery_aware_role_greedy import RecoveryAwareFixedMixGreedy
from src.envs.v0_env import ACCESS, RELAY
from src.envs.v4_fixed_mobile_battery_env import ACTIVE

PALETTE = {
    "access": "#1768B5",
    "relay": "#B93BCB",
    "direct": "#07956D",
    "via_relay": "#EE641C",
    "unused": "#A5ADB7",
    "failed": "#D63841",
    "bs": "#05A477",
    "station": "#D99300",
    "grid": "#E4EBF2",
    "affected": "#BBA2E4",
    "text": "#132A42",
}


def simulate_snapshot(seed: int, step: int) -> dict:
    """Capture the *executed* in-slot state before the subsequent UE movement."""
    if step < 1:
        raise ValueError("--step must be >= 1")
    env = make_continuous_env(seed, step + 1)
    observation = env.reset(seed=seed)
    policy = RecoveryAwareFixedMixGreedy(env, composition="3A2R", zero_service_patience=3)
    snapshot = None
    for i in range(1, step + 1):
        env._capture_network = False
        actions = policy.act(observation["action_mask"])
        env._capture_network = True
        observation = env.step(actions)
        env._capture_network = False
        if observation["terminated"] or observation["truncated"]:
            raise RuntimeError("Unexpected episode end while preparing a one-reset snapshot")
        snapshot = env.last_network_snapshot
        if snapshot is None or not np.isclose(
            snapshot["utility"], observation["reward"], atol=1e-8
        ):
            raise RuntimeError("Captured service snapshot does not match executed reward")
        policy.observe_service(float(observation["reward"]))
    assert snapshot is not None
    served, fully_satisfied, via_relay, relay_routes = counts_from_exact_snapshot(
        snapshot, env.cfg.ue_demand_mbps
    )
    return {
        "scenario_seed": seed,
        "step_global": step,
        "policy": "3A2R + Recovery (zero-service patience 3)",
        "source": "actual captured simulator state from env.step",
        "note": "Affected area is not a plotted BS coverage radius.",
        "bs_xy_m": env.bs_xy.astype(float).tolist(),
        "failed_bs_idx": int(env.cfg.failed_bs_idx),
        "charging_station_cells": env.charging_stations.astype(float).tolist(),
        "affected_mask": env.affected_mask.astype(int).tolist(),
        "cell_size_m": float(env.cfg.cell_size_m),
        "grid_size": int(env.cfg.grid_size),
        "access_radius_m": float(env.cfg.access_radius_cells * env.cfg.cell_size_m),
        "backhaul_radius_m": float(env.cfg.backhaul_radius_cells * env.cfg.cell_size_m),
        "ue_demand_mbps": float(env.cfg.ue_demand_mbps),
        "bs_height_m": float(env.cfg.bs_height_m),
        "uav_altitude_m": float(env.cfg.uav_altitude_m),
        "served_ues": served,
        "fully_satisfied_ues": fully_satisfied,
        "relay_served_ues": via_relay,
        "effective_relay_routes": relay_routes,
        "snapshot": snapshot,
    }


def cells_to_xy(cells: object, cell_size_m: float) -> np.ndarray:
    """Simulator uses [row, col]; physical XY uses [col, row]."""
    pos = np.asarray(cells, dtype=float).reshape(-1, 2)
    return (pos[:, ::-1] + 0.5) * cell_size_m


def best_gateway(env, uid: int) -> tuple[int, float]:
    """Reproduce max Shannon backhaul choice used by the network layer.

    Network snapshots store Access->Relay assignments but no gateway IDs.
    Therefore choose the BS exactly as the scorer does (maximum valid rate,
    first candidate wins when rates tie), rather than inventing a BS endpoint.
    """
    xy = env._cell_to_xy(env.uav_positions[uid])
    candidates = [i for i in range(len(env.bs_xy)) if i != env.cfg.failed_bs_idx]
    rates = [
        env._bh_a2g_rate(xy, env.cfg.uav_altitude_m, env.bs_xy[i], env.cfg.bs_height_m)
        if env._xy_within_radius_m(xy, env.bs_xy[i], env.cfg.backhaul_radius_cells)
        else 0.0
        for i in candidates
    ]
    max_idx = int(np.argmax(rates))
    return candidates[max_idx], float(rates[max_idx])


def gateway_for_snapshot(data: dict, uid: int) -> tuple[int, float]:
    """Use simulator radio functions at the captured coordinates, no rerun."""
    # A clean environment with matching config provides the original radio
    # functions; set the snapshot UAV cells before computing link capacities.
    env = make_continuous_env(int(data["scenario_seed"]), 2)
    env.reset(seed=int(data["scenario_seed"]))
    env.uav_positions[:] = np.asarray(data["snapshot"]["uav_cells"], dtype=int)
    return best_gateway(env, uid)


def route_links(data: dict) -> list[tuple[int, int, int | None]]:
    """Tuples (access UAV id, selected BS id, relay UAV id or None)."""
    s = data["snapshot"]
    active_roles = np.asarray(s["uav_roles"], dtype=int)
    statuses = np.asarray(s["uav_status"], dtype=int)
    rates = np.asarray(s["ue_rate_mbps"], dtype=float)
    result = []
    for aid_str, rid in s["routes"].items():
        aid = int(aid_str)
        if statuses[aid] != ACTIVE or active_roles[aid] != ACCESS:
            continue
        served = any(rates[int(u)] > 1e-9 for u in s["association"].get(aid_str, []))
        if not served:
            continue
        gateway_uid = aid if rid is None else int(rid)
        if rid is not None and (
            statuses[gateway_uid] != ACTIVE or active_roles[gateway_uid] != RELAY
        ):
            raise ValueError("Route uses a UAV that is not an active Relay")
        bid, capacity = gateway_for_snapshot(data, gateway_uid)
        if capacity <= 0.0:
            raise ValueError(f"Used network path has no viable gateway: UAV {gateway_uid}")
        result.append((aid, bid, rid))
    return result


def draw_map(data: dict, output_path: Path, dpi: int = 200) -> list[tuple[int, int, int | None]]:
    """Render same real locations and routes as the captured service snapshot."""
    s = data["snapshot"]
    m = float(data["cell_size_m"])
    side = int(data["grid_size"]) * m
    bs = np.asarray(data["bs_xy_m"], dtype=float)
    failed = int(data["failed_bs_idx"])
    uav = cells_to_xy(s["uav_cells"], m)
    ues = cells_to_xy(s["ue_cells"], m)
    chargers = cells_to_xy(data["charging_station_cells"], m)
    roles = np.asarray(s["uav_roles"], dtype=int)
    status = np.asarray(s["uav_status"], dtype=int)
    rates = np.asarray(s["ue_rate_mbps"], dtype=float)
    association = s["association"]
    routes = s["routes"]
    link_rows = route_links(data)
    shown = np.zeros(len(ues), dtype=int)
    for aid_str, ids in association.items():
        cls = 2 if routes[aid_str] is not None else 1
        for ue in ids:
            if rates[int(ue)] > 1e-9:
                shown[int(ue)] = cls

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10.6, 10.6), facecolor="white")
    ax.set_facecolor("white")
    affected = np.asarray(data["affected_mask"], dtype=float)
    ax.imshow(
        np.ma.masked_where(affected == 0, affected),
        extent=[0, side, 0, side],
        origin="lower",
        interpolation="nearest",
        cmap="Purples",
        alpha=0.18,
        vmin=0.0,
        vmax=1.0,
        zorder=0,
    )
    for coordinate in np.arange(0, side + m, m):
        ax.axvline(coordinate, color=PALETTE["grid"], lw=0.50, zorder=0)
        ax.axhline(coordinate, color=PALETTE["grid"], lw=0.50, zorder=0)

    # Coverage is geometric reach, not a guarantee of assigned UE service.
    for i, center in enumerate(uav):
        if status[i] != ACTIVE:
            continue
        if roles[i] == ACCESS:
            ax.add_patch(
                Circle(
                    center,
                    float(data["access_radius_m"]),
                    facecolor=PALETTE["access"],
                    edgecolor=PALETTE["access"],
                    lw=1.1,
                    alpha=0.085,
                    zorder=1,
                )
            )
        elif roles[i] == RELAY:
            ax.add_patch(
                Circle(
                    center,
                    float(data["backhaul_radius_m"]),
                    fill=False,
                    edgecolor=PALETTE["relay"],
                    lw=1.7,
                    linestyle=(0, (5, 5)),
                    alpha=0.65,
                    zorder=1,
                )
            )

    for cls, key, size in [(0, "unused", 15), (1, "direct", 19), (2, "via_relay", 19)]:
        chosen = shown == cls
        if np.any(chosen):
            ax.scatter(
                ues[chosen, 0],
                ues[chosen, 1],
                s=size,
                c=PALETTE[key],
                alpha=0.88,
                edgecolors="white",
                lw=0.24,
                zorder=3,
            )

    ax.scatter(
        bs[:, 0],
        bs[:, 1],
        s=125,
        c=PALETTE["bs"],
        marker="s",
        edgecolor="white",
        lw=1.25,
        zorder=7,
    )
    ax.scatter(
        [bs[failed, 0]],
        [bs[failed, 1]],
        s=225,
        c=PALETTE["failed"],
        marker="X",
        edgecolor="white",
        lw=1.4,
        zorder=9,
    )
    ax.scatter(
        chargers[:, 0],
        chargers[:, 1],
        s=145,
        c=PALETTE["station"],
        marker="^",
        edgecolor="white",
        lw=1.0,
        zorder=8,
    )
    for i, position in enumerate(bs):
        ax.annotate(
            f"BS{i}",
            position,
            xytext=(9, 6),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color=PALETTE["text"],
            zorder=11,
        )

    used_second_hop = set()
    for aid, bid, rid in link_rows:
        if rid is not None:
            ax.annotate(
                "",
                xy=uav[int(rid)],
                xytext=uav[aid],
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=2.5,
                    color=PALETTE["via_relay"],
                    mutation_scale=17,
                    shrinkA=13,
                    shrinkB=15,
                ),
                zorder=5,
            )
        from_uid = aid if rid is None else int(rid)
        if (from_uid, bid) in used_second_hop:
            continue
        used_second_hop.add((from_uid, bid))
        arrow_color = PALETTE["direct"] if rid is None else PALETTE["via_relay"]
        ax.annotate(
            "",
            xy=bs[bid],
            xytext=uav[from_uid],
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.5,
                color=arrow_color,
                linestyle="-" if rid is None else "--",
                mutation_scale=17,
                shrinkA=12,
                shrinkB=12,
            ),
            zorder=5,
        )

    for i, pos in enumerate(uav):
        if status[i] != ACTIVE:
            color, marker = PALETTE["station"], "v"
        elif roles[i] == RELAY:
            color, marker = PALETTE["relay"], "D"
        else:
            color, marker = PALETTE["access"], "o"
        ax.scatter(
            [pos[0]],
            [pos[1]],
            s=205,
            c=color,
            marker=marker,
            edgecolor="white",
            lw=2.0,
            zorder=12,
        )
        ax.annotate(
            f"U{i}",
            pos,
            xytext=(10, -15 if i % 2 == 0 else 8),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
            color=PALETTE["text"],
            zorder=13,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.70),
        )

    # A short, explicit bilingual-free legend to avoid mistaking Relay radius
    # for the coverage radius of a surviving BS.
    legend = [
        Line2D(
            [],
            [],
            ls="",
            marker="s",
            color=PALETTE["bs"],
            markersize=9,
            label="BS đang hoạt động",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="X",
            color=PALETTE["failed"],
            markersize=10,
            label="BS bị hỏng",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=PALETTE["access"],
            markersize=9,
            label="UAV Access",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="D",
            color=PALETTE["relay"],
            markersize=9,
            label="UAV Relay",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="^",
            color=PALETTE["station"],
            markersize=9,
            label="Trạm sạc",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=PALETTE["unused"],
            markersize=5,
            label="UE chưa phục vụ",
        ),
        Patch(facecolor=PALETTE["affected"], alpha=0.36, label="Vùng ảnh hưởng BS hỏng"),
        Patch(
            facecolor=PALETTE["access"],
            edgecolor=PALETTE["access"],
            alpha=0.14,
            label="Vùng phủ UAV Access",
        ),
        Line2D(
            [],
            [],
            color=PALETTE["relay"],
            lw=2,
            ls="--",
            label="Phạm vi backhaul UAV Relay",
        ),
        Line2D([], [], color=PALETTE["direct"], lw=2.5, label="Access → BS (trực tiếp)"),
        Line2D([], [], color=PALETTE["via_relay"], lw=2.5, label="Access → Relay → BS"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=PALETTE["via_relay"],
            markersize=5,
            label="UE qua Relay",
        ),
    ]
    lg = ax.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.012),
        ncol=2,
        frameon=True,
        framealpha=0.97,
        fancybox=True,
        fontsize=8.5,
        handlelength=2.1,
        columnspacing=1.1,
        labelspacing=0.85,
        borderpad=0.90,
        edgecolor="#C9D3E2",
    )
    lg.get_frame().set_linewidth(1.2)
    lg.set_zorder(30)

    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.012)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return link_rows


def _json_numpy(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-seed", type=int, default=3101)
    parser.add_argument(
        "--step",
        type=int,
        default=61,
        help="Executed continuous step to capture, default 61",
    )
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        default=None,
        help="Optional previously captured .snapshot.json; skips simulator replay",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/thesis_continuous_eval/figures/uav_relay_network_map.png"),
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.step < 1 or args.dpi < 72:
        parser.error("step must be >= 1, dpi must be >= 72")
    if args.snapshot_json is not None:
        data = json.loads(args.snapshot_json.read_text(encoding="utf-8"))
    else:
        data = simulate_snapshot(args.scenario_seed, args.step)
    output = args.out if args.out.is_absolute() else ROOT / args.out
    links = draw_map(data, output, dpi=args.dpi)
    snapshot_path = output.with_suffix(".snapshot.json")
    snapshot_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_numpy),
        encoding="utf-8",
    )
    print(f"PNG:      {output}")
    print(f"SNAPSHOT: {snapshot_path}")
    print(
        f"Scenario {data['scenario_seed']} | step {data['step_global']} | "
        f"served {data['served_ues']} | via Relay {data['relay_served_ues']} | "
        f"route links {links}"
    )


if __name__ == "__main__":
    main()
