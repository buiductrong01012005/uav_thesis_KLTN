"""Development fixtures/lifecycle checks only; no learned-policy benchmark."""

from pathlib import Path
import argparse, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.envs.bs5_multihop_train import BS5MultihopTrainEnv
from src.envs.v0_env import MOVE_DELTAS
from src.diagnostics.bs5_multihop import score_multihop
from src.diagnostics.bs5_hybrid_full import score_hybrid_full
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics


def check(output, slots=1200):
    e = BS5MultihopTrainEnv(arm="H-only")
    e.reset(seed=730001, assignment_seed=730001)
    e.surviving_bs_xy = np.asarray([e._cell_to_xy([0, 0])])
    e.uav_positions = np.asarray([[0, 2], [0, 8], [0, 14], [6, 14], [12, 14]])
    e.status[:] = 0
    e.ue_positions[:] = [12, 14]
    full = score_multihop(e, [2] * 5)
    old = score_hybrid_full(e, [2] * 5)
    fixture = dict(
        wireless_hops=len(full["paths"][4]),
        multihop_mbps=full["metrics"]["throughput_mbps"],
        old_one_relay_mbps=old["metrics"]["throughput_mbps"],
        flow_violation_mbps=full["metrics"]["flow_violation_mbps"],
    )
    assert (
        fixture["wireless_hops"] == 5
        and fixture["multihop_mbps"] > 0
        and fixture["old_one_relay_mbps"] == 0
    )
    target = np.asarray([[13, 10], [13, 13], [15, 14], [16, 16], [17, 17]])
    lifecycle = []
    for seed in [730001, 730002]:
        e = BS5MultihopTrainEnv(arm="H-only", horizon=slots)
        x = e.reset(seed=seed, assignment_seed=seed)
        start = float(e.battery.sum())
        used = charged = 0.0
        rejoins = 0
        rates = []
        maxhop = 0
        maxviolation = 0
        stranded = 0
        for _ in range(slots):
            before = e.status.copy()
            acts = []
            for a, mask in enumerate(x["action_mask"]):
                legal = np.flatnonzero(mask)
                acts.append(
                    int(
                        min(
                            legal,
                            key=lambda move: (
                                np.abs(e.uav_positions[a] + MOVE_DELTAS[move] - target[a]).sum(),
                                int(move),
                            ),
                        )
                    )
                )
            x = e.step(acts)
            used += e.last_energy_probe["depleted_points"]
            charged += e.last_energy_probe["charged_points"]
            rejoins += int(np.sum((before == 2) & (e.status == 0)))
            rates.append(e.last_v2_network["ue_rate_mbps"].copy())
            m = e.last_probe_result["metrics"]
            maxhop = max(maxhop, m["max_wireless_hops_used"])
            maxviolation = max(maxviolation, m["flow_violation_mbps"])
            stranded += int(bool(e.last_return_control["stranded_ids_post"]))
        balance = start + charged - used - float(e.battery.sum())
        assert abs(balance) < 1e-3 and maxviolation <= 1e-6
        assert rejoins > 0, "Lifecycle must exercise charging"
        lifecycle.append(
            dict(
                seed=seed,
                slots=slots,
                controller="Fixed development target positions; movement only; not learned",
                throughput_mbps=float(np.asarray(rates).sum(axis=1).mean()),
                charge_rejoins=rejoins,
                battery_balance_error=balance,
                max_wireless_hops_used=maxhop,
                max_flow_violation_mbps=maxviolation,
                stranded_slots=stranded,
                **episode_service_metrics(rates),
            )
        )
        print("H-only development lifecycle", seed, "passed", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                scope="Development fixture/controller only, no training or independent policy benchmark",
                fixture=fixture,
                lifecycle=lifecycle,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slots", type=int, default=1200)
    a = p.parse_args()
    check(a.output, a.slots)
