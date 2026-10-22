from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def stats(xs):
    xs = list(map(float, xs))
    return {
        "mean": mean(xs),
        "std": stdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
        "values": xs,
    }


def main():
    a2 = json.loads((RESULTS / "final_a2_staged_audit.json").read_text())["A2_mobile_no_battery"]
    a3 = json.loads((RESULTS / "final_a3_staged_audit.json").read_text())["A3_mobile_plus_battery"]

    a2s = a2["seeds"]
    a3s = a3["seeds"]
    out = {
        "status": "A2_A3_FINALIZED_NO_FURTHER_TUNING",
        "a2_mobile_no_battery": {
            "kind": "staged necessity audit; not learned-policy final performance",
            "seeds": [s["seed"] for s in a2s],
            "adaptive_utility": stats(s["movement"]["adaptive"] for s in a2s),
            "frozen_utility": stats(s["movement"]["frozen"] for s in a2s),
            "movement_gain_pp": stats(s["movement"]["gain_pp"] for s in a2s),
            "relay_count_changes": stats(s["role_snapshots"]["relay_count_changes"] for s in a2s),
            "interpretation": "Mobility robustly motivates continued UAV repositioning. It does not independently motivate temporal Relay-count changes in these trajectories.",
        },
        "a3_mobile_plus_battery": {
            "kind": "combined staged robustness audit; not learned-policy final performance",
            "seeds": [s["seed"] for s in a3s],
            "adaptive_utility": stats(s["mobility"]["adaptive"] for s in a3s),
            "frozen_utility": stats(s["mobility"]["frozen"] for s in a3s),
            "movement_gain_pp": stats(s["mobility"]["gain_pp"] for s in a3s),
            "rerole_gain_pp": stats(s["battery_role"]["rerole_gain_pp"] for s in a3s),
            "pre_return_utility": stats(s["battery_role"]["pre_return"] for s in a3s),
            "fixed_unavailable_utility": stats(s["battery_role"]["fixed_unavailable"] for s in a3s),
            "dynamic_unavailable_utility": stats(
                s["battery_role"]["dynamic_unavailable"] for s in a3s
            ),
            "dynamic_post_rejoin_utility": stats(
                s["battery_role"]["dynamic_post_rejoin"] for s in a3s
            ),
            "interpretation": "Mobility remains useful under battery dynamics, but extra dynamic re-role during Relay unavailability is not robustly beneficial in the locked BS3 topology.",
        },
        "claim_boundary": {
            "supported": [
                "adaptive repositioning under mobile UE demand",
                "automatic Return->Charging->Rejoin lifecycle is operational",
            ],
            "not_supported": [
                "battery-specific temporal role reassignment learned by MARL",
                "dynamic role switching improves A3 on average",
            ],
        },
    }
    (RESULTS / "final_stage_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
