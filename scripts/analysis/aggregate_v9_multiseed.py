from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(a)), "std": float(np.std(a))}


def get_phase_fracs(summary):
    return summary["heldout_test"]["dynamic"].get("relay_action_fraction_by_phase", {})


def row_from_summary(seed: int, d: dict, path: Path) -> dict:
    test = d["heldout_test"]
    cmp5 = test["comparison"]
    cmp8 = test["curriculum_vs_initial_v8"]
    fr = get_phase_fracs(d)
    gate = d.get("anticipatory_gate", {})
    return {
        "seed": int(seed),
        "best_validation": float(d["best_validation"]),
        "best_step": int(d["best_step"]),
        "test_dynamic_mean": float(test["dynamic"]["mean_utility"]),
        "test_5A_mean": float(test["movement_only_5A"]["mean_utility"]),
        "test_gain_vs_5A_pp": float(cmp5["dynamic_minus_5A_pp"]),
        "test_win_fraction_vs_5A": float(cmp5["win_fraction"]),
        "test_gain_vs_v8_pp": float(cmp8["finetuned_minus_initial_pp"]),
        "unavailable_gain_vs_v8_pp": float(cmp8.get("unavailable_gain_pp") or 0.0),
        "post_rejoin_gain_vs_v8_pp": float(cmp8.get("post_rejoin_gain_pp") or 0.0),
        "relay_fraction_pre_return_healthy": float(fr.get("pre_return_healthy", 0.0)),
        "relay_fraction_pre_return_near": float(fr.get("pre_return_near", 0.0)),
        "relay_fraction_unavailable": float(fr.get("unavailable", 0.0)),
        "relay_fraction_post_rejoin": float(fr.get("post_rejoin", 0.0)),
        "battery_lifecycle_exercised": bool(gate.get("battery_lifecycle_exercised", False)),
        "pass_v9_gate": bool(gate.get("pass_anticipatory_curriculum_gate", False)),
        "near_return_relay_lower_than_healthy": bool(
            gate.get("near_return_relay_lower_than_healthy", False)
        ),
        "summary_path": str(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[91, 92, 93])
    ap.add_argument(
        "--results-dir",
        default="results/v9_r6_battery_transition_curriculum",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    rows = []
    missing = []
    for seed in args.seeds:
        p = results_dir / f"seed{seed}_summary.json"
        if not p.exists():
            missing.append(int(seed))
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(row_from_summary(seed, d, p))

    out = {
        "protocol": "V9 R6 battery-transition curriculum multi-training-seed audit",
        "requested_seeds": [int(s) for s in args.seeds],
        "missing_seeds": missing,
        "rows": rows,
        "n_training_seeds": len(rows),
    }

    if rows:
        gains5 = [r["test_gain_vs_5A_pp"] for r in rows]
        gains8 = [r["test_gain_vs_v8_pp"] for r in rows]
        unavailable = [r["unavailable_gain_vs_v8_pp"] for r in rows]
        post = [r["post_rejoin_gain_vs_v8_pp"] for r in rows]
        wins = [r["test_win_fraction_vs_5A"] for r in rows]
        dyn = [r["test_dynamic_mean"] for r in rows]
        out["across_training_seeds"] = {
            "test_dynamic_mean": mean_std(dyn),
            "test_gain_vs_5A_pp": mean_std(gains5),
            "test_gain_vs_v8_pp": mean_std(gains8),
            "unavailable_gain_vs_v8_pp": mean_std(unavailable),
            "post_rejoin_gain_vs_v8_pp": mean_std(post),
            "test_win_fraction_vs_5A": mean_std(wins),
            "positive_gain_vs_5A_seed_fraction": float(np.mean(np.asarray(gains5) > 0.0)),
            "positive_gain_vs_v8_seed_fraction": float(np.mean(np.asarray(gains8) > 0.0)),
            "positive_unavailable_gain_seed_fraction": float(
                np.mean(np.asarray(unavailable) > 0.0)
            ),
            "battery_lifecycle_seed_fraction": float(
                np.mean([r["battery_lifecycle_exercised"] for r in rows])
            ),
            "individual_v9_gate_pass_fraction": float(np.mean([r["pass_v9_gate"] for r in rows])),
            "near_return_suppression_seed_fraction": float(
                np.mean([r["near_return_relay_lower_than_healthy"] for r in rows])
            ),
        }
        a = out["across_training_seeds"]
        out["multiseed_gate"] = {
            "all_three_seeds_present": len(rows) >= 3 and not missing,
            "positive_mean_gain_vs_5A": a["test_gain_vs_5A_pp"]["mean"] > 0.0,
            "majority_training_seeds_positive_vs_5A": a["positive_gain_vs_5A_seed_fraction"]
            >= 2 / 3,
            "positive_mean_gain_vs_v8": a["test_gain_vs_v8_pp"]["mean"] > 0.0,
            "majority_training_seeds_positive_vs_v8": a["positive_gain_vs_v8_seed_fraction"]
            >= 2 / 3,
            "battery_lifecycle_all_seeds": a["battery_lifecycle_seed_fraction"] == 1.0,
        }
        g = out["multiseed_gate"]
        g["pass_multiseed_gate"] = bool(
            g["all_three_seeds_present"]
            and g["positive_mean_gain_vs_5A"]
            and g["majority_training_seeds_positive_vs_5A"]
            and g["positive_mean_gain_vs_v8"]
            and g["majority_training_seeds_positive_vs_v8"]
            and g["battery_lifecycle_all_seeds"]
        )

    out_path = Path(args.out) if args.out else results_dir / "aggregate_all_available_seeds.json"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWROTE={out_path}")


if __name__ == "__main__":
    main()
