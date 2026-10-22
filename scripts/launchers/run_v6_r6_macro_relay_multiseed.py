from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_one(seed: int, total_transitions: int, root_out: Path) -> Path:
    outdir = root_out / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "training" / "run_v6_r6_macro_relay_pilot.py"),
        "--seed",
        str(seed),
        "--total-transitions",
        str(total_transitions),
        "--outdir",
        str(outdir),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary = outdir / f"seed{seed}_summary.json"
    if not summary.exists():
        raise FileNotFoundError(summary)
    return summary


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[92, 93])
    ap.add_argument("--total-transitions", type=int, default=30000)
    ap.add_argument("--outdir", default="results/v6_r6_macro_relay_multiseed")
    ap.add_argument(
        "--seed91-summary",
        default="results/v6_r6_macro_relay_pilot/seed91_summary.json",
        help="Existing seed91 summary to include in the 3-seed aggregate if present.",
    )
    args = ap.parse_args()

    root_out = Path(args.outdir)
    if not root_out.is_absolute():
        root_out = ROOT / root_out
    root_out.mkdir(parents=True, exist_ok=True)

    summary_paths: list[Path] = []

    seed91 = Path(args.seed91_summary)
    if not seed91.is_absolute():
        seed91 = ROOT / seed91
    if seed91.exists():
        summary_paths.append(seed91)
        print(f"INCLUDE_EXISTING {seed91}", flush=True)
    else:
        print(f"WARNING seed91 summary not found: {seed91}", flush=True)

    for seed in args.seeds:
        summary_paths.append(run_one(seed, args.total_transitions, root_out))

    rows = []
    for path in summary_paths:
        s = load_summary(path)
        tc = s["heldout_test"]["comparison"]
        rows.append(
            {
                "seed": int(s["seed"]),
                "best_validation": float(s["best_validation"]),
                "best_step": int(s["best_step"]),
                "test_dynamic_mean": float(s["heldout_test"]["dynamic"]["mean_utility"]),
                "test_5A_mean": float(s["heldout_test"]["movement_only_5A"]["mean_utility"]),
                "test_gain_pp": float(tc["dynamic_minus_5A_pp"]),
                "test_win_fraction": float(tc["win_fraction"]),
                "mean_relays": float(s["heldout_test"]["dynamic"]["mean_relays"]),
                "pass_core_gate": bool(s["pilot_gate"]["pass_core_gate"]),
                "summary_path": str(path),
            }
        )

    rows.sort(key=lambda r: r["seed"])

    def vals(k):
        return [float(r[k]) for r in rows]

    aggregate = {
        "protocol": "V6 R6 macro-relay multi-training-seed final core audit",
        "battery": False,
        "rows": rows,
        "n_training_seeds": len(rows),
        "across_training_seeds": {
            "test_dynamic_mean": {
                "mean": statistics.mean(vals("test_dynamic_mean")),
                "std": statistics.pstdev(vals("test_dynamic_mean")) if len(rows) > 1 else 0.0,
            },
            "test_5A_mean": {
                "mean": statistics.mean(vals("test_5A_mean")),
                "std": statistics.pstdev(vals("test_5A_mean")) if len(rows) > 1 else 0.0,
            },
            "test_gain_pp": {
                "mean": statistics.mean(vals("test_gain_pp")),
                "std": statistics.pstdev(vals("test_gain_pp")) if len(rows) > 1 else 0.0,
            },
            "test_win_fraction": {
                "mean": statistics.mean(vals("test_win_fraction")),
                "std": statistics.pstdev(vals("test_win_fraction")) if len(rows) > 1 else 0.0,
            },
            "mean_relays": {
                "mean": statistics.mean(vals("mean_relays")),
                "std": statistics.pstdev(vals("mean_relays")) if len(rows) > 1 else 0.0,
            },
            "all_seeds_pass_core_gate": all(r["pass_core_gate"] for r in rows),
            "positive_gain_seed_fraction": sum(r["test_gain_pp"] > 0 for r in rows) / len(rows),
        },
        "next_stage": "If multi-seed core is stable, enable battery lifecycle using the same 9-action macro-relay formulation; battery is an extension, not removed.",
    }

    out = root_out / "aggregate_all_available_seeds.json"
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("DONE", json.dumps(aggregate["across_training_seeds"]), flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
