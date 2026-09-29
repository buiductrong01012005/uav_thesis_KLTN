from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    outdir = ROOT / "results" / "final_movement_only_spatial_qmix"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        summary = outdir / f"seed{seed}_summary.json"
        if args.force or not summary.exists():
            cmd = [sys.executable, str(ROOT / "scripts" / "run_final_movement_only_qmix.py"), "--seed", str(seed)]
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        data = json.loads(summary.read_text(encoding="utf-8"))
        rows.append({
            "seed": int(seed),
            "test_mean": float(data["heldout_test"]["mean_utility"]),
            "test_std": float(data["heldout_test"]["std_utility"]),
            "decision_ms_per_slot": float(data["heldout_test"]["decision_ms_per_slot"]),
            "checkpoint": data["final_checkpoint"],
        })
    import numpy as np
    agg = {
        "protocol": "independent 5-action movement-only Spatial-QMIX multi-training-seed baseline",
        "rows": rows,
        "across_training_seeds": {
            "test_mean": float(np.mean([r["test_mean"] for r in rows])),
            "test_mean_std_across_training_seeds": float(np.std([r["test_mean"] for r in rows])),
            "decision_ms_per_slot_mean": float(np.mean([r["decision_ms_per_slot"] for r in rows])),
        },
    }
    (outdir / "aggregate_all_available_seeds.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
