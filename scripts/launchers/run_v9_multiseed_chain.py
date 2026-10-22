from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def stage_paths(seed: int) -> dict[str, Path]:
    return {
        "v7_best": ROOT / "results" / "v7_r6_macro_relay_stability_pilot" / f"seed{seed}_best.pt",
        "v7_summary": ROOT
        / "results"
        / "v7_r6_macro_relay_stability_pilot"
        / f"seed{seed}_summary.json",
        "v8_best": ROOT
        / "results"
        / "v8_r6_macro_relay_battery_curriculum"
        / f"seed{seed}_best.pt",
        "v8_summary": ROOT
        / "results"
        / "v8_r6_macro_relay_battery_curriculum"
        / f"seed{seed}_summary.json",
        "v9_best": ROOT / "results" / "v9_r6_battery_transition_curriculum" / f"seed{seed}_best.pt",
        "v9_summary": ROOT
        / "results"
        / "v9_r6_battery_transition_curriculum"
        / f"seed{seed}_summary.json",
    }


def ensure_script(name: str) -> Path:
    p = ROOT / "scripts" / name
    if not p.exists():
        raise FileNotFoundError(f"Required script missing: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run independent V7 -> V8 -> V9 chains for multiple training seeds."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[92, 93])
    ap.add_argument("--v7-transitions", type=int, default=30000)
    ap.add_argument("--v8-transitions", type=int, default=30000)
    ap.add_argument("--v9-transitions", type=int, default=20000)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--force",
        action="store_true",
        help="rerun stages even when the expected summary+checkpoint already exist",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    v7_script = ensure_script("training/run_v7_r6_macro_relay_stability_pilot.py")
    v8_script = ensure_script("training/run_v8_r6_macro_relay_battery_curriculum.py")
    v9_script = ensure_script("training/run_v9_r6_battery_transition_curriculum.py")
    aggregate_script = ensure_script("analysis/aggregate_v9_multiseed.py")

    manifest = {
        "protocol": "independent V7->V8->V9 training chains",
        "seeds": [int(s) for s in args.seeds],
        "v7_transitions": int(args.v7_transitions),
        "v8_transitions": int(args.v8_transitions),
        "v9_transitions": int(args.v9_transitions),
        "reuse_existing_unless_force": not bool(args.force),
        "chains": [],
    }

    for seed in args.seeds:
        seed = int(seed)
        p = stage_paths(seed)
        row = {"seed": seed}
        print(f"\n===== TRAINING SEED {seed} =====", flush=True)

        v7_exists = p["v7_best"].exists() and p["v7_summary"].exists()
        if args.force or not v7_exists:
            run(
                [
                    sys.executable,
                    str(v7_script),
                    "--seed",
                    str(seed),
                    "--total-transitions",
                    str(args.v7_transitions),
                    "--torch-threads",
                    str(args.torch_threads),
                ],
                dry_run=args.dry_run,
            )
            row["v7"] = "planned" if args.dry_run else "trained"
        else:
            print(f"Reuse V7 seed{seed}: {p['v7_best']}", flush=True)
            row["v7"] = "reused"

        if not args.dry_run and not p["v7_best"].exists():
            raise FileNotFoundError(f"V7 checkpoint missing after stage: {p['v7_best']}")

        v8_exists = p["v8_best"].exists() and p["v8_summary"].exists()
        if args.force or not v8_exists:
            run(
                [
                    sys.executable,
                    str(v8_script),
                    "--seed",
                    str(seed),
                    "--total-transitions",
                    str(args.v8_transitions),
                    "--init-checkpoint",
                    str(p["v7_best"]),
                    "--torch-threads",
                    str(args.torch_threads),
                ],
                dry_run=args.dry_run,
            )
            row["v8"] = "planned" if args.dry_run else "trained"
        else:
            print(f"Reuse V8 seed{seed}: {p['v8_best']}", flush=True)
            row["v8"] = "reused"

        if not args.dry_run and not p["v8_best"].exists():
            raise FileNotFoundError(f"V8 checkpoint missing after stage: {p['v8_best']}")

        v9_exists = p["v9_best"].exists() and p["v9_summary"].exists()
        if args.force or not v9_exists:
            run(
                [
                    sys.executable,
                    str(v9_script),
                    "--seed",
                    str(seed),
                    "--total-transitions",
                    str(args.v9_transitions),
                    "--init-checkpoint",
                    str(p["v8_best"]),
                    "--torch-threads",
                    str(args.torch_threads),
                ],
                dry_run=args.dry_run,
            )
            row["v9"] = "planned" if args.dry_run else "trained"
        else:
            print(f"Reuse V9 seed{seed}: {p['v9_best']}", flush=True)
            row["v9"] = "reused"

        if not args.dry_run and not p["v9_summary"].exists():
            raise FileNotFoundError(f"V9 summary missing after stage: {p['v9_summary']}")
        manifest["chains"].append(row)

    manifest_path = (
        ROOT / "results" / "v9_r6_battery_transition_curriculum" / "multiseed_chain_manifest.json"
    )
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Include seed91 if already available; aggregate script only uses summaries that exist.
    run(
        [
            sys.executable,
            str(aggregate_script),
            "--seeds",
            "91",
            *[str(s) for s in args.seeds],
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
