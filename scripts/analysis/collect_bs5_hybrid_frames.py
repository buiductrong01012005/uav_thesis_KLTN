"""Replay supplied legacy action tapes to obtain checked Hybrid development frames."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.envs.bs5_tcf_env import BS5TCFEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frames, checks = [], []
    for path in args.reference:
        protocol = json.loads((path.parent.parent / "protocol.json").read_text())
        for name, digest in protocol["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("Source mismatch: " + name)
        seed = int(path.stem.split("_seed")[1].split("_")[0])
        with np.load(path, allow_pickle=False) as data:
            saved = {k: data[k] for k in data.files}
        env = BS5TCFEnv(horizon=len(saved["actions"]))
        env.reset(seed=seed, assignment_seed=seed)
        for slot, actions in enumerate(saved["actions"]):
            env.step(actions.tolist())
            snap = env.last_service_snapshot
            for a, b in [("uav_positions", "positions"), ("roles", "roles"), ("status", "status")]:
                np.testing.assert_array_equal(snap[a], saved[b][slot])
            np.testing.assert_allclose(snap["battery"], saved["battery"][slot], atol=1e-5, rtol=0)
            np.testing.assert_allclose(
                env.last_v2_network["ue_rate_mbps"], saved["ue_rates"][slot], atol=1e-7, rtol=0
            )
            if slot % args.stride == 0:
                frames.append(
                    dict(
                        seed=seed,
                        slot=slot + 1,
                        origin=protocol["arm"],
                        positions=snap["uav_positions"].copy(),
                        ue_positions=snap["ue_positions"].copy(),
                        roles=snap["roles"].copy(),
                        status=snap["status"].copy(),
                        battery=snap["battery"].copy(),
                    )
                )
        checks.append(
            dict(
                file=str(path),
                sha256=sha(path),
                arm=protocol["arm"],
                seed=seed,
                slots=len(saved["actions"]),
                verified_source_files=len(protocol["source_sha256"]),
            )
        )
        print(f"verified {protocol['arm']} seed{seed}: {len(saved['actions'])} slots", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k: np.asarray([f[k] for f in frames]) for k in frames[0]})
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                frames=len(frames),
                stride=args.stride,
                inputs=checks,
                collector_sha256=sha(__file__),
                output_sha256=sha(args.output),
            ),
            indent=2,
        )
        + "\n"
    )
    print("saved", len(frames), "verified frames", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=100)
    args = p.parse_args()
    if args.stride < 1:
        p.error("Positive stride required")
    collect(args)
