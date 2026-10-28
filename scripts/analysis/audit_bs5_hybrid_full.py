"""Full-resource H versus A/R and prior50/50, on identical development frames."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.diagnostics.bs5_hybrid_full import HybridFullConfig
from src.diagnostics.bs5_hybrid_fdm import score_hybrid_fdm
from src.envs.bs5_tcf_env import BS5TCFEnv, TCFWeights


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.corpus, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    env = BS5TCFEnv(reward_weights=TCFWeights(0.45, 0.45, 0.10))
    env.reset(seed=620001, assignment_seed=620001)
    winners, all_rows = [], []
    started = time.perf_counter()
    for i in range(len(arrays["seed"])):
        for target, key in [
            ("uav_positions", "positions"),
            ("ue_positions", "ue_positions"),
            ("roles", "roles"),
            ("status", "status"),
            ("battery", "battery"),
        ]:
            setattr(env, target, arrays[key][i].copy())
        active = np.flatnonzero(env.status == 0)
        best = {
            (arm, criterion): None
            for arm in ["A-only", "AR", "ARH_full"]
            for criterion in ["tcf_score", "throughput_mbps"]
        }
        for combo in itertools.product([0, 1, 2], repeat=len(active)):
            roles = np.zeros(5, dtype=int)
            roles[active] = combo
            m = score_hybrid_fdm(env, roles, HybridFullConfig(), env.reward_weights)["metrics"]
            row = dict(
                frame=i,
                seed=int(arrays["seed"][i]),
                slot=int(arrays["slot"][i]),
                origin=str(arrays["origin"][i]),
                roles="".join(map(str, roles)),
                **m,
            )
            all_rows.append(row)
            legal = (
                ["ARH_full"]
                + (["AR"] if 2 not in combo else [])
                + (["A-only"] if all(r == 0 for r in combo) else [])
            )
            for arm, criterion in best:
                prev = best[arm, criterion]
                if arm not in legal:
                    continue
                if (
                    prev is None
                    or row[criterion] > prev[criterion] + 1e-10
                    or (
                        abs(row[criterion] - prev[criterion]) <= 1e-10
                        and (row["roles"].count("2"), row["roles"])
                        < (prev["roles"].count("2"), prev["roles"])
                    )
                ):
                    best[arm, criterion] = row
        for arm, criterion in best:
            winners.append(dict(arm=arm, criterion=criterion, **best[arm, criterion]))
        if i % 12 == 11:
            print(f"full-role search {i + 1}/{len(arrays['seed'])}", flush=True)
    reference = list(csv.DictReader(args.split_reference.open()))
    for row in winners:
        if row["arm"] != "AR":
            continue
        old = next(
            r
            for r in reference
            if int(r["frame"]) == row["frame"]
            and r["arm"] == "C-R"
            and r["criterion"] == row["criterion"]
        )
        assert abs(float(old["throughput_mbps"]) - row["throughput_mbps"]) < 1e-7
    aggregate = []
    for arm, criterion in itertools.product(
        ["A-only", "AR", "ARH_full"], ["tcf_score", "throughput_mbps"]
    ):
        rows = [r for r in winners if r["arm"] == arm and r["criterion"] == criterion]
        keys = [k for k in rows[0] if k in m]
        aggregate.append(
            dict(
                arm=arm,
                criterion=criterion,
                frames=len(rows),
                **{k: float(np.mean([r[k] for r in rows])) for k in keys},
            )
        )
    write_csv(args.output / "all_roles.csv", all_rows)
    write_csv(args.output / "selected_roles.csv", winners)
    write_csv(args.output / "aggregate.csv", aggregate)
    receipt = dict(
        version="bs5_hybrid_full_static_1",
        frames=len(arrays["seed"]),
        scores=len(all_rows),
        full_selected_frames=sum(
            r["active_hybrid"] > 0
            for r in winners
            if r["arm"] == "ARH_full" and r["criterion"] == "tcf_score"
        ),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/diagnostics/bs5_hybrid_full.py",
                ROOT / "src/diagnostics/bs5_hybrid_fdm.py",
            ]
        },
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        split_reference_sha256=hashlib.sha256(args.split_reference.read_bytes()).hexdigest(),
        wall_seconds=time.perf_counter() - started,
        scope="Static role enumeration only; full H has extra simultaneous RF capability, energy omitted here",
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--split-reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
