"""Check the delivered final tables, optional frozen models and original raw runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["C-N", "C-R", "C-H", "H-only"]
LABELS = ARMS + ["DYNAMIC_AR", "HEURISTIC_HO", "RANDOM"]
SCENARIOS = set(range(930201, 930211))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_tables(folder):
    frames = {name: pd.read_csv(folder / (name + ".csv")) for name in
              ["per_run", "aggregate", "per_test_seed", "per_training_seed", "temporal_1000slot"]}
    runs = frames["per_run"]
    audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    require(audit["status"] == "complete" and audit["expected_runs"] ==
            audit["completed_runs"] == len(runs) == 150, "Expected 150 complete runs")
    require(set(runs.configuration) == set(LABELS), "Expected seven configurations")
    require(runs.complete.all() and not runs.diagnostic.any(), "Incomplete or diagnostic run")
    require((runs.slots == 10000).all() and (runs.target_slots == 10000).all(), "Wrong horizon")
    require(int(runs.slots.sum()) == audit["slots"] == 1500000, "Wrong slot total")
    require(not runs.duplicated(["configuration", "training_seed", "seed"]).any(), "Duplicate run")
    for label in LABELS:
        frame = runs[runs.configuration == label]
        require(set(frame.seed) == SCENARIOS, "Wrong scenario seeds: " + label)
        if label in ARMS:
            require(set(frame.training_seed) == {551, 552, 553}, "Wrong train seeds: " + label)
            require(len(frame) == 30 and (frame.groupby("training_seed").size() == 10).all(),
                    "Incomplete crossed design: " + label)
            require((frame.groupby("training_seed").selected_step.nunique() == 1).all(),
                    "Checkpoint changed across scenarios")
        else:
            require(len(frame) == 10 and frame.training_seed.isna().all(), "Baseline copies: " + label)
    require((runs.groupby("seed").ue_trace_sha256.nunique() == 1).all(), "UE traces are not paired")
    aggregate = frames["aggregate"]
    require(len(aggregate) == 7 and set(aggregate.configuration) == set(LABELS), "Aggregate labels")
    metrics = ["throughput_mbps", "qos_ues", "service_coverage_ues", "energy_consumed_wh", "actor_ms"]
    for _, row in aggregate.iterrows():
        subset = runs[runs.configuration == row.configuration]
        require(row.actual_evaluation_runs == len(subset) and row.scenario_seeds == 10,
                "Aggregate run counts")
        require(row.training_replicates == (3 if row.configuration in ARMS else 0), "Replicate count")
        for metric in metrics:
            require(np.isclose(row[metric], subset[metric].mean(), rtol=1e-9, atol=1e-7),
                    "Aggregate mean mismatch: " + metric)
        pooled = subset.energy_consumed_wh.sum() * 3600 / subset.delivered_mbit.sum()
        require(np.isclose(row.joules_per_mbit_pooled, pooled), "Pooled energy mismatch")
    for name, keys, expected in [("per_test_seed", ["configuration", "seed"], 70),
                                 ("per_training_seed", ["configuration", "training_seed"], 12)]:
        table = frames[name]
        require(len(table) == expected and not table.duplicated(keys).any(), "Invalid " + name)
        actual = table.set_index(keys)[metrics].sort_index()
        wanted = runs.groupby(keys)[metrics].mean().sort_index()
        require(actual.index.equals(wanted.index) and np.allclose(actual, wanted, rtol=1e-9, atol=1e-7),
                "Run/summary mismatch: " + name)
    blocks = frames["temporal_1000slot"]
    keys = ["configuration", "training_seed", "scenario_seed"]
    require(len(blocks) == 1500 and not blocks.duplicated(keys + ["block"]).any(), "Invalid blocks")
    require((blocks.groupby(keys, dropna=False).block.apply(lambda s: set(s) == set(range(1, 11)))).all(),
            "Missing temporal block")
    # Pandas groupby and set_index encode missing MultiIndex levels differently.
    # Use a sentinel only in these comparison views; preserve the delivered CSVs.
    block_view = blocks.assign(training_seed=blocks.training_seed.fillna(-1))
    run_view = runs.assign(training_seed=runs.training_seed.fillna(-1))
    block_means = block_view.groupby(keys).throughput_mbps.mean().sort_index()
    run_means = run_view.rename(columns={"seed": "scenario_seed"}).set_index(keys).throughput_mbps.sort_index()
    require(block_means.index.equals(run_means.index) and np.allclose(block_means, run_means),
            "Temporal/run throughput mismatch")
    return {"runs": 150, "configurations": 7, "scenario_seeds": sorted(SCENARIOS),
            "slots": 1500000, "paired_ue_trace": True}, runs


def check_models(folder):
    import torch
    from scripts.evaluation.run_fourarm400k_drl_benchmark import read_model, FrozenActor
    from src.rl.bs5_fourarm400k_qmix import make_learner
    import io

    records = []
    for arm in ARMS:
        for seed in (551, 552, 553):
            record, protocol, payload, _ = read_model(folder, arm, seed)
            actor = FrozenActor(payload, protocol, record)
            actions = actor.act(np.zeros((5, 173)), np.ones((5, actor.n_actions), dtype=bool))
            require(len(actions) == 5, "Actor output shape")
            saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            learner = make_learner(arm, device="cpu")
            learner.mixer.load_state_dict(saved["mixer"], strict=True)
            records.append({"arm": arm, "training_seed": seed, "selected_step": record["selected_step"],
                            "checkpoint_sha256": record["checkpoint_sha256"]})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/final400k/benchmark")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/final400k/verification.json")
    args = parser.parse_args()
    result, runs = check_tables(args.input)
    result["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(args.input.iterdir()) if p.suffix in (".csv", ".json")}
    if args.models:
        result["models"] = check_models(args.models)
    if args.raw:
        from scripts.analysis.report_fourarm400k_multiseed import scan, expected_keys
        rows, _, origins = scan(args.raw, include_random=True)
        result["raw_verified_runs"] = len(rows)
        missing = expected_keys(LABELS, SCENARIOS) - set(rows)
        result["missing_raw_runs"] = sorted([list(k) for k in missing], key=str)
        result["raw_origins"] = origins
        for (label, train, seed), row in rows.items():
            selected = runs[(runs.configuration == label) & (runs.seed == seed) &
                            (runs.training_seed.isna() if train is None else runs.training_seed == train)]
            require(len(selected) == 1, "Raw run absent from final table")
            for metric in ["throughput_mbps", "qos_ues", "energy_consumed_wh", "reward_total",
                           "actor_ms", "mean_return_actual_m", "mean_return_duration_slots"]:
                expected = selected.iloc[0][metric]
                actual = row.get(metric)
                require((actual is None and pd.isna(expected)) or
                        (actual is not None and np.isclose(actual, expected, rtol=1e-9, atol=1e-7)),
                        f"Raw/final mismatch: {label}/{train}/{seed}/{metric}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("files_sha256", "models", "raw_origins")}, indent=2))


if __name__ == "__main__":
    main()
