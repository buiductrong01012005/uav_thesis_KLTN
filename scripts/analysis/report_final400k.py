"""Connect the reference raw auditor to the CSV names used by the thesis plots.

Keep imported final results separate. The reference crossed bootstrap is reused;
additional tables contain arithmetic means of the same verified runs/slots.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from scripts.analysis.fourarm400k_multiseed import sources
from scripts.analysis.report_fourarm400k_multiseed import report

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_METRICS = ["throughput_mbps", "qos_ues", "service_coverage_ues", "energy_consumed_wh",
                   "jain_time_average", "rolling_unique_ge1_5min", "actor_ms", "reward_total"]
BLOCK_METRICS = ["throughput_mbps", "qos_ues", "service_coverage_ues", "consumed_wh",
                 "reward_total", "effective_return_fraction", "effective_swap_fraction"]


def temporal_table(input_dir, runs):
    expected = {(row.configuration, None if pd.isna(row.training_seed) else int(row.training_seed),
                 int(row.seed)) for row in runs.itertuples()}
    found = {}
    fingerprints = {}
    for protocol, read, _ in sources(input_dir):
        if protocol.get("version") not in ("fourarm400k_drl_test_v1", "fourarm400k_heuristic_benchmark_v1"):
            continue
        train = protocol.get("training_seed")
        label = protocol["arm"] if train is not None else protocol["label"]
        for seed in protocol["target_seeds"]:
            key = (label, train, seed)
            if key not in expected:
                continue
            prefix = f"seed{seed}/"
            try:
                commit = json.loads(read(prefix + "COMMIT.json"))
            except (KeyError, FileNotFoundError):
                continue
            blob = read(prefix + "slots.csv")
            sha = hashlib.sha256(blob).hexdigest()
            if sha != commit["files_sha256"]["slots.csv"]:
                raise ValueError("Slot CSV changed since the raw audit")
            if key in fingerprints:
                if fingerprints[key] != sha:
                    raise ValueError("Conflicting duplicate slot CSV")
                continue
            fingerprints[key] = sha
            frame = pd.read_csv(io.BytesIO(blob))
            if len(frame) != 10000 or frame.slot.tolist() != list(range(1, 10001)):
                raise ValueError("Expected 10,000 sequential slots")
            frame["block"] = (frame.slot - 1) // 1000 + 1
            blocks = frame.groupby("block", as_index=False)[BLOCK_METRICS].mean()
            blocks["configuration"] = label
            blocks["training_seed"] = train
            blocks["scenario_seed"] = seed
            found[key] = blocks
    if set(found) != expected:
        raise ValueError("Missing slot CSVs for verified runs")
    return pd.concat([found[key] for key in sorted(found, key=str)], ignore_index=True)


def rebuild(input_dir, output):
    output = Path(output)
    if output.resolve() in {(ROOT / name).resolve() for name in
                            ("results/final400k/benchmark", "results/w00_main/benchmark")}:
        raise ValueError("Choose a new output directory; preserve the delivered final CSVs")
    metadata = report(input_dir, output, include_random=True)
    products = ["per_test_seed.csv", "per_training_seed.csv", "temporal_1000slot.csv", "audit.json"]
    if metadata["status"] != "complete":
        for name in products:
            (output / name).unlink(missing_ok=True)
        return metadata
    runs = pd.read_csv(output / "per_run.csv")
    runs.groupby(["configuration", "seed"], as_index=False)[SUMMARY_METRICS].mean().to_csv(
        output / "per_test_seed.csv", index=False)
    runs.groupby(["configuration", "training_seed"], as_index=False)[SUMMARY_METRICS].mean().to_csv(
        output / "per_training_seed.csv", index=False)
    temporal_table(input_dir, runs).to_csv(output / "temporal_1000slot.csv", index=False)
    audit = dict(metadata, slots=int(runs.slots.sum()), paired_ue_trace=True,
                 reconstruction="Reference raw auditor and crossed bootstrap; fresh local audit, not the historical staged audit")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/w00_main/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/local/w00_raw_rebuilt")
    args = parser.parse_args()
    metadata = rebuild(args.input, args.output)
    print(json.dumps({key: metadata[key] for key in
                      ("status", "expected_runs", "completed_runs", "missing")}, indent=2))


if __name__ == "__main__":
    main()
