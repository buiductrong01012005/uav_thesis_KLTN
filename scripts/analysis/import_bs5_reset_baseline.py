"""Create compact, hash-checked historical references; never rewrite old evidence."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile


def import_baseline(source, destination):
    destination = Path(destination)
    assert not destination.exists(), "Keep the frozen baseline; choose a new destination"
    references = {}
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("SHA256SUMS.json"))

        def read(name):
            data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            assert digest == manifest[name], f"baseline hash mismatch: {name}"
            references[name] = dict(sha256=digest, bytes=len(data))
            return data

        protocol = json.loads(read("protocol.json"))
        identity = json.loads(read("source_identity.json"))
        registry = json.loads(read("run_registry.json"))
        assert registry["status"] == "complete"
        assert protocol["training_seeds"] == [551, 552]
        assert protocol["steps_per_run"] == 24000
        summary = json.loads(read("analysis/summary.json"))
        metas = {}
        for seed in protocol["training_seeds"]:
            for arm in ("C-R", "D-R"):
                name = f"{arm}_seed{seed}"
                metas[name] = json.loads(read(f"runs/{name}/meta.json"))
                assert metas[name]["source_sha256"] == protocol["source_sha256"]
            assert (
                metas[f"C-R_seed{seed}"]["initial_weights_sha256"]
                == metas[f"D-R_seed{seed}"]["initial_weights_sha256"]
            )
        text = read("analysis/validation_episodes.csv").decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = [
            r
            for r in reader
            if r["arm"] in {"C-R", "D-R"} and int(r["step"]) in protocol["common_evaluation_steps"]
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Exogenous traces are independent of policy; retain all slots of step0,
        # not the large old geometry/shadow packets or checkpoints.
        exogenous, residual = {}, {}
        for scenario in protocol["validation_pool"]:
            packet = json.loads(
                gzip.decompress(read(f"runs/D-R_seed551/evaluations/000000/{scenario}.json.gz"))
            )
            exogenous[str(scenario)] = [r["exogenous_sha256"] for r in packet["trace"]]
            residual[str(scenario)] = packet["initial_residual"]
        training = {}
        names = set(archive.namelist())
        for seed in protocol["training_seeds"]:
            episodes = []
            for eid, spec in enumerate(protocol["scenario_schedule"][str(seed)]):
                prefix = f"runs/D-R_seed{seed}"
                ep = json.loads(read(f"{prefix}/episodes/{eid:06d}.json"))
                assert (
                    ep["scenario_seed"] == spec["scenario_seed"]
                    and ep["assignment_seed"] == spec["assignment_seed"]
                )
                reference = dict(
                    residual=ep["residual"],
                    scenario_seed=ep["scenario_seed"],
                    assignment_seed=ep["assignment_seed"],
                )
                trace_name = f"{prefix}/train_traces/{eid:06d}.json.gz"
                if trace_name in names:
                    packet = json.loads(gzip.decompress(read(trace_name)))
                    reference["exogenous"] = [r["exogenous_sha256"] for r in packet["trace"]]
                episodes.append(reference)
            training[str(seed)] = episodes
        result = dict(
            version="bs5_reset_baseline_refs_v2",
            original_archive=Path(source).name,
            verification_scope="SHA-checked original protocol/meta/cached evaluation tables and step0 raw exogenous/residual references; baseline performance tables are not re-audited from all historical raw packets by this importer",
            original_members=references,
            protocol=protocol,
            source_identity=identity,
            metas=metas,
            runs=[r for r in summary["runs"] if r["arm"] in {"C-R", "D-R"}],
            validation_csv=stream.getvalue(),
            evaluation_exogenous=exogenous,
            evaluation_residual=residual,
            training_references=training,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            dict(baseline=str(destination), bytes=destination.stat().st_size, episodes=len(rows))
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_baseline(args.bundle, args.output)
