"""One Kaggle source dataset: NEW four-run training, frozen best eval600, delivery.

No historical weights/audit are required. Training is delegated unchanged to the
existing fresh60k runner. Historical eval600 remains a separate, strict mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.analyze_bs5_curriculum_extend import verify_manifest
from scripts.evaluation import eval_bs5_continuous as continuous
from scripts.training.run_bs5_curriculum_fresh_60k import budget, prepare_fresh, run_fresh
from scripts.training.run_bs5_curriculum_extend import finalize
from scripts.training.run_bs5_launch_ab import experiment_lock

SPEC = ROOT / "configs/bs5_retrain_eval600_v1.json"


def prepare(output, *, device="cuda"):
    output = Path(output).resolve()
    assert not output.exists(), "keep existing experiment; run it rather than prepare again"
    spec = read_json(SPEC)
    assert spec["new_training_steps"] == 240000 and spec["eval_total_cap"] == 25200
    assert spec["arms"] == ["C-R", "C-N"] and spec["training_seeds"] == [551, 552]
    assert spec["steps_per_run"] == 60000 and spec["eval_horizon"] == 600
    output.mkdir(parents=True)
    prepare_fresh(output / "training", device=device)
    budget(output / "training")
    atomic_json(
        output / "workflow.json",
        dict(
            spec=spec,
            training_experiment_sha256=canonical_hash(
                read_json(output / "training/fresh_experiment.json")
            ),
            note="New authorized training; no historical checkpoint replacement or curve splicing.",
        ),
    )


def unpack_repro(output, artifact):
    """Extract one immutable verified repro, retaining interrupted staging evidence."""
    path = Path(artifact["absolute_path"])
    assert file_hash(path) == artifact["sha256"], "repro artifact changed"
    target = output / ("repro_" + artifact["sha256"][:16])
    if target.exists():
        assert (target / "SHA256SUMS.json").is_file(), "incomplete immutable repro"
        verify_manifest(target)
        return target
    staging = target.with_name(target.name + "_extracting")
    assert not staging.exists(), f"incomplete extraction retained: {staging}"
    staging.mkdir()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "duplicate archive paths"
        for name in names:
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and ".." not in rel.parts and "\\" not in name
            assert (staging / name).resolve().is_relative_to(staging.resolve())
            assert archive.getinfo(name).external_attr >> 16 & 0o170000 != 0o120000
        archive.extractall(staging)
    assert (staging / "SHA256SUMS.json").is_file(), "repro manifest required"
    verify_manifest(staging)
    staging.rename(target)
    return target


def checkpoint_map(repro, tail):
    """The best may belong to prefix24k; its exact weights live under parents/."""
    mapping = {}
    sums = read_json(repro / "SHA256SUMS.json")
    for name in continuous.MODELS:
        step = read_json(tail / "runs" / name / "commit.json")["best_step"]
        candidates = [f"runs/{name}/checkpoints/{step:06d}.pt"]
        if step <= 24000:
            candidates.append(f"parents/runs/{name}/checkpoints/{step:06d}.pt")
        found = [rel for rel in candidates if (repro / rel).is_file()]
        assert found, f"missing NEW selected best: {name} step{step}"
        rel = found[0]
        assert file_hash(repro / rel) == sums[rel]
        mapping[name] = rel
    return mapping


def delivery(output, context):
    """One download includes both audit AND repro; never lose best weights again."""
    eval_root = output / "continuous600"
    eval_receipt = output / "continuous600_export_receipt.json"
    info = read_json(eval_receipt)
    assert info["accepted"] and read_json(eval_root / "verification.json")["accepted"]
    eval_package = Path(info["path"])
    assert file_hash(eval_package) == info["sha256"]
    files = {
        "evaluation/" + eval_package.name: eval_package,
        "evaluation/" + eval_receipt.name: eval_receipt,
        "workflow.json": output / "workflow.json",
        "checkpoint_selection.json": eval_root / "checkpoint_selection.json",
        "training/fresh_experiment.json": output / "training/fresh_experiment.json",
        "training/fresh_budget_ledger.json": output / "training/fresh_budget_ledger.json",
        "training/" + Path(context["receipt_path"]).name: Path(context["receipt_path"]),
    }
    for item in context["artifacts"].values():
        path = Path(item["absolute_path"])
        assert file_hash(path) == item["sha256"]
        files["training/" + path.name] = path
    hashes = {rel: file_hash(path) for rel, path in sorted(files.items())}
    token = canonical_hash(hashes)
    receipt = output / "delivery_receipt.json"
    if receipt.exists():
        previous = read_json(receipt)
        assert previous["evidence_sha256"] == token
        assert file_hash(Path(previous["path"])) == previous["sha256"]
        return previous
    target = output.parent / (output.name + "_results.zip")
    assert not target.exists(), "retain existing delivery; do not overwrite"
    temporary = target.with_suffix(".zip.tmp")
    assert not temporary.exists(), f"retain interrupted packaging evidence: {temporary}"
    with zipfile.ZipFile(temporary, "x", allowZip64=True) as archive:
        for rel, path in sorted(files.items()):
            archive.write(
                path,
                "bs5_retrain_eval600/" + rel,
                compress_type=(
                    zipfile.ZIP_STORED if path.suffix == ".zip" else zipfile.ZIP_DEFLATED
                ),
            )
        import json

        archive.writestr("bs5_retrain_eval600/SHA256SUMS.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files) + 1
    temporary.rename(target)
    result = dict(
        path=str(target),
        sha256=file_hash(target),
        bytes=target.stat().st_size,
        evidence_sha256=token,
        includes_selected_weights=True,
        accepted=True,
    )
    atomic_json(receipt, result)
    return result


def run(output):
    output = Path(output).resolve()
    workflow = read_json(output / "workflow.json")
    fresh = output / "training"
    assert workflow["spec"] == read_json(SPEC), "use frozen workflow configuration"
    assert workflow["training_experiment_sha256"] == canonical_hash(
        read_json(fresh / "fresh_experiment.json")
    )
    with experiment_lock(output):
        print("NEW training: C-R/C-N x seeds551/552; 60000/run, 240000 total planned.", flush=True)
        run_fresh(fresh, resume=True)
        tail = fresh / "tail36k"
        finalize(tail)  # Idempotent: reuses the final export already made by training.
        context = continuous.retraining_context(fresh, tail, read_json(tail / "protocol.json"))
        repro = unpack_repro(output, context["artifacts"]["repro_full"])
        mapping = checkpoint_map(repro, tail)
        map_file = output / "new_checkpoint_map.json"
        if map_file.exists():
            assert read_json(map_file) == mapping, "frozen new best selection changed"
        else:
            atomic_json(map_file, mapping)
        evaluation = output / "continuous600"
        if not evaluation.exists():
            selection = continuous.prepare(
                tail, repro, evaluation, checkpoint_map=map_file, device="auto", fresh_root=fresh
            )
            assert selection["status"] == "READY", selection["missing_weights"]
        print(
            "Training complete. Frozen NEW bests; eval600 cap25200, no further updates.", flush=True
        )
        continuous.run(evaluation, checkpoint_root=repro)
        result = delivery(output, context)
        print("DONE. Download:", result["path"], flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, device=args.device)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
