"""Frozen best models: one training replicate and test partition per offline CPU worker."""

from pathlib import Path
import argparse
import hashlib
import json
import signal
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import run_fourarm400k_heuristics as core
from scripts.analysis.fourarm400k_multiseed import (
    ARMS,
    SEEDS,
    PLAN,
    BUNDLE_VERSION,
    digest,
    normalized,
    safe,
)
from scripts.source_integrity import verify_source

VERSION = "fourarm400k_drl_test_v1"
SCIENCE_KEYS = [
    "version",
    "config",
    "training_protocol",
    "source_sha256",
    "environment_contract",
    "environment_fingerprint",
    "label",
    "arm",
    "training_seed",
    "selection",
    "model_bundle_sha256",
    "model_protocol_sha256",
    "benchmark_source_sha256",
]


def read_model(input_dir, arm, training_seed):
    candidates = []
    for path in sorted(Path(input_dir).rglob("FROZEN_MODELS_MANIFEST.json")):
        m = json.loads(path.read_text())
        if m.get("version") == BUNDLE_VERSION:
            root = path.parent

            def read(rel, root=root):
                f = root / safe(rel)
                if not f.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Bundle path escapes root")
                return f.read_bytes()

            candidates.append((m, read, str(path)))
    # Read only manifest + selected checkpoint + protocols, without extracting ZIPs.
    for path in (
        [Path(input_dir)]
        if Path(input_dir).is_file() and Path(input_dir).suffix == ".zip"
        else sorted(Path(input_dir).rglob("*.zip"))
    ):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n == "FROZEN_MODELS_MANIFEST.json" or n.endswith("/FROZEN_MODELS_MANIFEST.json"):
                    m = json.loads(z.read(n))
                    if m.get("version") != BUNDLE_VERSION:
                        continue
                    if z.testzip() is not None:
                        raise ValueError("Frozen model ZIP CRC failed")
                    prefix = n[: -len("FROZEN_MODELS_MANIFEST.json")]
                    data = {k: z.read(prefix + safe(k).as_posix()) for k in m["files_sha256"]}
                    candidates.append(
                        (m, lambda rel, data=data: data[safe(rel).as_posix()], str(path))
                    )
    if not candidates:
        raise FileNotFoundError(
            "Attach FOURARM400K_FROZEN_MODELS.zip or its extracted folder from the 12-run training merge"
        )
    if len({digest(m) for m, _, _ in candidates}) != 1:
        raise ValueError("Multiple different frozen model bundles")
    expected = {(a, s) for a in ARMS for s in SEEDS}
    for m, read, origin in candidates:
        if m["plan"] != PLAN or m["plan_sha256"] != digest(PLAN):
            raise ValueError("Different frozen experiment plan")
        if (
            len(m["models"]) != len(expected)
            or {(r["arm"], r["training_seed"]) for r in m["models"]} != expected
        ):
            raise ValueError("Model bundle must contain all 12 training replicates")
        for rel, sha in m["files_sha256"].items():
            if hashlib.sha256(read(rel)).hexdigest() != sha:
                raise ValueError("Corrupt frozen model file: " + rel)
    m, read, _ = candidates[0]
    protocols = []
    for rec in m["models"]:
        tp = json.loads(read(rec["protocol"]))
        protocols.append(tp)
        if (
            tp["arm"] != rec["arm"]
            or tp["training_seed"] != rec["training_seed"]
            or digest(tp) != rec["protocol_sha256"]
            or tp.get("diagnostic")
            or tp["total_steps"] != PLAN["total_training_steps"]
        ):
            raise ValueError("Frozen model protocol mismatch")
        st = json.loads(read(Path(rec["protocol"]).with_name("status.json").as_posix()))
        b = st["best_by_metric"]["reward_total"]
        if (
            st["status"] != "complete"
            or st["committed_training_steps"] != PLAN["total_training_steps"]
            or b["step"] != rec["selected_step"]
            or not np.isclose(b["score"], rec["validation_score"])
        ):
            raise ValueError("Frozen model is not the declared completed best")
        for rel, sha in tp["source_sha256"].items():
            safe(rel)
            verify_source(rel, sha, ROOT)
    if len({digest(normalized(p)) for p in protocols}) != 1:
        raise ValueError("Frozen runs differ beyond arm/training seed")
    rec = next(r for r in m["models"] if r["arm"] == arm and r["training_seed"] == training_seed)
    payload = read(rec["checkpoint"])
    if hashlib.sha256(payload).hexdigest() != rec["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    return rec, json.loads(read(rec["protocol"])), payload, digest(m)


def make_protocol(
    arm, training_seed, part, model, train_protocol, bundle_hash, diagnostic=False, seeds=None
):
    base = core.make_protocol(
        "HEURISTIC_HO" if arm == "H-only" else "DYNAMIC_AR", part, diagnostic, seeds
    )
    tp = base["training_protocol"]
    for k in tp:
        if tp[k] != (
            train_protocol["resolved_config"]["environment"]
            if k == "resolved_environment"
            else train_protocol[k]
        ):
            raise ValueError("Test and training environment mismatch: " + k)
    base.update(
        version=VERSION,
        label=f"DRL_{arm}_S{training_seed}",
        arm=arm,
        training_seed=training_seed,
        selection=dict(
            selected_step=model["selected_step"],
            validation_score=model["validation_score"],
            metric="reward_total",
            checkpoint_sha256=model["checkpoint_sha256"],
        ),
        model_bundle_sha256=bundle_hash,
        model_protocol_sha256=digest(train_protocol),
        benchmark_source_sha256={
            rel: core.sha(ROOT / rel)
            for rel in [
                "scripts/evaluation/run_fourarm400k_drl_benchmark.py",
                "scripts/analysis/fourarm400k_multiseed.py",
                "src/configuration.py",
                "scripts/source_integrity.py",
            ]
        },
    )
    base["comparison_fingerprint"] = digest({k: base[k] for k in SCIENCE_KEYS})
    return base


from scripts.evaluation.actor import FrozenActor, runtime


def run(args):
    model, tp, payload, bundle_hash = read_model(args.models, args.arm, args.training_seed)
    actor = FrozenActor(payload, tp, model)

    def factory(label, part, diagnostic=False, seeds=None):
        return make_protocol(
            args.arm, args.training_seed, part, model, tp, bundle_hash, diagnostic, seeds
        )

    def starter(p, seed):
        return core.start_state(p, seed, controller=actor)

    args.label = f"DRL_{args.arm}_S{args.training_seed}"
    return core.run(args, protocol_factory=factory, state_factory=starter, runtime_factory=runtime)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--training-seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--part", type=int, choices=[1, 2], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--max-seconds", type=float, default=36000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--stop-after-slots", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.save_every < 1 or a.log_every < 1:
        raise ValueError("Positive log/save intervals required")
    signal.signal(signal.SIGTERM, core.stop_handler)
    signal.signal(signal.SIGINT, core.stop_handler)
    run(a)
