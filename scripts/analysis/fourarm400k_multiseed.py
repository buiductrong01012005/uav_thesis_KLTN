"""Strict training catalog and compact frozen-best bundle; no torch or test execution."""

from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.compare_bs5_fourarm400k_training import parse_csv, digest
from scripts.source_integrity import canonical_training, verify_source, describe_source
from src.configuration import experiment_plan

PLAN = experiment_plan()
ARMS = PLAN["arms"]
SEEDS = PLAN["training_seeds"]
TRAIN_VERSION = "bs5_fourarm400k_v1"
BUNDLE_VERSION = "fourarm400k_frozen_models_v1"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("Unsafe result path")
    return p


def sources(input_dir):
    """Yield readers while their ZIP context remains open; never extract raw reviews."""
    for path in sorted(Path(input_dir).rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entries: " + str(path))
            candidates = [n for n in names if n == "protocol.json" or n.endswith("/protocol.json")]
            if not candidates:
                continue
            if z.testzip() is not None:
                raise ValueError("Corrupt ZIP: " + str(path))
            for n in candidates:
                p = json.loads(z.read(n))
                prefix = n[: -len("protocol.json")]
                yield (
                    p,
                    lambda rel, z=z, prefix=prefix: z.read(prefix + safe(rel).as_posix()),
                    str(path) + ":" + prefix,
                )
    for path in sorted(Path(input_dir).rglob("protocol.json")):
        p = json.loads(path.read_text())
        root = path.parent

        def read(rel, root=root):
            f = root / safe(rel)
            if not f.resolve().is_relative_to(root.resolve()):
                raise ValueError("Result symlink escapes root")
            return f.read_bytes()

        yield p, read, str(root)


def normalized(p):
    p = canonical_training(p)
    for key in (
        "arm",
        "policy_actions",
        "training_seed",
        "runtime",
        "torch_version",
        "comparison_fingerprint",
        "output",
        "resolved_config_sha256",
    ):
        p.pop(key, None)
    if "resolved_config" in p:
        for key in ("arm", "training_seed", "output"):
            p["resolved_config"]["training"].pop(key, None)
    return p


def collect_training(input_dir, include_models=False):
    runs = {}
    provenance = []
    for p, read, origin in sources(input_dir):
        if p.get("version") != TRAIN_VERSION or p.get("diagnostic"):
            continue
        a, s = p["arm"], p["training_seed"]
        if a not in ARMS or s not in SEEDS or p["total_steps"] != PLAN["total_training_steps"]:
            raise ValueError("Unexpected arm, training seed or budget: " + origin)
        if (
            p["validation_seeds"] != PLAN["validation_seeds"]
            or p["reserved_test_seeds"] != PLAN["test_seeds"]
        ):
            raise ValueError("Changed validation/test seeds")
        common = {
            k: v
            for k, v in p.items()
            if k
            not in {"arm", "policy_actions", "runtime", "torch_version", "comparison_fingerprint"}
        }
        if digest(common) != p["comparison_fingerprint"]:
            raise ValueError("Invalid training fingerprint")
        # Every replicate must match the exact source recorded by training.
        for rel, expected in p["source_sha256"].items():
            safe(rel)
            verify_source(rel, expected, ROOT)
        try:
            status = json.loads(read("status.json"))
            v = parse_csv(read("validation.csv"))
            tr = parse_csv(read("training.csv"))
        except (KeyError, FileNotFoundError):
            continue
        if (
            status.get("arm") != a
            or status.get("training_env_version") != TRAIN_VERSION
            or status.get("target_steps") != PLAN["total_training_steps"]
        ):
            raise ValueError("Training status mismatch")
        committed = status["committed_training_steps"]
        if not 0 <= committed <= PLAN["total_training_steps"]:
            raise ValueError("Invalid committed steps")
        seen = {(r["step"], r["seed"]) for r in v}
        allowed = {
            (st, sd)
            for st in p["evaluation_steps"]
            if st <= committed
            for sd in p["validation_seeds"]
        }
        if len(seen) != len(v) or not seen <= allowed:
            raise ValueError("Invalid validation schedule")
        complete = status["status"] == "complete"
        if complete and (committed != PLAN["total_training_steps"] or seen != allowed):
            raise ValueError("Complete training lacks scheduled validations")
        if tr and tr[-1]["step"] != committed:
            raise ValueError("Training CSV and committed status disagree")
        best = status.get("best_by_metric", {}).get("reward_total")
        model = None
        if best:
            grouped = {
                st: np.mean([r["reward_total"] for r in v if r["step"] == st])
                for st in sorted({r["step"] for r in v})
            }
            selected = [r for r in v if r["step"] == best["step"]]
            if len(selected) != len(p["validation_seeds"]) or not np.isclose(
                grouped[best["step"]], best["score"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError("Best/validation mismatch")
            if best["step"] != max(grouped, key=grouped.get):
                raise ValueError("Checkpoint was not selected by declared reward maximum")
            if include_models and complete:
                # Public alias must equal the immutable selected model when that file is present.
                try:
                    model = read("best.pt")
                except (KeyError, FileNotFoundError):
                    raise ValueError(
                        "Attach full REVIEW with best.pt, not just slim CSVs: " + origin
                    )
                if best.get("file"):
                    try:
                        stable = read(safe(best["file"]).as_posix())
                    except (KeyError, FileNotFoundError):
                        stable = None
                    if stable is not None and stable != model:
                        raise ValueError("best.pt alias differs from committed selection")
        data = dict(
            protocol=p,
            status=status,
            training=tr,
            validation=v,
            model_sha256=sha_bytes(model) if model is not None else None,
        )
        key = (a, s)
        if key in runs and runs[key]["data"] != data:
            raise ValueError("Conflicting versions of the same training run: " + str(key))
        runs[key] = dict(data=data, model=model)
        provenance.append(dict(origin=origin, arm=a, training_seed=s, **describe_source(p)))
    if not runs:
        raise ValueError("Attach REVIEW ZIPs or extracted folders for seeds 551/552/553")
    if len({digest(normalized(r["data"]["protocol"])) for r in runs.values()}) != 1:
        raise ValueError("Science differs beyond arm and training seed")
    return runs, provenance


def make_bundle(runs, output):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    if set(runs) != expected or any(
        r["data"]["status"]["status"] != "complete" or r["model"] is None for r in runs.values()
    ):
        raise ValueError("Need all 12 complete runs and their best.pt files")
    entries = []
    files = {}
    for (arm, seed), r in sorted(runs.items()):
        d = r["data"]
        p = d["protocol"]
        best = d["status"]["best_by_metric"]["reward_total"]
        slug = arm.replace("-", "_") + f"_S{seed}"
        prefix = "models/" + slug + "/"
        record = dict(
            arm=arm,
            training_seed=seed,
            selected_step=best["step"],
            validation_score=best["score"],
            checkpoint=prefix + "best.pt",
            checkpoint_sha256=d["model_sha256"],
            protocol=prefix + "protocol.json",
            protocol_sha256=digest(p),
        )
        files[record["checkpoint"]] = r["model"]
        files[record["protocol"]] = (json.dumps(p, indent=2) + "\n").encode()
        files[prefix + "status.json"] = (json.dumps(d["status"], indent=2) + "\n").encode()
        files[prefix + "resolved_config.json"] = (
            json.dumps(p["resolved_config"], indent=2) + "\n"
        ).encode()
        entries.append(record)
    manifest = dict(
        version=BUNDLE_VERSION,
        plan=PLAN,
        plan_sha256=digest(PLAN),
        models=entries,
        files_sha256={k: sha_bytes(v) for k, v in files.items()},
        scope="All twelve best checkpoints selected on validation; no best training seed or test selection",
        source_policy="exact_sha256",
    )
    files["FROZEN_MODELS_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Frozen bundle CRC")
    return manifest
