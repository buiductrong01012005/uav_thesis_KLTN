"""Versioned additional fixed-H heuristic; frozen six-arm science stays intact."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import signal
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filelock import FileLock
from scripts.evaluation import run_bs5_multihop_benchmark6 as base
from src.baselines.bs5_delayed_h_only import DelayedHybridOnly
from src.diagnostics.bs5_benchmark_explain import slot_sample, network_sample
from scripts.analysis.report_bs5_heuristic_ho import generate

sha, digest, atomic_json = base.sha, base.digest, base.atomic_json
VERSION = "bs5_heuristic_ho_benchmark_1"
LABEL = "HEURISTIC_HO"
STOP = False
EXTENSION_FILES = [
    "src/baselines/bs5_delayed_h_only.py",
    "scripts/evaluation/run_bs5_heuristic_ho.py",
    "scripts/analysis/report_bs5_heuristic_ho.py",
    "scripts/analysis/merge_bs5_heuristic_ho.py",
    "configs/bs5_heuristic_ho_extension.json",
    "configs/bs5_heuristic_ho_repo_import.json",
]


def runtime_versions():
    import scipy

    try:
        torch = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch = "not-installed"
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch,
        device="cpu",
        threads=1,
    )


def stop_handler(*_):
    global STOP
    STOP = True


def verify_reference_source(cfg):
    """Validate explicit import adaptations without rewriting historical identities."""
    common = cfg["reference_common"]
    locked = json.loads((ROOT / "configs/bs5_multihop_benchmark6.json").read_text())
    reference_test = dict(common, phase="test", target_seeds=locked["test_seeds"])
    if digest(reference_test) != cfg["reference_test_fingerprint"]:
        raise ValueError("Invalid frozen reference identity")
    receipt = json.loads((ROOT / "configs/bs5_heuristic_ho_repo_import.json").read_text())
    if receipt["reference_test_fingerprint"] != cfg["reference_test_fingerprint"]:
        raise ValueError("Different import reference identity")
    adaptations = receipt["integration_files"]
    if not set(adaptations) <= set(common["source_sha256"]):
        raise ValueError("Unknown source import adaptation")
    actual = {}
    for rel, original in common["source_sha256"].items():
        expected = original
        if rel in adaptations:
            item = adaptations[rel]
            if item["original_sha256"] != original:
                raise ValueError("Invalid source import adaptation: " + rel)
            expected = item["integrated_sha256"]
        variants = receipt["checkout_line_endings"][rel]
        if expected not in variants:
            raise ValueError("Invalid checkout line-ending receipt: " + rel)
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Frozen benchmark science changed: " + rel)
        current = sha(path)
        if current not in variants:
            raise ValueError("Frozen benchmark science changed: " + rel)
        actual[rel] = current
    return actual


def protocol(args):
    cfg = json.loads((ROOT / "configs/bs5_heuristic_ho_extension.json").read_text())
    old = json.loads((ROOT / "configs/bs5_multihop_benchmark6.json").read_text())
    integrated_source = verify_reference_source(cfg)
    common = cfg["reference_common"].copy()
    runtime = runtime_versions()
    if not args.diagnostic and runtime != common["runtime"]:
        raise ValueError(
            "Production runtime must match the six completed arms. Expected "
            + json.dumps(common["runtime"])
            + "; got "
            + json.dumps(runtime)
            + ". Diagnostic mode is separate and cannot be merged as test."
        )
    common.update(
        phase=args.phase,
        diagnostic=args.diagnostic,
        runtime=runtime,
        target_seeds=old["diagnostic_seeds"] if args.diagnostic else old[args.phase + "_seeds"],
        horizon=old["diagnostic_horizon"] if args.diagnostic else old["horizon"],
        network_sample_every=16 if args.diagnostic else old["network_sample_every_slots"],
    )
    common["counterfactual_bandwidth_probes"] = True
    common["runtime"] = runtime
    p = dict(
        common,
        version=VERSION,
        label=LABEL,
        arm="H-only",
        base_common=common,
        base_experiment_fingerprint=digest(common),
        extension_source_sha256={rel: sha(ROOT / rel) for rel in EXTENSION_FILES},
        integrated_source_sha256=integrated_source,
        algorithm_version=cfg["algorithm_version"],
        additional_information_contract=cfg["information_contract"],
        controller_contract=cfg["controller_contract"],
        extension_scope=cfg["scope"],
    )
    p["experiment_fingerprint"] = digest(p)
    return p


def start_state(p, seed):
    state = base.start_state(p, seed)
    state["controller"] = DelayedHybridOnly()
    return state


def export(root, p):
    root = Path(root)
    generate(root)
    outputs = []
    for kind, resume in [("REVIEW", False), ("RESUME", True)]:
        path = root.parent / f"{LABEL}_{p['phase']}_BENCH_{kind}.zip"
        temp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix not in [".lock", ".tmp"]
                    and (resume or f.name != "resume.pkl")
                ):
                    z.write(f, f.relative_to(root))
        with zipfile.ZipFile(temp) as z:
            if z.testzip():
                raise ValueError("Corrupt export")
        temp.replace(path)
        outputs.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    atomic_json(
        root.parent / f"{LABEL}_{p['phase']}_BENCH_INDEX.json",
        dict(
            version=VERSION,
            base_experiment_fingerprint=p["base_experiment_fingerprint"],
            outputs=outputs,
        ),
    )
    return outputs


def run(args):
    global STOP
    STOP = False
    p = protocol(args)
    chosen = args.seeds or p["target_seeds"]
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(p["target_seeds"]):
        raise ValueError("Use unique locked seeds for this phase")
    root = Path(args.output)
    folder = root / LABEL
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    with FileLock(str(folder) + ".lock", timeout=0):
        path = folder / "protocol.json"
        if path.is_file():
            if json.loads(path.read_text()) != p:
                raise ValueError("Resume refused: source/runtime/phase/protocol changed")
        else:
            atomic_json(path, p)
        try:
            for seed in chosen:
                dest = folder / f"seed{seed}"
                dest.mkdir(exist_ok=True)
                if (dest / "summary.json").is_file() and json.loads(
                    (dest / "summary.json").read_text()
                ).get("complete"):
                    print("Reuse completed", LABEL, seed, flush=True)
                    continue
                if (dest / "resume.pkl").is_file():
                    progress = json.loads((dest / "progress.json").read_text())
                    if sha(dest / "resume.pkl") != progress["resume_sha256"]:
                        raise ValueError("Resume checksum failed")
                    with (dest / "resume.pkl").open("rb") as f:
                        saved = pickle.load(f)
                    if saved["protocol_fingerprint"] != digest(p):
                        raise ValueError("Resume protocol changed")
                    state = saved["state"]
                    print("Resume", LABEL, seed, "slot", len(state["rows"]), flush=True)
                else:
                    if (dest / "summary.json").is_file():
                        raise ValueError(
                            "Partial REVIEW has no resume.pkl; attach RESUME or choose a fresh output folder"
                        )
                    state = start_state(p, seed)
                seed_started = time.perf_counter()
                previous_wall = state["wall_seconds"]
                while len(state["rows"]) < p["horizon"]:
                    env = state["env"]
                    current = state["current"]
                    before = dict(
                        status=env.status.copy(),
                        battery=env.battery.copy(),
                        positions=env.uav_positions.copy(),
                    )
                    tic = time.perf_counter()
                    actions = state["controller"].act(current["obs"], current["action_mask"])
                    stats = state["controller"].last_stats.copy()
                    stats["decision_ms"] = (time.perf_counter() - tic) * 1000
                    if not all(current["action_mask"][a, int(x)] for a, x in enumerate(actions)):
                        raise AssertionError("Illegal action")
                    state["current"] = env.step(actions)
                    row, arrays, served = slot_sample(env, before, state["current"]["info"], stats)
                    row.update(
                        heuristic_fixed_hybrid=1,
                        heuristic_estimated_common_tcf=stats["estimated_common_tcf"],
                        heuristic_reported_ue_count=stats["reported_ue_count"],
                    )
                    arrays["actions"] = np.asarray(actions, dtype=np.int8)
                    state["rows"].append(row)
                    for k, v in arrays.items():
                        state["arrays"].setdefault(k, []).append(v)
                    data = np.asarray(
                        env.last_service_snapshot["ue_positions"], dtype="<f8"
                    ).tobytes(order="C")
                    state["ue_trace_hash"] = hashlib.sha256(
                        bytes.fromhex(state["ue_trace_hash"]) + data
                    ).hexdigest()
                    base.update_cycles(state, before, served)
                    slot = len(state["rows"])
                    processed += 1
                    if slot % p["network_sample_every"] == 0:
                        state["network"].append(network_sample(env, True))
                    if slot % args.log_every == 0 or slot == p["horizon"]:
                        rs = state["rows"]
                        rates = np.asarray(state["arrays"]["rates"])
                        event = dict(
                            label=LABEL,
                            seed=seed,
                            slot=slot,
                            target=p["horizon"],
                            mean_mbps=float(np.mean([r["throughput_mbps"] for r in rs])),
                            covered=float(np.mean([r["covered_ues"] for r in rs])),
                            reachable=float(np.mean([r["reachable_ues"] for r in rs])),
                            ue_ge1=float(np.mean([r["service_coverage_ues"] for r in rs])),
                            unique_ge1=int((rates >= 1 - 1e-9).any(axis=0).sum()),
                            return_charge_pct=100
                            * float(np.mean([r["unavailable_uavs"] / 5 for r in rs])),
                            actor_ms=float(np.mean([r["actor_ms"] for r in rs[-args.log_every :]])),
                        )
                        state["live_events"].append(event)
                        print(
                            f"{LABEL} seed={seed} {slot}/{p['horizon']} | Mbps={event['mean_mbps']:.2f} | covered/BH/ge1={event['covered']:.1f}/{event['reachable']:.1f}/{event['ue_ge1']:.1f} | unique_ge1={event['unique_ge1']}/90 | Return+Charge={event['return_charge_pct']:.1f}% | actor={event['actor_ms']:.1f}ms",
                            flush=True,
                        )
                    state["wall_seconds"] = previous_wall + time.perf_counter() - seed_started
                    stop = (
                        STOP
                        or (
                            args.max_seconds > 0
                            and time.perf_counter() - started >= args.max_seconds
                        )
                        or (args.stop_after_slots > 0 and processed >= args.stop_after_slots)
                    )
                    if slot % args.save_every == 0 or stop or slot == p["horizon"]:
                        base.save_state(dest, p, state)
                    if stop:
                        break
                complete = len(state["rows"]) == p["horizon"]
                base.save_seed(dest, p, state, complete)
                if not complete:
                    print("Paused safely at committed slot", len(state["rows"]), flush=True)
                    break
        finally:
            export(root, p)
    return p


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["pilot", "test"], default="test")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--max-seconds", type=float, default=10 * 3600)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--stop-after-slots", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if min(args.save_every, args.log_every) < 1:
        raise ValueError("Invalid log/save interval")
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    run(args)
