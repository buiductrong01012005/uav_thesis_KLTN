"""Evaluate frozen P03/P05 policies; never train or select on test traces."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import sys
import time
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from src.envs.bs5_energy_collision import BS5EnergyCollisionEnv, CollisionConfig
from src.envs.bs5_energy_swap import EnergyConfig
from src.envs.bs5_service_v2 import FULL_CONDITION, ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights, jain
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics, longest_true
from src.diagnostics.bs5_benchmark_explain import network_sample, waterfall
from scripts.evaluation.energy_benchmark_metrics import observe_step
from scripts.evaluation.benchmark_asset_paths import benchmark_asset_dir, benchmark_asset_path

CONFIG = ROOT / "configs/bs5_energy_collision_benchmark120k.json"
ASSETS = benchmark_asset_dir(ROOT, "energy_collision_120k")
STOP = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_npz(path, arrays):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def verify_assets():
    """Check scientific source against each actual training protocol."""
    registry = json.loads((ASSETS / "selection_manifest.json").read_text())
    fix = json.loads((ROOT / "configs/metrics_numeric_fix1_manifest.json").read_text())
    allowed = fix["changed_file"]
    for label, record in registry.items():
        p = json.loads(benchmark_asset_path(ASSETS, record["training_protocol_file"]).read_text())
        if digest(p) != record["protocol_sha256"] or sha(benchmark_asset_path(ASSETS, record["file"])) != record["sha256"]:
            raise ValueError("Checkpoint or training protocol mismatch: " + label)
        if p["arm"] != "C-R" or p["training_seed"] != 551 or p["total_steps"] != 120000:
            raise ValueError("Unexpected training experiment")
        for rel, expected in p["source_sha256"].items():
            current = sha(ROOT / rel)
            if current == expected:
                continue
            if not (rel == allowed and expected == fix["old_sha256"] and current == fix["new_sha256"]):
                raise ValueError("Frozen training source changed: " + rel)
    a, b = [json.loads(benchmark_asset_path(ASSETS, registry[k]["training_protocol_file"]).read_text()) for k in ("P03", "P05")]
    if {k: v for k, v in a.items() if k != "collision_config"} != {k: v for k, v in b.items() if k != "collision_config"}:
        raise ValueError("Training protocols differ beyond collision weight")
    return registry


def make_protocol(label, part, diagnostic=False):
    cfg = json.loads(CONFIG.read_text())
    registry = verify_assets()
    source = {}
    for folder in ("src", "scripts/evaluation"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            source[str(path.relative_to(ROOT))] = sha(path)
    for rel in ("scripts/analysis/report_energy_collision_benchmark.py", "configs/bs5_energy_collision_benchmark120k.json"):
        source[rel] = sha(ROOT / rel)
    common = dict(version=cfg["version"], config=cfg, registry=registry, source_sha256=source)
    training = json.loads(benchmark_asset_path(ASSETS, registry[label]["training_protocol_file"]).read_text())
    seeds = cfg["diagnostic_seeds"] if diagnostic else cfg["test_seeds"][(part-1)*5:part*5]
    if set(cfg["test_seeds"]) & (set(training["validation_seeds"]) | set(cfg["diagnostic_seeds"])):
        raise ValueError("Test and validation/diagnostic seeds overlap")
    return dict(**common, comparison_fingerprint=digest(common), label=label, part=part,
                diagnostic=diagnostic, target_seeds=seeds,
                horizon=cfg["diagnostic_horizon"] if diagnostic else cfg["horizon"],
                training_protocol=training, selection=registry[label])


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def runtime():
    import torch
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), cpu)
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=str(torch.__version__), device="cpu", threads=1, cpu_model=cpu,
                torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest())


def load_policy(p):
    import torch
    from src.rl.bs5_multihop_qmix import make_learner
    record = p["selection"]
    saved = torch.load(benchmark_asset_path(ASSETS, record["file"]), map_location="cpu", weights_only=True)
    expected = dict(selected_step=record["selected_step"], selection_metric="reward_total",
                    protocol_sha256=record["protocol_sha256"], obs_dim=148, state_dim=66,
                    n_actions=10, arm="C-R", preset="balanced",
                    training_env_version=p["training_protocol"]["training_env_version"],
                    reward_version=p["training_protocol"]["reward_version"])
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError("Wrong selected model metadata: " + key)
    learner = make_learner("C-R", device="cpu")
    learner.agent.load_state_dict(saved["agent"], strict=True)
    learner.mixer.load_state_dict(saved["mixer"], strict=True)
    learner.agent.eval()
    learner.mixer.eval()
    for module in (learner.agent, learner.mixer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return learner


def make_env(p):
    training = p["training_protocol"]
    return BS5EnergyCollisionEnv(FULL_CONDITION, arm="C-R", horizon=p["horizon"],
        residual_battery=30.0, service_config=ServiceV2Config(**training["service"]),
        reward_weights=TCFWeights(**training["reward_weights"]),
        coverage_floor_mbps=training["coverage_floor_mbps"],
        energy_config=EnergyConfig(**training["energy_config"]),
        collision_config=CollisionConfig(**training["collision_config"]))


def start_state(p, seed):
    env = make_env(p)
    current = env.reset(seed=seed, assignment_seed=seed)
    return dict(seed=seed, env=env, current=current, rng=np.random.default_rng(seed),
                rows=[], arrays={}, energy=[], collision=[], network=[],
                cycles=[], pending={}, rejoins=0, ue_trace_sha256="00"*32,
                initial_energy_j=env.energy_j.copy(), wall_seconds=0.0)


def advance(state, policy, p):
    env = state["env"]
    before = env.status.copy()
    tick = time.perf_counter()
    actions = policy.act(state["current"]["obs"], state["current"]["action_mask"], 0.0, state["rng"])
    actor_ms = 1000*(time.perf_counter()-tick)
    state["current"] = env.step(actions)
    sample = dict(slot=int(env.t), **observe_step(env, state["current"]["info"]), actor_ms=actor_ms,
                  radio_ms=float(env.last_v2_network["radio_runtime_ms"]))
    sample.update(waterfall(360.0, sample["covered_ues"], sample["associated_ues"],
                            sample["access_offer_mbps"], sample["reachable_offer_mbps"], sample["throughput_mbps"]))
    state["rows"].append(sample)
    state["energy"].append(deepcopy(env.last_energy_audit))
    state["collision"].append(deepcopy(env.last_collision_audit))
    snap, net = env.last_service_snapshot, env.last_v2_network
    arrays = dict(ue_rates=net["ue_rate_mbps"].copy(), ue_positions=snap["ue_positions"].copy(),
                  positions=snap["uav_positions"].copy(), roles=snap["roles"].copy(),
                  status=snap["status"].copy(), battery=snap["battery"].copy(),
                  actions=np.asarray(actions, dtype=np.int8), proxy_actions=np.asarray(env._slot_actions, dtype=np.int8),
                  post_status=env.status.copy(), post_battery=env.battery.copy(),
                  path_wireless_hops=np.array([net["path_wireless_hops"].get(a, 0) for a in range(5)], dtype=np.int8))
    for key, value in arrays.items():
        state["arrays"].setdefault(key, []).append(value)
    state["ue_trace_sha256"] = hashlib.sha256(bytes.fromhex(state["ue_trace_sha256"])
        + np.asarray(snap["ue_positions"], dtype="<f8").tobytes(order="C")).hexdigest()
    served = {a: bool(np.any(net["ue_rate_mbps"][users] > 1e-6)) for a, users in net["association"].items()}
    for aid, flow in net["forwarded_mbps_by_uav"].items():
        if flow > 1e-6:
            served[aid] = True
    pending = state["pending"]
    for aid in range(5):
        if before[aid] == 2 and env.status[aid] == 0:
            state["rejoins"] += 1
            if aid in pending:
                pending[aid]["rejoin_slot"] = int(env.t)
        if aid in pending and pending[aid]["rejoin_slot"] is not None and served.get(aid, False):
            event = pending.pop(aid)
            event.update(served_again_slot=int(env.t), outcome="served_again")
            state["cycles"].append(event)
        if before[aid] == 0 and env.status[aid] == 1:
            if aid in pending:
                event = pending.pop(aid)
                event["outcome"] = "next_return_without_service"
                state["cycles"].append(event)
            pending[aid] = dict(agent=aid, return_slot=int(env.t), rejoin_slot=None)
    if env.t % p["config"]["network_sample_every_slots"] == 0:
        state["network"].append(network_sample(env, counterfactual=False))


def summarize(state, p, complete):
    env, rows = state["env"], state["rows"]
    rates = np.asarray(state["arrays"]["ue_rates"])
    row = dict(label=p["label"], seed=state["seed"], slots=len(rows), target_slots=p["horizon"],
               complete=complete, diagnostic=p["diagnostic"], selected_step=p["selection"]["selected_step"],
               comparison_fingerprint=p["comparison_fingerprint"], ue_trace_sha256=state["ue_trace_sha256"],
               **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "slot"},
               **episode_service_metrics(rates, p["training_protocol"]["coverage_floor_mbps"]))
    average = np.maximum(rates, 0).mean(axis=0)
    total_j = sum(r["consumed_j"] for r in rows)
    delivered_mbit = float(np.maximum(rates, 0).sum()*env.cfg.slot_s)
    requests = sum(r["policy_move_requests"] for r in rows)
    blocked = sum(r["active_blocked_uavs"] for r in rows)
    row.update(energy_consumed_wh=total_j/3600, delivered_mbit=delivered_mbit,
               joules_per_mbit=total_j/delivered_mbit if delivered_mbit > 0 else None,
               active_blocked_total=blocked, policy_move_requests_total=requests,
               blocked_per_policy_move=blocked/requests if requests else None,
               returning_blocked_total=sum(r["returning_blocked_uavs"] for r in rows),
               swaps_completed_total=sum(r["swaps_completed"] for r in rows),
               energy_shortfall_total_j=sum(r["energy_shortfall_j"] for r in rows),
               mean_ue_rate_mbps=float(average.mean()), p05_ue_mean_rate_mbps=float(np.quantile(average, .05)),
               unavailable_uav_slot_fraction=float(np.mean([(r["returning_uavs"]+r["swapping_service_snapshot_uavs"])/5 for r in rows])),
               saturated_backhaul_slot_fraction=float(np.mean([r["schedule_airtime"] >= .999999 for r in rows])),
               actor_ms_p95=float(np.quantile([r["actor_ms"] for r in rows], .95)),
               actor_ms_max=max(r["actor_ms"] for r in rows),
               max_flow_violation_mbps=max(r["flow_violation_mbps"] for r in rows),
               charge_rejoin_events=state["rejoins"],
               served_again_cycles=sum(r["outcome"] == "served_again" for r in state["cycles"]),
               unfinished_cycles=len(state["pending"]), simulation_wall_seconds=state["wall_seconds"])
    for key in ("flight_j", "circuit_j", "access_pa_j", "backhaul_pa_j"):
        row[key.replace("_j", "_wh")] = sum(r[key] for r in rows)/3600
    energy = {k: np.asarray([r[k] for r in state["energy"]]) for k in state["energy"][0]}
    collision = np.asarray([r["active_blocked"] for r in state["collision"]])
    row["longest_consecutive_blocked_slots"] = max(longest_true(collision[:, a].astype(bool)) for a in range(5))
    row["energy_balance_max_abs_j"] = float(np.abs(energy["energy_before_j"]+energy["inserted_j"]
        -energy["removed_j"]-energy["consumed_j"]-energy["energy_after_j"]).max())
    balance = state["initial_energy_j"]+energy["inserted_j"].sum(axis=0)-energy["removed_j"].sum(axis=0)-energy["consumed_j"].sum(axis=0)-env.energy_j
    if np.max(np.abs(balance)) > 1e-5:
        raise AssertionError("Episode battery balance failed")
    arrivals = [float(energy["energy_after_j"][r["slot"]-1, r["agent"]]/env.energy_config.capacity_j*100)
                for r in env.energy_events if r["event"] == "swap_start"]
    row.update(arrival_events=len(arrivals), arrival_below20_events=sum(v < 20-1e-7 for v in arrivals),
               min_arrival_soc_percent=min(arrivals) if arrivals else None)
    return row


def save_state(folder, p, state):
    previous = json.loads((folder / "progress.json").read_text()) if (folder / "progress.json").exists() else {}
    generation = 1-int(previous.get("generation", 1))
    filename = f"resume_{generation}.pkl"
    temp = folder / (filename+".tmp")
    with temp.open("wb") as stream:
        pickle.dump(dict(protocol_sha256=digest(p), state=state), stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(folder / filename)
    atomic_json(folder / "progress.json", dict(slot=len(state["rows"]), seed=state["seed"],
        protocol_sha256=digest(p), generation=generation, resume_file=filename, resume_sha256=sha(folder / filename)))


def load_state(folder, p):
    progress = json.loads((folder / "progress.json").read_text())
    if progress["protocol_sha256"] != digest(p) or progress["resume_sha256"] != sha(folder / progress["resume_file"]):
        raise ValueError("Resume integrity/protocol mismatch")
    # Only resume archives produced by this package; never load arbitrary pickle files.
    with (folder / progress["resume_file"]).open("rb") as stream:
        saved = pickle.load(stream)
    if saved["protocol_sha256"] != digest(p) or len(saved["state"]["rows"]) != progress["slot"]:
        raise ValueError("Resume commit mismatch")
    return saved["state"]


def save_seed(folder, p, state, complete):
    if not state["rows"]:
        return
    write_csv(folder / "slots.csv", state["rows"])
    rates = np.maximum(np.asarray(state["arrays"]["ue_rates"]), 0.)
    write_csv(folder / "ue_summary.csv", [dict(ue=u, mean_rate_mbps=float(rates[:, u].mean()),
        fraction_positive=float(np.mean(rates[:, u] > 1e-9)),
        fraction_ge1=float(np.mean(rates[:, u] >= 1-1e-9)),
        fraction_ge4=float(np.mean(rates[:, u] >= 4-1e-6)),
        longest_zero_gap_slots=longest_true(rates[:, u] <= 1e-9)) for u in range(rates.shape[1])])
    atomic_npz(folder / "trace.npz", {k: np.asarray(v) for k, v in state["arrays"].items()})
    for name in ("energy", "collision"):
        atomic_npz(folder / (name+".npz"), {k: np.asarray([x[k] for x in state[name]]) for k in state[name][0]})
        atomic_json(folder / (name+"_events.json"), getattr(state["env"], name+"_events"))
    atomic_json(folder / "cycles.json", dict(rejoins=state["rejoins"], completed=state["cycles"], pending=list(state["pending"].values())))
    (folder / "network_samples.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in state["network"]))
    # Summary is the final commit marker for seed data, and is written last.
    atomic_json(folder / "summary.json", summarize(state, p, complete))


def export(root):
    records = []
    for kind in ("REVIEW", "RESUME"):
        path = root.parent / (root.name+"_"+kind+".zip")
        temp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix not in (".tmp", ".lock") and (kind == "RESUME" or not (f.name.startswith("resume_") and f.suffix == ".pkl")):
                    z.write(f, str(Path(root.name) / f.relative_to(root)))
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError("Export CRC failed")
        temp.replace(path)
        records.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    atomic_json(root.parent / (root.name+"_INDEX.json"), records)
    print("Exported", [r["file"] for r in records], flush=True)


def stop_handler(*_):
    global STOP
    STOP = True


def run(args):
    global STOP
    STOP = False
    configure_torch()
    p = make_protocol(args.label, args.part, args.diagnostic)
    p["runtime"] = runtime()
    policy = load_policy(p)
    # Warm up without stepping an environment or touching test results.
    fixture = start_state(p, p["config"]["diagnostic_seeds"][0])
    for _ in range(3):
        policy.act(fixture["current"]["obs"], fixture["current"]["action_mask"], 0.0, np.random.default_rng(0))
    if args.check_only:
        print(json.dumps(dict(status="preflight_ok", label=args.label, selected_step=p["selection"]["selected_step"],
            seeds=p["target_seeds"], horizon=p["horizon"], runtime=p["runtime"]), indent=2), flush=True)
        return
    import fcntl

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != p:
            raise ValueError("Resume refused: source/checkpoint/partition/runtime changed")
        atomic_json(path, p)
        try:
            for seed in p["target_seeds"]:
                folder = root / f"seed{seed}"
                folder.mkdir(exist_ok=True)
                if (folder / "summary.json").exists() and json.loads((folder / "summary.json").read_text())["complete"]:
                    summary = json.loads((folder / "summary.json").read_text())
                    if summary["comparison_fingerprint"] != p["comparison_fingerprint"] or summary["slots"] != p["horizon"]:
                        raise ValueError("Completed seed belongs to another experiment")
                    print("Reuse completed", args.label, seed, flush=True)
                    continue
                state = load_state(folder, p) if (folder / "progress.json").exists() else start_state(p, seed)
                previous_wall = state["wall_seconds"]
                tick = time.perf_counter()
                print("Start/resume", args.label, seed, "slot", len(state["rows"]), flush=True)
                while len(state["rows"]) < p["horizon"]:
                    advance(state, policy, p)
                    state["wall_seconds"] = previous_wall+time.perf_counter()-tick
                    processed += 1
                    slot = len(state["rows"])
                    if slot % args.log_every == 0 or slot == p["horizon"]:
                        rows = state["rows"]
                        print(f"{args.label} seed={seed} {slot}/{p['horizon']} | Mbps={np.mean([r['throughput_mbps'] for r in rows]):.2f} | Wh={sum(r['consumed_wh'] for r in rows):.2f} | blocked={sum(r['active_blocked_uavs'] for r in rows):.0f} | actor={np.mean([r['actor_ms'] for r in rows[-args.log_every:]]):.2f}ms", flush=True)
                    stop = STOP or (args.max_seconds > 0 and time.perf_counter()-started >= args.max_seconds) or (args.stop_after_slots > 0 and processed >= args.stop_after_slots)
                    if slot % args.save_every == 0 or stop or slot == p["horizon"]:
                        save_state(folder, p, state)
                    if stop:
                        break
                complete = len(state["rows"]) == p["horizon"]
                save_seed(folder, p, state, complete)
                if complete and seed != p["target_seeds"][-1]:
                    export(root)
                if not complete:
                    print("Paused safely at", len(state["rows"]), flush=True)
                    break
            completed = sum((root / f"seed{s}" / "summary.json").exists() and json.loads((root / f"seed{s}" / "summary.json").read_text())["complete"] for s in p["target_seeds"])
            atomic_json(root / "status.json", dict(status="complete" if completed == len(p["target_seeds"]) else "paused", completed=completed, target=len(p["target_seeds"])))
        finally:
            export(root)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=("P03", "P05"), required=True)
    p.add_argument("--part", type=int, choices=(1, 2), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=36000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--stop-after-slots", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--check-only", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.log_every < 1 or args.save_every < 1:
        raise ValueError("Log/save intervals must be positive")
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    run(args)
