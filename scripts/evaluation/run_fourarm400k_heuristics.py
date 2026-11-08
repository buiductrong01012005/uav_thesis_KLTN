"""Three frozen causal baselines in the exact FourArm400k physical environment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
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
from dataclasses import asdict
from src.envs.bs5_fourarm400k import BS5FourArm400kEnv, FairConfig, CollisionConfig
from src.baselines.bs5_delayed_multihop import RandomAR, DelayedDynamicRole
from src.baselines.bs5_delayed_h_only import DelayedHybridOnly
from src.envs.bs5_energy_swap import EnergyConfig
from src.envs.bs5_service_v2 import FULL_CONDITION, ServiceV2Config
from src.envs.bs5_tcf_env import TCFWeights, jain
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics, longest_true
from src.diagnostics.bs5_benchmark_explain import network_sample, waterfall
from scripts.evaluation.fair300k_benchmark_metrics import extend_summary
from src.diagnostics.bs5_fourarm400k_metrics import observe_step

CONFIG = ROOT / "configs/bs5_fourarm400k_heuristic_benchmark.json"
TRAIN_CONFIG = ROOT / "configs/bs5_fourarm400k.json"
VERSION = "fourarm400k_heuristic_benchmark_v1"
LABELS = {"RANDOM":"C-R", "DYNAMIC_AR":"C-R", "HEURISTIC_HO":"H-only"}
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





def make_protocol(label, part, diagnostic=False, seeds=None):
    if label not in LABELS or part not in (1,2):raise ValueError("Unknown baseline/partition")
    cfg=json.loads(CONFIG.read_text());train=json.loads(TRAIN_CONFIG.read_text())
    if train["reserved_test_seeds"]!=cfg["test_seeds"]:raise ValueError("Test seeds differ from the frozen DRL experiment")
    if set(cfg["test_seeds"]) & (set(train["validation_seeds"])|set(cfg["diagnostic_seeds"])|set(range(train["train_seed_base"],train["train_seed_base"]+40))):raise ValueError("Seed leakage")
    tp=dict(service=train["service"],reward_weights=train["presets"]["service_fair"],coverage_floor_mbps=train["coverage_floor_mbps"],
            energy_config=asdict(EnergyConfig(**train["energy_config"])),collision_config=asdict(CollisionConfig(**train["collision_config"])),
            fair_config=asdict(FairConfig(mode="instant")),training_env_version=train["version"],obs_dim=173,state_dim=156)
    if tp["reward_weights"]!={"throughput":.4,"coverage":0.,"fairness":.6} or tp["energy_config"]["energy_weight"]!=.1 or tp["collision_config"]["collision_weight"]!=.2:raise ValueError("Frozen reward changed")
    source={path.relative_to(ROOT).as_posix():sha(path) for path in sorted((ROOT/"src").rglob("*.py"))}
    for rel in cfg["locked_files"]:source[rel]=sha(ROOT/rel)
    physics={k:v for k,v in source.items() if k.startswith(("src/envs/","src/diagnostics/"))}
    evaluation_contract=dict(environment=tp,horizon=cfg["horizon"],test_seeds=cfg["test_seeds"],initial_residual_battery=30.,assignment_seed="scenario seed",physics_source_sha256=physics)
    common=dict(version=VERSION,config=cfg,training_protocol=tp,source_sha256=source,
                environment_contract=evaluation_contract,environment_fingerprint=digest(evaluation_contract))
    group=cfg["diagnostic_seeds"] if diagnostic else cfg["test_seeds"][(part-1)*5:part*5]
    chosen=list(group) if seeds is None else list(seeds)
    if not chosen or len(chosen)!=len(set(chosen)) or not set(chosen)<=set(group):raise ValueError("Use unique seeds from this fixed partition")
    return dict(**common,comparison_fingerprint=digest(common),label=label,arm=LABELS[label],part=part,
                diagnostic=diagnostic,target_seeds=chosen,
                horizon=cfg["diagnostic_horizon"] if diagnostic else cfg["horizon"],selection=dict(selected_step=None))






def runtime():
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines() if x.startswith("model name")), cpu)
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                device="cpu",threads=1,cpu_model=cpu)






def make_env(p):
    tp=p["training_protocol"]
    return BS5FourArm400kEnv(FULL_CONDITION,arm=p["arm"],horizon=p["horizon"],residual_battery=30.,
        service_config=ServiceV2Config(**tp["service"]),reward_weights=TCFWeights(**tp["reward_weights"]),
        coverage_floor_mbps=tp["coverage_floor_mbps"],energy_config=EnergyConfig(**tp["energy_config"]),
        collision_config=CollisionConfig(**tp["collision_config"]),fair_config=FairConfig(**tp["fair_config"]))



class FrozenController:
    """Keep old controllers unchanged, including their original one-step TCF objective.

    New physical energy and blocked-move reward are evaluated by the real env.
    No simulator state, diagnostics, future rollout or EMA truth enters the actor.
    """
    def __init__(self,label,seed):
        self.label=label
        self.inner=(RandomAR(np.random.SeedSequence([seed,0x52414E44])) if label=="RANDOM" else
                    DelayedDynamicRole() if label=="DYNAMIC_AR" else DelayedHybridOnly())
    def act(self,obs,mask):
        obs=np.asarray(obs)
        if obs.shape!=(5,173) or not np.isfinite(obs).all():raise ValueError("Expected the new173D observations")
        actions=self.inner.act(obs[:,:148].copy(),np.asarray(mask))
        if not all(mask[i,int(a)] for i,a in enumerate(actions)):raise AssertionError("Illegal baseline action")
        return actions
    @property
    def last_stats(self):return self.inner.last_stats


def start_state(p, seed, controller=None):
    env = make_env(p)
    current = env.reset(seed=seed, assignment_seed=seed)
    return dict(seed=seed, env=env, current=current, controller=FrozenController(p["label"],seed) if controller is None else controller, rng=np.random.default_rng(seed),
                rows=[], arrays={}, energy=[], collision=[], network=[],
                cycles=[], pending={}, rejoins=0, ue_trace_sha256="00"*32,
                initial_energy_j=env.energy_j.copy(), wall_seconds=0.0)


def advance(state, p):
    env = state["env"]
    before = env.status.copy()
    tick = time.perf_counter()
    actions = state["controller"].act(state["current"]["obs"], state["current"]["action_mask"])
    actor_ms = 1000*(time.perf_counter()-tick)
    state["current"] = env.step(actions)
    sample = dict(slot=int(env.t), **observe_step(env, state["current"]["info"]), actor_ms=actor_ms,
                  radio_ms=float(env.last_v2_network["radio_runtime_ms"]))
    sample.update(waterfall(360.0, sample["covered_ues"], sample["associated_ues"],
                            sample["access_offer_mbps"], sample["reachable_offer_mbps"], sample["throughput_mbps"]))
    stats=state["controller"].last_stats
    sample.update(heuristic_estimated_score_calls=float(stats.get("estimated_score_calls",0)),
        heuristic_roles_evaluated=float(stats.get("roles_evaluated",0)),heuristic_role_cache_hit=float(stats.get("role_cache_hit",0)),
        heuristic_recovery_plan=float(stats.get("recovery_plan",0)),heuristic_fixed_hybrid=float(p["label"]=="HEURISTIC_HO"),
        actor_exceeds_slot=float(actor_ms>env.cfg.slot_s*1000))
    state["rows"].append(sample)
    state["energy"].append(deepcopy(env.last_energy_audit))
    state["collision"].append(deepcopy(env.last_collision_audit))
    snap, net = env.last_service_snapshot, env.last_v2_network
    covered = np.zeros(90, bool)
    association = np.full(90, -1, dtype=np.int8)
    reachable = np.zeros(90, bool)
    for aid, users in net["association"].items():
        association[users] = aid
        reachable[users] = bool(env.last_v2_radio_detail["paths"].get(aid))
    for aid in np.flatnonzero((snap["status"]==0)&np.isin(snap["roles"],[0,2])):
        covered |= np.linalg.norm(snap["ue_positions"]-snap["uav_positions"][aid],axis=1)<=env.cfg.access_radius_cells+1e-9
    arrays = dict(ue_association=association, ue_covered=covered, ue_reachable=reachable,
                  access_raw_rates=net["access_raw_ue_rates"].copy(), ue_rates=net["ue_rate_mbps"].copy(), ema_rates=env.ema_rates.copy(),
                  history_map=env.history_reports.vector(env.t).copy(), ue_positions=snap["ue_positions"].copy(),
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
    row = dict(label=p["label"], arm=p["arm"], seed=state["seed"], slots=len(rows), target_slots=p["horizon"],
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
    row.update(extend_summary(state, p))
    zero=np.asarray([r["throughput_mbps"]<=1e-9 for r in rows])
    access=np.asarray([r["active_access"]+r["active_hybrid"] for r in rows])
    associated=np.asarray([r["associated_ues"] for r in rows]);reachable=np.asarray([r["reachable_ues"] for r in rows])
    row.update(zero_no_active_access_slots=int((zero&(access==0)).sum()),zero_no_association_slots=int((zero&(access>0)&(associated==0)).sum()),zero_no_route_slots=int((zero&(associated>0)&(reachable==0)).sum()),zero_other_slots=int((zero&(reachable>0)).sum()))
    row["environment_fingerprint"]=p["environment_fingerprint"]
    row["actor_over_5s_slot_fraction"]=float(np.mean([r["actor_exceeds_slot"] for r in rows]))
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
    if saved["protocol_sha256"] != digest(p) or len(saved["state"]["rows"]) != progress["slot"] or saved["state"]["seed"] != progress["seed"] or progress["seed"] not in p["target_seeds"]:
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
    public = ["slots.csv", "ue_summary.csv", "trace.npz", "energy.npz", "energy_events.json",
              "collision.npz", "collision_events.json", "cycles.json", "network_samples.jsonl", "summary.json"]
    atomic_json(folder / "COMMIT.json", dict(protocol_sha256=digest(p),
        files_sha256={name:sha(folder/name) for name in public}))


def verify_seed_commit(folder, p):
    record = json.loads((folder / "COMMIT.json").read_text())
    if record["protocol_sha256"] != digest(p):
        raise ValueError("Seed commit belongs to another protocol")
    for name, expected in record["files_sha256"].items():
        if Path(name).name != name or sha(folder/name) != expected:
            raise ValueError("Corrupt completed seed: " + name)
    return json.loads((folder / "summary.json").read_text())


def clear_completed_resume(folder):
    # Full scientific trace is committed; only redundant restart snapshots are removed.
    for name in ("resume_0.pkl", "resume_1.pkl", "progress.json"):
        (folder/name).unlink(missing_ok=True)


def export(root):
    records = []
    for kind in ("REVIEW", "RESUME"):
        path = root.parent / (root.name+"_"+kind+".zip")
        temp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix not in (".tmp", ".lock") and (kind == "RESUME" or not ((f.name.startswith("resume_") and f.suffix == ".pkl") or f.name == "progress.json")):
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


def run(args, *, protocol_factory=None, state_factory=None, runtime_factory=None):
    global STOP
    STOP = False
    factory = protocol_factory or make_protocol
    starter = state_factory or start_state
    p = factory(args.label,args.part,args.diagnostic,args.seeds)
    p["runtime"] = (runtime_factory or runtime)()
    if args.check_only:
        # Always use a separate diagnostic seed; never execute a test slot here.
        check=factory(args.label,args.part,True)
        fixture=starter(check,check["target_seeds"][0]);advance(fixture,check)
        print(json.dumps(dict(status="preflight_ok",label=args.label,seeds=p["target_seeds"],
            horizon=p["horizon"],environment_fingerprint=p["environment_fingerprint"],runtime=p["runtime"]),indent=2),flush=True)
        return p
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
                if (folder / "COMMIT.json").exists() and verify_seed_commit(folder, p)["complete"]:
                    summary = verify_seed_commit(folder, p)
                    if summary["comparison_fingerprint"] != p["comparison_fingerprint"] or summary["slots"] != p["horizon"] or summary["seed"] != seed or summary["label"] != args.label:
                        raise ValueError("Completed seed belongs to another experiment")
                    print("Reuse completed", args.label, seed, flush=True)
                    continue
                if (folder / "COMMIT.json").exists() and not (folder / "progress.json").exists():
                    raise ValueError("Incomplete seed has no restart snapshot. Use the matching RESUME archive, not a partial REVIEW.")
                state = load_state(folder, p) if (folder / "progress.json").exists() else starter(p, seed)
                if state["seed"] != seed:
                    raise ValueError("Resume state belongs to another scenario seed")
                previous_wall = state["wall_seconds"]
                tick = time.perf_counter()
                print("Start/resume", args.label, seed, "slot", len(state["rows"]), flush=True)
                while len(state["rows"]) < p["horizon"]:
                    advance(state, p)
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
                if complete:
                    clear_completed_resume(folder)
                if complete and seed != p["target_seeds"][-1]:
                    export(root)
                if not complete:
                    print("Paused safely at", len(state["rows"]), flush=True)
                    break
            completed = sum((root / f"seed{s}" / "summary.json").exists() and json.loads((root / f"seed{s}" / "summary.json").read_text())["complete"] for s in p["target_seeds"])
            atomic_json(root / "status.json", dict(status="complete" if completed == len(p["target_seeds"]) else "paused", completed=completed, target=len(p["target_seeds"])))
        except BaseException as exc:
            atomic_json(root / "status.json", dict(status="error", error_type=type(exc).__name__, error=str(exc)))
            raise
        finally:
            export(root)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", choices=tuple(LABELS), required=True)
    p.add_argument("--seeds",type=int,nargs="+",help="Optional subset of the fixed partition")
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
