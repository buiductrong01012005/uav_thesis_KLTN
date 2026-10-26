"""Shared locked exogenous tapes and compact evidence for fresh Return-v4 work.

No historical checkpoints/assets are imported. Original physics and learner
implementations are used unchanged. Source and runtime are locked per run.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json
from scripts.analysis.bs5_return_v4_results import metric
from scripts.training.run_bs5_launch_ab import atomic_torch, experiment_lock, source_identity, weights_hash
from src.envs.bs5_curriculum_env import condition_for
from src.envs.bs5_return_commit_env import BS5CommittedReturnEnv, RETURN_VERSION, RETURN_CONTRACT
from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
from src.rl.bs5_curriculum_replay import CurriculumReplay, CurriculumTransition


def setup(device="auto", threads=1):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preserve the previous runtime when resuming.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return dict(device=device, python=sys.version, numpy=np.__version__, torch=torch.__version__,
                threads=threads, cuda=torch.version.cuda if device == "cuda" else None,
                hardware=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                deterministic=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(stream.getvalue())
    temp.replace(path)


def save_gz(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=3) as f:
        json.dump(value, f, separators=(",", ":"), allow_nan=False)
    temp.replace(path)


def position_hash(tape):
    return hashlib.sha256(np.asarray(tape, dtype="<f8").tobytes()).hexdigest()


class TraceBank:
    """UE-only ticks, zero env.step calls. Arm is deliberately absent from key."""
    def __init__(self, home, fixture):
        self.home, self.fixture = Path(home), fixture

    def get(self, spec):
        key = canonical_hash(spec)
        self.home.mkdir(parents=True, exist_ok=True)
        path, meta = self.home / f"{key}.npz", self.home / f"{key}.json"
        if meta.exists():
            receipt = read_json(meta)
            assert receipt["spec"] == spec and file_hash(path) == receipt["file_sha256"]
            with np.load(path, allow_pickle=False) as z:
                tape = z["positions"]
            assert position_hash(tape) == receipt["position_sha256"]
            tape.flags.writeable = False
            return tape, receipt
        assert not path.exists(), "Uncommitted tape: preserve evidence; do not silently replace."
        env = BS5CommittedReturnEnv(spec["condition"], allow_relay=True, horizon=spec["horizon"],
                                   residual_battery=spec["residual_battery"], fixture=self.fixture)
        env.reset(seed=spec["scenario_seed"], assignment_seed=spec["assignment_seed"], diagnostic=spec["diagnostic"])
        initial = initial_snapshot(env)
        tape = [env.ue_positions.copy()]
        for _ in range(spec["horizon"]):
            env._advance_mobile_ues()
            env.t += 1
            tape.append(env.ue_positions.copy())
        tape = np.asarray(tape, dtype=np.float64)
        save_npz(path, {"positions": tape})
        receipt = dict(key=key, spec=spec, position_sha256=position_hash(tape),
                       file_sha256=file_hash(path), initial=initial,
                       origin="Return-v4 source UE-only generator", environment_steps=0,
                       ue_only_ticks=spec["horizon"])
        atomic_json(meta, receipt)
        tape.flags.writeable = False
        return tape, receipt


def initial_snapshot(env):
    return dict(battery=env.battery.tolist(), residual_uav=int(env.residual_uav_idx),
                positions=env.uav_positions.tolist(), charging_station_cells=env.charging_stations.tolist())


class PairedReturnV4Env(BS5CommittedReturnEnv):
    def __init__(self, *args, trace_home, **kwargs):
        self._ue_tape, self.trace_receipt, self.trace_home = None, None, Path(trace_home)
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, options=None, assignment_seed=0, diagnostic=False):
        self._ue_tape = None
        out = super().reset(seed=seed, options=options, assignment_seed=assignment_seed, diagnostic=diagnostic)
        spec = dict(condition=deepcopy(self.condition), horizon=int(self.cfg.horizon),
                    scenario_seed=int(seed), assignment_seed=int(assignment_seed),
                    residual_battery=float(self.cfg.residual_initial_battery), diagnostic=bool(diagnostic))
        tape, receipt = TraceBank(self.trace_home, self.fixture).get(spec)
        assert initial_snapshot(self) == receipt["initial"]
        assert np.array_equal(self.ue_positions, tape[0]), "UE generator changed."
        self._ue_tape, self.trace_receipt = tape, receipt
        self.ue_positions = tape[0].copy()
        self._refresh_request_state()
        out.update(obs=self.get_obs(), state=self.get_state(), action_mask=self.get_action_mask())
        return out

    def _advance_mobile_ues(self):
        super()._advance_mobile_ues()
        if self._ue_tape is not None:
            self.ue_positions = self._ue_tape[self.t + 1].copy()

    def step(self, actions):
        t = self.t
        out = super().step(actions)
        assert np.array_equal(self.last_service_snapshot["ue_positions"], self._ue_tape[t])
        assert np.array_equal(self.ue_positions, self._ue_tape[t + 1])
        if not self.cfg.allow_relay:
            assert not out["action_mask"][:, 1::2].any() and not any(int(a) % 2 for a in actions)
            assert all(r is None for r in self.last_service_snapshot["network"]["routes"].values())
        return out


def env_for(p, spec, trace_home, arm):
    return PairedReturnV4Env(spec["condition"], allow_relay=arm != "C-N" and arm != "5A-Recovery",
                            horizon=spec["horizon"], residual_battery=spec["residual_battery"],
                            fixture=p["fixture"], trace_home=trace_home)


def full_condition(p):
    return condition_for(p["stage_ends"][2], "C-R", p)


def pack_replay(r):
    names = [f.name for f in fields(CurriculumTransition)]
    return dict(capacity=r.capacity, pos=r.pos, rng=r.rng.getstate(), stage_counts=r.stage_counts,
                arrays={k: np.stack([getattr(t, k) for t in r.data]) for k in names} if r.data else {}, count=len(r))


def unpack_replay(d):
    r = CurriculumReplay(d["capacity"])
    r.pos, r.stage_counts = d["pos"], d["stage_counts"]
    r.rng.setstate(d["rng"])
    r.data = [CurriculumTransition(**{k: v[i] for k, v in d["arrays"].items()}) for i in range(d["count"])]
    return r


def sample_audit(replay, size):
    indices = replay.rng.sample(range(len(replay.data)), size)
    items = [replay.data[i] for i in indices]
    batch = {k: np.stack([getattr(t, k) for t in items]).astype(dtype) for k, dtype in (
        ("obs", np.float32), ("state", np.float32), ("actions", np.int64), ("reward", np.float32),
        ("next_obs", np.float32), ("next_state", np.float32), ("next_action_mask", np.bool_), ("done", np.float32))}
    digest = hashlib.sha256()
    for k in sorted(batch):
        digest.update(k.encode())
        digest.update(batch[k].tobytes())
    return batch, dict(indices=indices, collection_steps=[int(t.collection_step) for t in items], sha256=digest.hexdigest())


def act(learner, out, epsilon, rng):
    with torch.inference_mode():
        q = learner.agent(torch.as_tensor(out["obs"], device=learner.device)).cpu().numpy()
    actions = []
    for i, mask in enumerate(out["action_mask"]):
        valid = np.flatnonzero(mask)
        actions.append(int(rng.choice(valid)) if rng.random() < epsilon else int(valid[np.argmax(q[i, valid])]))
    return actions


def allocate(h):
    a = {k: np.empty((h, 5, 2), np.int16) for k in ["positions_before", "positions_service", "positions_post"]}
    for phase in ["before", "service", "post"]:
        for field, dtype in [("status", np.int8), ("roles", np.int8), ("battery", np.float32)]:
            a[field + "_" + phase] = np.empty((h, 5), dtype)
    a.update(actions=np.empty((h, 5), np.int8), rates=np.empty((h, 90), np.float64),
             association=np.full((h, 90), -1, np.int8), routes=np.full((h, 5), -1, np.int8),
             mask=np.empty((h + 1, 5, 10), bool), obs=np.empty((h + 1, 5, 116), np.float32),
             state=np.empty((h + 1, 111), np.float32), rewards=np.empty(h, np.float64),
             epsilon=np.empty(h, np.float32), learning_done=np.empty(h, bool),
             terminated=np.empty(h, bool), truncated=np.empty(h, bool),
             decision_s=np.empty(h, np.float64), preview_calls=np.zeros(h, np.int16),
             stranded=np.zeros((h, 5), bool), wait=np.zeros((h, 5), bool), reversals=np.zeros((h, 5), bool))
    return a


class Budget:
    def __init__(self, path, caps=None):
        self.path = Path(path)
        if caps is not None:
            atomic_json(self.path, {k: dict(cap=n, charged=0, observed=0) for k, n in caps.items()})
        self.data = read_json(self.path)

    def reserve(self, phase, n):
        row = self.data[phase]
        assert row["charged"] + n <= row["cap"], "Physical-step cap exceeded."
        row["charged"] += n
        self.flush()

    def flush(self):
        atomic_json(self.path, self.data)


def rollout(directory, p, spec, arm, trace_home, budget, phase, choose, on_transition=None):
    """One real reset/game. What-if previews are counted separately by policies."""
    directory = Path(directory)
    assert not directory.exists(), f"Uncommitted/duplicate game {directory}; preserve all evidence."
    env = env_for(p, spec, trace_home, arm)
    start = time.monotonic()
    out = env.reset(seed=spec["scenario_seed"], assignment_seed=spec["assignment_seed"], diagnostic=spec["diagnostic"])
    initial = initial_snapshot(env)
    directory.mkdir(parents=True)
    h, a, chunks, controls = spec["horizon"], allocate(spec["horizon"]), [], []
    budget.reserve(phase, h)
    previous_old = previous_effective = None
    for t in range(h):
        for k in ["obs", "state"]:
            a[k][t] = out[k]
        a["mask"][t] = out["action_mask"]
        started = time.monotonic()
        actions, epsilon, previews = choose(env, out)
        a["decision_s"][t] = time.monotonic() - started
        assert all(out["action_mask"][i, x] for i, x in enumerate(actions)), "Invalid action."
        a["actions"][t], a["epsilon"][t], a["preview_calls"][t] = actions, epsilon, previews
        for field, attr in [("positions", "uav_positions"), ("status", "status"), ("roles", "roles"), ("battery", "battery")]:
            a[field + "_before"][t] = getattr(env, attr)
        old = env.uav_positions.copy()
        nxt = env.step(actions)
        budget.data[phase]["observed"] += 1
        assert not nxt["terminated"] and bool(nxt["truncated"]) == (t + 1 == h)
        done = bool(nxt["terminated"] or (nxt["truncated"] and spec["condition"]["stage"] != "S4"))
        a["learning_done"][t], a["terminated"][t], a["truncated"][t] = done, nxt["terminated"], nxt["truncated"]
        a["rewards"][t] = nxt["reward"]
        if on_transition is not None:
            on_transition(out, actions, nxt, done, float(epsilon))
        service = env.last_service_snapshot
        net = service["network"]
        for field, attr in [("positions", "uav_positions"), ("status", "status"), ("roles", "roles"), ("battery", "battery")]:
            a[field + "_service"][t] = service[attr]
            a[field + "_post"][t] = getattr(env, attr)
        a["rates"][t] = net["ue_rate_mbps"]
        assert abs(a["rates"][t].sum() / 360 - nxt["reward"]) < 1e-10
        for aid, users in net["association"].items():
            a["association"][t, users] = int(aid)
        for aid, rid in net["routes"].items():
            if rid is not None:
                a["routes"][t, int(aid)] = int(rid)
        d = env.last_return_control
        a["stranded"][t, d["stranded_ids_post"]] = True
        a["wait"][t, d["wait_ids"]] = True
        effective = np.asarray(d["effective_status"])
        if t:
            a["reversals"][t] = ((effective == 1) & (previous_effective == 1)
                                  & (env.uav_positions == previous_old).all(1) & (old != env.uav_positions).any(1))
        previous_old, previous_effective = old, effective
        controls.append(dict(slot=t + 1, return_control=deepcopy(d), collision=deepcopy(env.last_collision_resolution)))
        out = nxt
        if len(controls) == 250 or t + 1 == h:
            name = f"control_{len(chunks):04d}.json.gz"
            save_gz(directory / name, controls)
            chunks.append(dict(path=name, sha256=file_hash(directory / name)))
            controls = []
            budget.flush()
        if (t + 1) % 1000 == 0:
            print(json.dumps(dict(phase=phase, arm=arm, scenario=spec["scenario_seed"], slot=t + 1, horizon=h)), flush=True)
    for k in ["obs", "state"]:
        a[k][h] = out[k]
    a["mask"][h] = out["action_mask"]
    save_npz(directory / "slots.npz", a)
    index = dict(arm=arm, scenario=spec["scenario_seed"], group=phase, spec=spec, initial=initial,
                 horizon=h, condition=env.condition, resolved_config=asdict(env.cfg),
                 return_version=RETURN_VERSION, trace_key=env.trace_receipt["key"],
                 trace_sha256=env.trace_receipt["position_sha256"], data="slots.npz",
                 data_sha256=file_hash(directory / "slots.npz"), control_chunks=chunks,
                 runtime_s=time.monotonic() - start, metrics=metric(a),
                 preview_calls=int(a["preview_calls"].sum()), decision_median_ms=float(np.median(a["decision_s"]) * 1000),
                 decision_p95_ms=float(np.quantile(a["decision_s"], .95) * 1000))
    atomic_json(directory / "index.json", index)
    return index
