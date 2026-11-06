"""Offline/synthetic tests only: these tests spend ZERO real environment steps."""

from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.analysis.bs5_bundle import atomic_json, file_hash
from scripts.analysis.bs5_continuous_common import (
    MODELS,
    busy_relays,
    contract,
    lifecycle,
    prefix_compare,
)
from scripts.analysis.analyze_bs5_continuous import (
    analyze,
    read_rows,
    validate_ledger,
    validate_rows,
)
from scripts.evaluation import eval_bs5_continuous as runner

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def forbid_real_env_steps(monkeypatch):
    from src.envs.v4_fixed_mobile_battery_env import V4FixedMobileBatteryEnv

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test must not consume a real env step")

    monkeypatch.setattr(V4FixedMobileBatteryEnv, "step", forbidden)


def reference():
    with gzip.open(FIXTURES / "bs5_continuous_reference150.json.gz", "rt", encoding="utf-8") as f:
        packet = json.load(f)
    return packet


def adapted_reference():
    rows = reference()["trace"]
    for i, r in enumerate(rows):
        r["decision"]["ue_positions"] = deepcopy(r["service"]["ue_positions"])
        r.update(
            env_t_before=i,
            env_t_after=i + 1,
            shadow_pure=True,
            env_rng_before="offline-fixture",
            env_rng_after_step="offline-fixture",
            env_rng_after_logging="offline-fixture",
            obs=[[i]],
            state=[i],
            next_obs=[[i + 1]],
            next_state=[i + 1],
            lifecycle_transitions=[
                dict(uav=a, before=b, after=s, slot=i + 1, phase="post")
                for a, (b, s) in enumerate(zip(r["decision"]["status"], r["post"]["status"]))
                if b != s
            ],
        )
        r["shadow"].pop("role_enumeration", None)
    return rows


def test_prefix_frozen_tolerance_actions_exact_and_physical_max_error():
    old = reference()["trace"][-1]
    new = deepcopy(old)
    new["truncated"] = False
    new["decision"]["battery"][0] += 5e-7
    errors = {}
    prefix_compare(new, old, errors)
    assert 0 < errors["decision.battery[]"] < 1e-6
    new["raw_actions"][0] = (new["raw_actions"][0] + 2) % 10
    with pytest.raises(AssertionError, match="mismatch"):
        prefix_compare(new, old, {})
    new = deepcopy(old)
    new["truncated"] = False
    new["reward"] += 1e-3
    with pytest.raises(AssertionError, match="reward"):
        prefix_compare(new, old, {})


def test_raw_validator_missing_duplicate_corrupt_and_no_relay_masks():
    cfg = json.loads((FIXTURES / "bs5_continuous_meta.json").read_text())["resolved_config"]
    rows = adapted_reference()
    validate_rows(rows, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    for damaged in (rows[:-1], rows[:20] + rows[19:]):
        with pytest.raises(AssertionError, match="missing/duplicate"):
            validate_rows(
                damaged, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150
            )
    bad = deepcopy(rows)
    bad[149]["env_t_after"] = 1
    with pytest.raises(AssertionError):
        validate_rows(bad, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    bad = deepcopy(rows)
    bad[20]["service"]["network"]["ue_rate_mbps"][0] = 400
    with pytest.raises(AssertionError):
        validate_rows(bad, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    # Reject a purported C-N rollout carrying the C-R masks/actions, even if rates are zeroed.
    bad = deepcopy(rows)
    for r in bad:
        r["arm"] = "C-N"
    with pytest.raises(AssertionError):
        validate_rows(bad, name="C-N_seed551", scenario=130001, config=cfg, horizon=150, length=150)


def test_continuous_rows_without_legacy_flag_still_audit_failed_bs_capacities():
    from scripts.analysis.analyze_bs5_curriculum_extend import gateway_auditor

    cfg = json.loads((FIXTURES / "bs5_continuous_meta.json").read_text())["resolved_config"]
    rows = adapted_reference()
    for row in rows:
        row.pop("failed_bs_gateway_excluded", None)
    before = deepcopy(rows)
    validate_rows(rows, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    assert rows == before, "reader must not fabricate metadata or change raw evidence"
    # Legacy training validation remains strict about its own schema.
    with pytest.raises(KeyError, match="failed_bs_gateway_excluded"):
        gateway_auditor(cfg)(rows[0])
    bad = deepcopy(rows)
    caps = next(
        r["service"]["network"]["direct_caps_mbps"]
        for r in bad
        if r["service"]["network"]["direct_caps_mbps"]
    )
    caps[next(iter(caps))] += 1.0
    with pytest.raises(AssertionError, match="gateway capacity"):
        validate_rows(bad, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    bad = deepcopy(rows)
    bad[0]["failed_bs_gateway_excluded"] = False
    with pytest.raises(AssertionError):
        validate_rows(bad, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)


def test_extracted_real_reference_rows_keep_all_physical_and_continuity_checks(tmp_path):
    from scripts.analysis.analyze_bs5_continuous import EvidenceTransport

    cfg = json.loads((FIXTURES / "bs5_continuous_meta.json").read_text())["resolved_config"]
    rows = adapted_reference()
    for row in rows:
        row.pop("failed_bs_gateway_excluded", None)
    path = tmp_path / "slots_0001_0150.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    transport = EvidenceTransport(tmp_path, allow_extracted=True)
    index = [dict(path="slots_0001_0150.json.gz", sha256="missing-old-container", slots=150)]
    decoded = read_rows(tmp_path, index, transport=transport)
    validate_rows(decoded, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150)
    assert decoded == rows and transport.unverified
    decoded[7]["env_t_after"] = 2
    with pytest.raises(AssertionError):
        validate_rows(
            decoded, name="C-R_seed551", scenario=130001, config=cfg, horizon=150, length=150
        )


def test_lifecycle_multiple_cycles_service_next_slot_and_censoring():
    events = [(0, 1), (1, 2), (2, 0), (0, 0), (0, 1), (1, 2), (2, 0), (0, 0), (0, 1)]
    rows = []
    for slot, (before, after) in enumerate(events, 1):
        statuses = [before, 0, 0, 0, 0]
        post = [after, 0, 0, 0, 0]
        rate = 4.0 if slot in (3, 4, 5, 7) else 0.0
        rows.append(
            dict(
                slot=slot,
                decision=dict(status=statuses),
                post=dict(status=post, positions=[[0, 0]] * 5),
                service=dict(
                    status=statuses,
                    roles=[0] * 5,
                    uav_positions=[[0, 0]] * 5,
                    network=dict(
                        association={"0": [0]}, routes={"0": None}, ue_rate_mbps=[rate] + [0.0] * 89
                    ),
                ),
            )
        )
    cycles, uavs = lifecycle(
        rows, dict(battery=[100, 100, 100, 100, 30], residual_uav=4), run="SYNTHETIC", scenario=0
    )
    assert len(cycles) == 3
    first, second, third = cycles
    assert first["rejoin_active_slot"] == 3 and first["useful_service_return_slot"] == 4
    assert first["useful_service_slots"] == 2 and first["service_latency_slots"] == 1
    assert second["useful_service_return_slot"] is None and second["useful_service_censored"]
    assert second["censor_reason"] == "next_return" and second["time_at_risk_slots"] == 2
    assert third["rejoin_active_slot"] is None and third["censor_reason"] == "horizon"
    assert third["service_latency_slots"] is None and not third["complete_charge_cycle"]
    assert uavs[0]["cycles_started"] == 3 and uavs[0]["useful_cycles_censored"] == 2
    assert uavs[0]["initial_group"] == "initial-full"


def test_busy_relay_needs_delivered_traffic_not_allocation():
    row = dict(
        service=dict(
            network=dict(
                routes={"1": 0},
                association={"1": [0]},
                ue_rate_mbps=[0.0] * 90,
                relay_allocations_mbps={"0": {"1": 100}},
            )
        )
    )
    assert busy_relays(row) == set()
    row["service"]["network"]["ue_rate_mbps"][0] = 1e-5
    assert busy_relays(row) == {0}


def test_budget_reservations_cap_and_no_restart(tmp_path):
    atomic_json(
        tmp_path / "budget_ledger.json", dict(training_steps=0, optimizer_updates=0, attempts=[])
    )
    budget = runner.Budget(tmp_path)
    attempt = budget.start("control", "control")
    budget.reserve(attempt, 1200)
    for _ in range(1199):
        budget.stepped(attempt)
    budget.flush()
    assert runner.Budget(tmp_path).data["attempts"][0]["uncertain_steps_upper_bound"] == 1
    with pytest.raises(AssertionError, match="cannot be restarted"):
        budget.start("control", "control")
    budget.stepped(attempt)
    with pytest.raises(AssertionError):
        budget.reserve(attempt, 1)
    assert attempt["charged_steps"] == 1200
    bad = deepcopy(budget.data)
    bad["training_steps"] = 1
    with pytest.raises(AssertionError):
        validate_ledger(bad)


def test_strict_inference_checkpoint_hash_and_no_update(tmp_path):
    from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialAgent
    from src.rl.qmix import QMixer

    cfg = json.loads((FIXTURES / "bs5_continuous_meta.json").read_text())["resolved_config"]
    agent = EpisodeFailureSpatialAgent(5, 20, 5, 10)
    mixer = QMixer(5, 111)
    path = tmp_path / "057000.pt"
    torch.save(dict(agent=agent.state_dict(), mixer=mixer.state_dict(), step=57000), path)
    item = dict(
        path=str(path),
        file_sha256=file_hash(path),
        step=57000,
        resolved_config=cfg,
        state_dim=111,
        obs_dim=116,
    )
    loaded, mixed, _ = runner.load_model(item, "cpu")
    before = runner.model_digest(loaded, mixed)
    with torch.inference_mode():
        loaded(torch.zeros((5, 116)))
    assert runner.model_digest(loaded, mixed) == before and not loaded.training
    assert all(p.grad is None and not p.requires_grad for p in loaded.parameters())
    state = torch.load(path, weights_only=True)
    state["agent"].pop(next(iter(state["agent"])))
    torch.save(state, path)
    item["file_sha256"] = file_hash(path)
    with pytest.raises(RuntimeError):
        runner.load_model(item, "cpu")
    state["step"] = 60000
    torch.save(state, path)
    item["file_sha256"] = file_hash(path)
    with pytest.raises(AssertionError, match="wrong checkpoint"):
        runner.load_model(item, "cpu")


def test_chunk_hash_and_completeness_fail_closed(tmp_path):
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump([dict(slot=1)], f)
    entry = dict(path=path.name, sha256=file_hash(path), slots=1)
    assert len(read_rows(tmp_path, [entry])) == 1
    with pytest.raises(AssertionError, match="duplicate chunk"):
        read_rows(tmp_path, [entry, entry])
    path.write_bytes(b"corrupt")
    with pytest.raises(AssertionError, match="corrupt/missing"):
        read_rows(tmp_path, [entry])
    atomic_json(tmp_path / "protocol.json", contract())
    atomic_json(
        tmp_path / "run_registry.json",
        dict(status="complete", protocol_sha256="wrong", trajectories={}),
    )
    atomic_json(tmp_path / "checkpoint_selection.json", dict(status="READY", models={}))
    with pytest.raises(AssertionError):
        analyze(tmp_path, tmp_path.parent / "out", require_complete=True)


def test_standalone_import_without_torch_numpy_or_environment(tmp_path):
    for name in runner.ANALYZERS:
        (tmp_path / name).write_bytes((runner.ROOT / "scripts/analysis" / name).read_bytes())
    script = """import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','src','matplotlib'}: raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
import analyze_bs5_continuous
print('OFFLINE_IMPORT_OK')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "OFFLINE_IMPORT_OK" in result.stdout


@pytest.mark.parametrize("interrupt", [False, True])
def test_mock_runner_one_unbroken600_no_resets_at_window_boundaries(
    tmp_path, monkeypatch, interrupt
):
    """Synthetic object exercises orchestration; it is NOT an env rollout or result."""
    from src.envs.bs5_launch_ab_env import BS5LaunchConfig
    import src.envs.bs5_launch_ab_env as launch
    import scripts.training.run_bs5_launch_ab as original

    cfg = BS5LaunchConfig()
    name = "C-R_seed551"

    class Fake:
        def __init__(self):
            self.cfg = deepcopy(cfg)
            self.cfg.horizon = 600
            self.t = 0
            self.resets = 0
            self.rng = np.random.default_rng(7)
            self.residual_uav_idx = 1
            self.uav_positions = np.array([[14, 8], [13, 8], [15, 8], [14, 7], [14, 9]])
            self.roles = np.zeros(5, dtype=int)
            self.status = np.zeros(5, dtype=int)
            self.battery = np.array([100.0, 30.0, 100.0, 100.0, 100.0])
            self.ue_positions = np.zeros((90, 2))
            self.charging_stations = np.array([[1, 1], [18, 1], [10, 18]])

        def get_obs(self):
            return np.full((5, 116), self.t, dtype=np.float32)

        def get_state(self):
            return np.full(111, self.t, dtype=np.float32)

        def output(self):
            return dict(
                obs=self.get_obs(),
                state=self.get_state(),
                action_mask=np.ones((5, 10), dtype=int),
                reward=0.0,
                terminated=False,
                truncated=self.t == 600,
            )

        def reset(self, **kwargs):
            self.resets += 1
            assert self.t == 0
            return self.output()

        def compute_network(self):
            return dict(
                utility=0.0,
                ue_rate_mbps=[0.0] * 90,
                association={str(i): [] for i in range(5)},
                routes={str(i): None for i in range(5)},
                backhaul_caps_mbps={str(i): 0.0 for i in range(5)},
                relay_gateway_caps_mbps={},
                relay_allocations_mbps={},
            )

        def step(self, actions):
            if interrupt and self.t == 150:
                raise RuntimeError("synthetic interruption")
            self.snapshot = runner.serial(
                dict(
                    status=self.status,
                    roles=self.roles,
                    uav_positions=self.uav_positions,
                    battery=self.battery,
                    ue_positions=self.ue_positions,
                    ue_demand_mbps=[4.0] * 90,
                    network=self.compute_network(),
                )
            )
            self.t += 1
            self.rng.random()
            self.ue_positions[:] = self.t / 1000
            return self.output()

    fake = Fake()
    agent = torch.nn.Linear(116, 10).eval().requires_grad_(False)
    mixer = torch.nn.Linear(111, 1).eval().requires_grad_(False)
    cp = tmp_path / "synthetic_weights"
    cp.write_bytes(b"NOT A CHECKPOINT")
    item = dict(
        step=57000,
        tensor_digest=runner.model_digest(agent, mixer),
        file_sha256=file_hash(cp),
        resolved_config=runner.serial(asdict(cfg)),
    )
    atomic_json(tmp_path / "checkpoint_selection.json", dict(models={name: item}))
    atomic_json(
        tmp_path / "budget_ledger.json", dict(training_steps=0, optimizer_updates=0, attempts=[])
    )
    p = dict(eval_source_sha256="synthetic", device="cpu")
    monkeypatch.setattr(runner, "source_identity", lambda: dict(sha256="synthetic"))
    monkeypatch.setattr(
        runner, "setup_env", lambda *args: (fake, dict(stage="S4", battery=True, mobility=True))
    )
    monkeypatch.setattr(launch, "Geometry", lambda env: None)

    def record(env, decision, mask, actions, out, geometry, **kwargs):
        net = env.snapshot["network"]
        return dict(
            arm="C-R",
            seed=551,
            scenario_seed=130001,
            slot=env.t,
            decision=deepcopy(decision),
            service=deepcopy(env.snapshot),
            post=runner.serial(
                dict(
                    positions=env.uav_positions,
                    status=env.status,
                    roles=env.roles,
                    battery=env.battery,
                    ue_positions=env.ue_positions,
                )
            ),
            raw_actions=actions,
            decoded_actions=[[a // 2, a % 2] for a in actions],
            action_mask=mask.tolist(),
            reward=0.0,
            shadow={k: deepcopy(net) for k in ("actual", "same_access_direct", "all_access")},
            relay_required_ue=[False] * 90,
            uav_in_relay_required_access_region=[False] * 5,
            nearest_station_steps=[0] * 5,
        )

    monkeypatch.setattr(runner, "step_record", record)
    # The direct import of capture_decision continues to use the real snapshot helper.
    assert original.capture_decision(fake)["status"].tolist() == [0] * 5
    if interrupt:
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            runner.trajectory(
                tmp_path,
                p,
                name,
                130001,
                600,
                600,
                "SYNTHETIC600",
                "main",
                runner.Budget(tmp_path),
                {name: (agent, mixer, cp)},
            )
        ledger = runner.Budget(tmp_path)
        a = ledger.data["attempts"][0]
        assert a["status"] == "incomplete" and a["executed_steps"] == 150
        assert a["charged_steps"] == 160 and a["uncertain_steps_upper_bound"] == 10
        assert fake.resets == 1 and fake.t == 150
        with pytest.raises(AssertionError, match="cannot be restarted"):
            ledger.start("SYNTHETIC600", "main")
        return
    receipt, _ = runner.trajectory(
        tmp_path,
        p,
        name,
        130001,
        600,
        600,
        "SYNTHETIC600",
        "main",
        runner.Budget(tmp_path),
        {name: (agent, mixer, cp)},
    )
    assert fake.resets == 1 and fake.t == 600
    rows = read_rows(tmp_path, receipt["chunks"])
    assert len(rows) == 600 and len(receipt["chunks"]) == 60
    for boundary in (150, 300, 450):
        assert not rows[boundary - 1]["truncated"] and rows[boundary]["env_t_before"] == boundary
        assert rows[boundary - 1]["post"] == rows[boundary]["decision"]
    assert rows[-1]["truncated"] and receipt["tensor_before"] == receipt["tensor_after"]
    assert receipt["training_steps"] == receipt["optimizer_updates"] == 0


def test_horizon_static_audit_and_locked_config():
    audit = runner.horizon_audit()
    assert audit["policy_inputs_time_free"]
    assert (
        json.loads((runner.ROOT / "configs/bs5_continuous_eval600_v1.json").read_text())
        == contract()
    )


def test_blocked_run_stops_before_models_or_environment(tmp_path, monkeypatch):
    atomic_json(tmp_path / "protocol.json", contract())
    atomic_json(
        tmp_path / "checkpoint_selection.json",
        dict(
            status="BLOCKED",
            missing_weights=[f"runs/{n}/checkpoints/{s:06d}.pt" for n, (s, _) in MODELS.items()],
        ),
    )
    monkeypatch.setattr(
        runner, "load_model", lambda *a: pytest.fail("must not load replacement weights")
    )
    with pytest.raises(AssertionError, match="BLOCKED: missing weights"):
        runner.run(tmp_path)


def test_kaggle_notebook_discovers_extracted_source_and_stops_missing_weights(
    tmp_path, monkeypatch
):
    notebook = json.loads(
        (runner.ROOT / "scripts/notebooks/bs5_continuous_eval600_kaggle.ipynb").read_text()
    )
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 4
    for c in cells:
        compile(c, "notebook", "exec")
    inputs = tmp_path / "input"
    work = tmp_path / "working"
    work.mkdir()
    source = inputs / "source/uav_thesis/scripts/evaluation"
    source.mkdir(parents=True)
    (source / "eval_bs5_continuous.py").write_text("# source-only test")
    audit = inputs / "audit"
    (audit / "runs/C-R_seed551/evaluations/057000").mkdir(parents=True)
    (audit / "runs/C-R_seed551/evaluations/057000/130001.json.gz").write_bytes(b"discovery stub")
    atomic_json(audit / "protocol.json", dict(continuation=dict(origin="fresh_0_to_60k_v1")))
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        assert "prepare" in args and "run" not in args
        out = work / "bs5_continuous600"
        out.mkdir()
        atomic_json(out / "protocol.json", contract())
        atomic_json(
            out / "checkpoint_selection.json", dict(status="BLOCKED", missing_weights=list(MODELS))
        )

    monkeypatch.setattr(subprocess, "run", command)
    monkeypatch.chdir(tmp_path)
    first = (
        cells[0]
        .replace("'/kaggle/input'", repr(str(inputs)))
        .replace("'/kaggle/working'", repr(str(work)))
    )
    namespace = {}
    exec(first, namespace)
    with pytest.raises(AssertionError, match="No control/main steps"):
        exec(cells[1], namespace)
    assert len(calls) == 1 and "prepare" in calls[0]
