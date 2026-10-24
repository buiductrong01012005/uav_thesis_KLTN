"""Frozen-policy continuous600 evaluation; no training/optimizer/replay creation.

prepare --result-root AUDIT --checkpoint-root REPRO --output NEW
run --output NEW ; export --output NEW
Missing weights produce BLOCKED evidence and zero environment steps.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.bs5_bundle import (
    atomic_json,
    canonical_hash,
    file_hash,
    load_trace,
    read_json,
)
from scripts.analysis.bs5_continuous_common import (
    MODELS,
    SCENARIOS,
    contract,
    check_contract,
    compare,
    prefix_compare,
    select_validation_bests,
)
from scripts.analysis.analyze_bs5_continuous import (
    analyze,
    expected_keys,
    interval_metrics,
    read_rows,
    validate_ledger,
)
from scripts.analysis.analyze_bs5_curriculum_extend import (
    validate as validate_training,
    verify_manifest,
)
from scripts.training.run_bs5_launch_ab import (
    experiment_lock,
    freeze_source,
    serial,
    source_identity,
    step_record,
    write_trace,
)

AUDIT_SHA = "82797392e0a2febcc68f74bc64c4353c73ea540d8b191e47508058741ffa86e8"
TRAINING_SOURCE = "a2481d9c5c705745e3ec2747eedfa0fc0c2cd064de85d4da05597b967782a399"
ANALYZERS = (
    "analyze_bs5_continuous.py",
    "bs5_continuous_common.py",
    "bs5_bundle.py",
    "analyze_bs5_curriculum_bundle.py",
    "analyze_bs5_curriculum_extend.py",
    "bs5_curve_plot.py",
)


def model_digest(agent, mixer):
    h = hashlib.sha256()
    for group, model in (("agent", agent), ("mixer", mixer)):
        for name, value in sorted(model.state_dict().items()):
            h.update(f"{group}.{name}:{value.dtype}:{tuple(value.shape)}".encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(item, device, checkpoint_root=None):
    import torch
    from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialAgent
    from src.rl.qmix import QMixer

    path = Path(checkpoint_root) / item["relative_path"] if checkpoint_root else Path(item["path"])
    assert path.is_file() and file_hash(path) == item["file_sha256"], (
        f"checkpoint changed/missing: {path}"
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == {"agent", "mixer", "step"} and state["step"] == item["step"], (
        "wrong checkpoint format/step; resume.pt is not accepted"
    )
    cfg = item["resolved_config"]
    agent = EpisodeFailureSpatialAgent(5, cfg["grid_size"], cfg["coarse_obs_size"], 10, n_bs=6)
    mixer = QMixer(5, item["state_dim"])
    assert (
        agent.state_dim == item["state_dim"] == 111
        and agent.expected_obs_dim == item["obs_dim"] == 116
    )
    agent.load_state_dict(state["agent"], strict=True)
    mixer.load_state_dict(state["mixer"], strict=True)
    agent.to(device).eval()
    mixer.to(device).eval()
    for model in (agent, mixer):
        model.requires_grad_(False)
    return agent, mixer, path


def scientific_lock(training_identity=None):
    lock = read_json(ROOT / "configs/bs5_scientific_source_lock_v1.json")
    for rel, digest in lock["files"].items():
        assert file_hash(ROOT / rel) == digest, f"scientific source drift: {rel}"
        if training_identity:
            assert training_identity["files"][rel] == digest, rel


def horizon_audit():
    files = (
        "bs5_curriculum_env.py",
        "bs5_launch_ab_env.py",
        "episode_random_failure_env.py",
        "thesis_cartesian_r6_curriculum_env.py",
        "v7_r6_macro_relay_stable_env.py",
        "v4_final_fixed_map_env.py",
        "v4_fixed_mobile_battery_env.py",
    )
    reads = []
    for filename in files:
        tree = ast.parse((ROOT / "src/envs" / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "horizon"
                and isinstance(node.ctx, ast.Load)
            ):
                reads.append(dict(file=filename, line=node.lineno))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "get_obs",
                "get_state",
                "_advance_mobile_ues",
                "get_action_mask",
            }:
                assert not any(
                    isinstance(x, ast.Attribute)
                    and x.attr in {"t", "horizon"}
                    and isinstance(x.ctx, ast.Load)
                    for x in ast.walk(node)
                ), f"time/horizon-dependent policy input or physics: {filename}:{node.name}"
    assert len(reads) == 1 and reads[0]["file"] == "v4_fixed_mobile_battery_env.py", reads
    return dict(
        horizon_reads=reads,
        physics_source_locked=True,
        policy_inputs_time_free=True,
        runtime_prefix_controls="deferred until exact weights are available; within1200 cap",
    )


def retraining_context(fresh_root, result_root, training_protocol):
    """Verify a newly authorized lineage without relaxing the historical mode."""
    fresh_root, result_root = Path(fresh_root).resolve(), Path(result_root).resolve()
    assert result_root == fresh_root / "tail36k", "use this fresh experiment's own tail"
    experiment = read_json(fresh_root / "fresh_experiment.json")
    budget = read_json(fresh_root / "fresh_budget_ledger.json")
    assert experiment["origin"] == "fresh_0_to_60k_v1" and not experiment["diagnostic"]
    assert experiment["historical_imported_steps"] == budget["historical_imported_steps"] == 0
    assert (
        experiment["planned_new_training_steps"] == budget["planned_new_training_steps"] == 240000
    )
    assert budget["whole_fresh_durable_steps"] == 240000
    assert budget["whole_fresh_executed_steps"] >= 240000
    assert budget["uncertain_unflushed_steps_upper_bound"] >= 0
    policy = experiment["policy"]
    assert policy["arms"] == ["C-R", "C-N"] and policy["training_seeds"] == [551, 552]
    assert policy["parent_steps"] == 24000 and policy["total_steps"] == 60000
    assert (
        experiment["source_sha256"]
        == training_protocol["source_sha256"]
        == source_identity()["sha256"]
    )
    assert set(budget["runs"]) == set(MODELS)
    assert all(r["total_steps"] == 60000 for r in budget["runs"].values())
    registry = read_json(result_root / "run_registry.json")
    assert registry["status"] == "complete" and set(registry["runs"]) == set(MODELS)
    assert all(r["status"] == "complete" and r["steps"] == 60000 for r in registry["runs"].values())
    token = canonical_hash(
        dict(
            protocol=training_protocol,
            commits={
                n: read_json(result_root / "runs" / n / "commit.json") for n in registry["runs"]
            },
        )
    )
    receipt_path = result_root / "exports" / f"receipt_{token}.json"
    receipt = read_json(receipt_path)
    assert receipt["snapshot"] == token
    assert len(receipt["artifacts"]) == 3
    artifacts = {}
    for item in receipt["artifacts"]:
        path = (result_root / item["path"]).resolve()
        assert path.is_relative_to(result_root) and path.is_file()
        assert file_hash(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        kind = next(
            k
            for k in ("review_small", "audit_light", "repro_full")
            if path.name.startswith(k + "_")
        )
        assert kind not in artifacts
        artifacts[kind] = dict(item, absolute_path=str(path))
    assert set(artifacts) == {"review_small", "audit_light", "repro_full"}
    return dict(
        experiment=experiment,
        budget=budget,
        receipt=receipt,
        artifacts=artifacts,
        receipt_path=str(receipt_path),
    )


def prepare(
    result_root, checkpoint_root, output, *, checkpoint_map=None, device="auto", fresh_root=None
):
    import torch
    from scripts.training.run_bs5_curriculum_extend import runtime

    result_root, checkpoint_root, output = (
        Path(result_root).resolve(),
        Path(checkpoint_root).resolve(),
        Path(output).resolve(),
    )
    assert not output.exists(), "new output required; retain existing budget/output"
    scientific_lock(read_json(result_root / "source_identity.json"))
    audited_horizon = horizon_audit()
    # This is offline raw analysis, not simulation; selection is frozen before loading weights.
    training = validate_training(result_root, require_complete=True)
    tp = training["protocol"]
    assert tp["continuation"].get("origin") == "fresh_0_to_60k_v1"
    fresh = retraining_context(fresh_root, result_root, tp) if fresh_root is not None else None
    if fresh is None:
        assert tp["source_sha256"] == TRAINING_SOURCE, "historical source mismatch"
    locked_models = select_validation_bests(training["curves"]) if fresh else contract()["models"]
    audit_sha = fresh["artifacts"]["audit_light"]["sha256"] if fresh else AUDIT_SHA
    assert tp["validation_pool"] == SCENARIOS and tp["horizon"] == 150
    chosen = {}
    mapping = read_json(checkpoint_map) if checkpoint_map else {}
    assert not set(mapping) - set(MODELS)
    for name, locked in locked_models.items():
        step, utility = locked["step"], locked["validation150_utility"]
        curve = [r for r in training["curves"] if r["run"] == name]
        best = max(curve, key=lambda r: (r["utility"], -r["step"]))
        commit, meta = (
            read_json(result_root / "runs" / name / n) for n in ("commit.json", "meta.json")
        )
        assert best["step"] == commit["best_step"] == step
        assert abs(best["utility"] - utility) <= 1e-12 and best["utility"] == commit["best_utility"]
        rel = mapping.get(name, f"runs/{name}/checkpoints/{step:06d}.pt")
        path = (checkpoint_root / rel).resolve()
        assert path.is_relative_to(checkpoint_root) and Path(rel).name in (
            f"{step:06d}.pt",
            "best.pt",
        )
        chosen[name] = dict(
            step=step,
            validation150_utility=best["utility"],
            selection_rule="max mean common full-task validation150; exact ties earliest",
            selection_curve=curve,
            path=str(path),
            relative_path=rel,
            arm=meta["arm"],
            training_seed=meta["seed"],
            state_dim=meta["state_dim"],
            obs_dim=meta["obs_dim"],
            n_actions=meta["n_actions"],
            resolved_config=meta["resolved_config"],
            config_sha256=meta["config_sha256"],
            training_source_sha256=tp["source_sha256"],
            training_protocol_sha256=canonical_hash(tp),
            commit_sha256=file_hash(result_root / "runs" / name / "commit.json"),
        )
    missing = [
        dict(model=n, path=m["relative_path"], selected_step=m["step"])
        for n, m in chosen.items()
        if not Path(m["path"]).is_file()
    ]
    output.mkdir(parents=True)
    atomic_json(output / "horizon_audit.json", audited_horizon)
    atomic_json(
        output / "budget_ledger.json", dict(training_steps=0, optimizer_updates=0, attempts=[])
    )
    selection = dict(
        status="BLOCKED" if missing else "READY",
        models=chosen,
        missing_weights=missing,
        expected_repro=Path(fresh["artifacts"]["repro_full"]["path"]).name
        if fresh
        else "repro_full_6d851d3f2b66.zip",
        expected_audit_sha256=audit_sha,
        checkpoint_origin="new_authorized_fresh_0_to_60k" if fresh else "historical_fresh60k",
        historical_expectation="not applicable to new training"
        if fresh
        else "40/40 validation150 rejoined initial-residual UAV; not a600 measurement",
    )
    if not missing:
        verify_manifest(checkpoint_root)
        assert (checkpoint_root / "SHA256SUMS.json").is_file(), (
            "checkpoint provenance manifest required"
        )
        companions = read_json(checkpoint_root / "EVIDENCE_COMPANIONS.json")
        assert any(c["sha256"] == audit_sha for c in companions), "wrong fresh lineage companion"
        assert canonical_hash(read_json(checkpoint_root / "protocol.json")) == canonical_hash(tp)
        sums = read_json(checkpoint_root / "SHA256SUMS.json")
        for name, item in chosen.items():
            assert item["relative_path"] in sums
            item.update(
                file_sha256=file_hash(Path(item["path"])),
                provenance_manifest_sha256=sums[item["relative_path"]],
            )
            assert item["file_sha256"] == item["provenance_manifest_sha256"]
            assert file_hash(checkpoint_root / "runs" / name / "meta.json") == file_hash(
                result_root / "runs" / name / "meta.json"
            )
            agent, mixer, _ = load_model(item, "cpu")
            item["tensor_digest"] = model_digest(agent, mixer)
        atomic_json(
            output / "checkpoint_provenance.json", dict(manifest=sums, companions=companions)
        )
    atomic_json(output / "checkpoint_selection.json", selection)
    references = output / "references"
    references.mkdir()
    if fresh:
        for key, filename in (
            ("experiment", "fresh_experiment.json"),
            ("budget", "fresh_budget_ledger.json"),
            ("receipt", "training_export_receipt.json"),
        ):
            atomic_json(references / filename, fresh[key])
    shutil.copyfile(result_root / "protocol.json", references / "training_protocol.json")
    shutil.copyfile(
        result_root / "source_identity.json", references / "training_source_identity.json"
    )
    shutil.copyfile(
        result_root / "source_snapshot.zip", references / "training_source_snapshot.zip"
    )
    for name, item in chosen.items():
        folder = references / name
        folder.mkdir()
        for f in ("commit.json", "meta.json"):
            shutil.copyfile(result_root / "runs" / name / f, folder / f)
        for scenario in SCENARIOS:
            source = (
                result_root
                / "runs"
                / name
                / "evaluations"
                / f"{item['step']:06d}"
                / f"{scenario}.json.gz"
            )
            if fresh and item["step"] <= 24000:
                source = result_root / "parents" / source.relative_to(result_root)
            shutil.copyfile(source, folder / source.name)
    torch.set_num_threads(tp["torch_threads"])
    actual_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    assert actual_device != "cuda" or torch.cuda.is_available(), "enable Kaggle GPU"
    identity = source_identity()
    freeze_source(output, identity)
    p = contract(retrained_models=locked_models) if fresh else contract()
    p.update(
        checkpoint_selection_sha256=canonical_hash(selection),
        device=actual_device,
        runtime=runtime(),
        training_runtime=tp["execution_environment"],
        torch_threads=tp["torch_threads"],
        eval_source_sha256=identity["sha256"],
        training_protocol_sha256=canonical_hash(tp),
        evidence_files={
            f.relative_to(output).as_posix(): file_hash(f)
            for f in references.rglob("*")
            if f.is_file()
        },
        fixture=tp["fixture"],
        original_validation_residual_battery=30.0,
    )
    atomic_json(output / "protocol.json", p)
    atomic_json(
        output / "run_registry.json",
        dict(
            status=selection["status"],
            protocol_sha256=canonical_hash(p),
            trajectories={key: dict(status="not_started") for key in sorted(expected_keys())},
        ),
    )
    atomic_json(output / "controls.json", dict(status="not_started", runs={}))
    atomic_json(
        output / "verification.json",
        dict(
            accepted=False,
            status=selection["status"],
            main_steps=0,
            control_steps=0,
            training_steps=0,
            missing_weights=missing,
        ),
    )
    message = (
        "BLOCKED: thiếu đúng weights đã chọn; chưa chạy env/control/main."
        if missing
        else "READY: đã khóa checkpoint; chưa chạy env/control/main."
    )
    report = [
        "# BS5 continuous600 — preflight",
        "",
        message,
        "",
        "Best được chọn bằng validation150 của fresh60k; không chọn lại theo600.",
        "Nguồn weights: lượt train mới được cho phép; không phải checkpoint lịch sử."
        if fresh
        else "Nguồn weights: đúng lượt fresh60k lịch sử đã khóa.",
    ]
    report += [f"- {m['model']}: `{m['path']}`" for m in missing]
    report += [
        "",
        "Cần repro_full_6d851d3f2b66.zip hoặc các file trên kèm manifest/provenance của đúng fresh run. Không dùng checkpoint khác; không retrain.",
        "Budget khóa: main24000 + control1200, tổng25200; training0. Historical validation150 không phải kết quả600.",
        "Mặc định training150 và mọi source khoa học giữ nguyên. Timestamp lifecycle cũ chỉ lần đầu; analyzer mới dựng tất cả cycle từ từng transition.",
    ]
    (output / "REPORT_VI.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return selection


class Budget:
    def __init__(self, root):
        self.root = Path(root)
        self.data = read_json(self.root / "budget_ledger.json")
        validate_ledger(self.data)

    def flush(self):
        atomic_json(self.root / "budget_ledger.json", self.data)

    def start(self, key, kind):
        assert not any(a["key"] == key for a in self.data["attempts"]), (
            "incomplete attempt cannot be restarted/reset"
        )
        attempt = dict(
            key=key,
            kind=kind,
            status="running",
            executed_steps=0,
            charged_steps=0,
            uncertain_steps_upper_bound=0,
        )
        self.data["attempts"].append(attempt)
        self.flush()
        return attempt

    def reserve(self, attempt, n):
        assert n > 0 and attempt["charged_steps"] == attempt["executed_steps"]
        attempt["charged_steps"] += n
        attempt["uncertain_steps_upper_bound"] += n
        try:
            validate_ledger(self.data)
        except BaseException:
            attempt["charged_steps"] -= n
            attempt["uncertain_steps_upper_bound"] -= n
            raise
        self.flush()

    def stepped(self, attempt):
        attempt["executed_steps"] += 1
        attempt["uncertain_steps_upper_bound"] -= 1
        assert attempt["uncertain_steps_upper_bound"] >= 0


class Timing:
    def __init__(self):
        self.values = {}
        self.started = time.perf_counter()

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] = self.values.get(name, 0) + time.perf_counter() - start

    def snapshot(self):
        return dict(
            phases=self.values,
            wall=time.perf_counter() - self.started,
            semantics="phases are disjoint; wall includes overhead; do not add wall to phases",
        )


def setup_env(p, name, horizon):
    from src.envs.bs5_curriculum_env import BS5CurriculumEnv, condition_for

    # D-R selects full-task semantics immediately, not a direct training arm.
    condition = condition_for(0, "D-R", dict(stage_ends=[3000, 6000, 12000, 60000]))
    env = BS5CurriculumEnv(
        condition,
        allow_relay=name.startswith("C-R"),
        horizon=horizon,
        residual_battery=30.0,
        fixture=p["fixture"],
    )
    return env, condition


def fingerprint(env):
    return canonical_hash(
        serial(
            dict(
                positions=env.uav_positions,
                battery=env.battery,
                status=env.status,
                roles=env.roles,
                ue_positions=env.ue_positions,
                obs=env.get_obs(),
                state=env.get_state(),
                rng=env.rng.bit_generator.state,
            )
        )
    )


def trajectory(
    root,
    p,
    name,
    scenario,
    horizon,
    length,
    key,
    kind,
    budget,
    models,
    reference=None,
    control=None,
    paired=None,
    fixed_actions=False,
):
    import torch
    from src.envs.bs5_launch_ab_env import Geometry
    from scripts.training.run_bs5_launch_ab import capture_decision

    timer = Timing()
    agent, mixer, path = models[name]
    item = read_json(root / "checkpoint_selection.json")["models"][name]
    before = model_digest(agent, mixer)
    checkpoint_before = file_hash(path)
    source_before = source_identity()["sha256"]
    assert (
        before == item["tensor_digest"]
        and source_before == p["eval_source_sha256"]
        and checkpoint_before == item["file_sha256"]
    )
    env, condition = setup_env(p, name, horizon)
    config = serial(asdict(env.cfg))
    expected = dict(item["resolved_config"], horizon=horizon)
    assert config == expected, "evaluation config drift beyond horizon"
    out = env.reset(seed=scenario, assignment_seed=scenario, diagnostic=True)
    initial = serial(
        dict(
            positions=env.uav_positions,
            battery=env.battery,
            ue_positions=env.ue_positions,
            residual_uav=int(env.residual_uav_idx),
            charging_station_cells=env.charging_stations,
            seed=scenario,
            assignment_seed=scenario,
            diagnostic=True,
        )
    )
    geom = Geometry(env)
    errors = {}
    chunks = []
    rows = []
    pending = []
    original_compute = env.compute_network
    scoring_seconds = [0.0]

    def measured_compute():
        started = time.perf_counter()
        try:
            return original_compute()
        finally:
            scoring_seconds[0] += time.perf_counter() - started

    env.compute_network = measured_compute
    folder = root / ("trajectories" if kind == "main" else "control_raw") / key
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(
        folder / "reset.json",
        dict(
            initial=initial,
            reset_count=1,
            protocol_sha256=canonical_hash(p),
            model=name,
            scenario=scenario,
        ),
    )
    attempt = budget.start(key, kind)

    def flush_chunk():
        if not pending:
            return
        dest = folder / f"slots_{pending[0]['slot']:04d}_{pending[-1]['slot']:04d}.json.gz"
        with timer.phase("serialization"):
            write_trace(dest, pending)
            chunks.append(
                dict(
                    path=dest.relative_to(root).as_posix(),
                    sha256=file_hash(dest),
                    slots=len(pending),
                )
            )
            atomic_json(
                folder / "partial.json",
                dict(status="running", chunks=chunks, executed_steps=attempt["executed_steps"]),
            )
        pending.clear()
        budget.flush()

    try:
        for idx in range(length):
            if idx % 10 == 0:
                budget.reserve(attempt, min(10, length - idx))
            rng_before = canonical_hash(serial(env.rng.bit_generator.state))
            decision = serial(capture_decision(env))
            decision["ue_positions"] = env.ue_positions.tolist()
            mask = out["action_mask"].copy()
            with timer.phase("inference"):
                if fixed_actions:
                    actions = [8] * 5
                else:
                    with torch.inference_mode():
                        q = agent(torch.as_tensor(out["obs"], device=p["device"]))
                        # Same masked first-argmax tie convention as original act().
                        actions = [
                            int(
                                torch.argmax(
                                    q[i].masked_fill(
                                        ~torch.as_tensor(m, dtype=torch.bool, device=p["device"]),
                                        -torch.inf,
                                    )
                                ).item()
                            )
                            for i, m in enumerate(mask)
                        ]
                if p["device"] == "cuda":
                    torch.cuda.synchronize()
            started = time.perf_counter()
            score_before = scoring_seconds[0]
            nxt = env.step(actions)
            elapsed = time.perf_counter() - started
            score = scoring_seconds[0] - score_before
            timer.values["actual_network_scoring"] = (
                timer.values.get("actual_network_scoring", 0) + score
            )
            timer.values["env_excluding_actual_scoring"] = (
                timer.values.get("env_excluding_actual_scoring", 0) + elapsed - score
            )
            budget.stepped(attempt)
            after_rng = canonical_hash(serial(env.rng.bit_generator.state))
            pure_before = fingerprint(env)
            with timer.phase("shadow_and_record"):
                row = step_record(
                    env,
                    decision,
                    mask,
                    actions,
                    nxt,
                    geom,
                    arm=name.split("_seed")[0],
                    seed=int(name.split("_seed")[1]),
                    scenario=scenario,
                    slot=idx + 1,
                    checkpoint_step=item["step"],
                    oracle=False,
                )
                pure_after = fingerprint(env)
                assert pure_before == pure_after, "logging/shadow mutated env or RNG"
                row.update(
                    task_condition=condition,
                    next_action_mask=nxt["action_mask"].tolist(),
                    terminated=bool(nxt["terminated"]),
                    truncated=bool(nxt["truncated"]),
                    q_summary=dict(agent_forward_calls=0 if fixed_actions else 1, recurrent=False),
                    recurrent_reset=idx == 0,
                    ue_ids=list(range(90)),
                    surviving_bs_ids=[0, 1, 2, 3, 4],
                    charging_station_cells=env.charging_stations.tolist(),
                    exogenous_sha256=canonical_hash(
                        [row["service"]["ue_positions"], row["service"]["ue_demand_mbps"]]
                    ),
                    env_t_before=idx,
                    env_t_after=int(env.t),
                    env_rng_before=rng_before,
                    env_rng_after_step=after_rng,
                    env_rng_after_logging=canonical_hash(serial(env.rng.bit_generator.state)),
                    shadow_pure=True,
                    obs=out["obs"].tolist(),
                    state=out["state"].tolist(),
                    next_obs=nxt["obs"].tolist(),
                    next_state=nxt["state"].tolist(),
                    lifecycle_transitions=[
                        dict(uav=a, before=b, after=s, slot=idx + 1, phase="post")
                        for a, (b, s) in enumerate(
                            zip(row["decision"]["status"], row["post"]["status"])
                        )
                        if b != s
                    ],
                )
            rows.append(row)
            pending.append(row)
            assert row["env_t_after"] == idx + 1, "env time reset/discontinuity"
            if idx:
                assert rows[-2]["post"] == row["decision"]
                assert rows[-2]["next_action_mask"] == row["action_mask"]
                assert rows[-2]["env_rng_after_logging"] == row["env_rng_before"]
            assert not row["terminated"] and row["truncated"] == (idx + 1 == horizon)
            if paired is not None:
                assert row["exogenous_sha256"] == paired[idx]["exogenous_sha256"], (
                    "paired exogenous mismatch"
                )
                if idx == 0:
                    for field in ("positions", "battery", "ue_positions"):
                        assert row["decision"][field] == paired[0]["decision"][field], (
                            "paired reset mismatch"
                        )
            if idx < 150 and reference is not None:
                prefix_compare(row, reference[idx], errors)
            if idx < 150 and control is not None:
                prefix_compare(row, control[idx], errors)
                for f in ("obs", "state", "next_obs", "next_state"):
                    compare(row[f], control[idx][f], errors, f)
            out = nxt
            if len(pending) == 10:
                flush_chunk()
        flush_chunk()
        after = model_digest(agent, mixer)
        assert (
            before == after
            and file_hash(path) == checkpoint_before
            and source_identity()["sha256"] == source_before
        )
        metrics = interval_metrics(rows, initial, name, scenario, 1, length)[0]
        receipt = dict(
            model=name,
            scenario=scenario,
            protocol_sha256=canonical_hash(p),
            reset_count=1,
            initial=initial,
            chunks=chunks,
            tensor_before=before,
            tensor_after=after,
            checkpoint_before=checkpoint_before,
            checkpoint_after=file_hash(path),
            source_before=source_before,
            source_after=source_before,
            training_steps=0,
            optimizer_updates=0,
            metrics=metrics,
            parity_max_errors=errors,
            timing=timer.snapshot(),
            horizon=horizon,
        )
        atomic_json(folder / "receipt.json", receipt)
        atomic_json(
            folder / "partial.json", dict(status="complete", chunks=chunks, executed_steps=length)
        )
        attempt.update(
            status="complete",
            committed_steps=length,
            receipt=(folder / "receipt.json").relative_to(root).as_posix(),
            receipt_sha256=file_hash(folder / "receipt.json"),
        )
        budget.flush()
        return receipt, folder / "receipt.json"
    except BaseException as exc:
        flush_chunk()
        attempt.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", committed_steps=0)
        budget.flush()
        atomic_json(
            folder / "failure.json",
            dict(
                status="incomplete", error=attempt["error"], chunks=chunks, timing=timer.snapshot()
            ),
        )
        raise


def run(root, *, checkpoint_root=None):
    import torch
    from scripts.training.run_bs5_curriculum_extend import runtime

    root = Path(root).resolve()
    p = read_json(root / "protocol.json")
    check_contract(p)
    selection = read_json(root / "checkpoint_selection.json")
    assert selection["status"] == "READY", (
        f"BLOCKED: missing weights {selection['missing_weights']}"
    )
    assert canonical_hash(selection) == p["checkpoint_selection_sha256"]
    for rel, digest in p["evidence_files"].items():
        assert file_hash(root / rel) == digest, f"frozen evidence changed: {rel}"
    torch.set_num_threads(p["torch_threads"])
    assert runtime() == p["runtime"], (
        "runtime changed since prepare; restore original session runtime"
    )
    scientific_lock()
    assert source_identity()["sha256"] == p["eval_source_sha256"], "use frozen evaluation source"
    registry = read_json(root / "run_registry.json")
    assert registry["protocol_sha256"] == canonical_hash(p)
    with experiment_lock(root):
        budget = Budget(root)
        # Crash recovery is deliberately fail-closed, without resetting/replaying a partial game.
        assert all(a["status"] == "complete" for a in budget.data["attempts"]), (
            "incomplete attempt: no automatic reset/retry; preserve evidence and budget"
        )
        if registry["status"] == "complete":
            return export(root)
        models = {
            n: load_model(item, p["device"], checkpoint_root)
            for n, item in selection["models"].items()
        }
        controls = read_json(root / "controls.json")
        # A valid committed receipt may precede a registry rename at a crash.
        # Reconcile metadata only; never execute a committed game again.
        for attempt in budget.data["attempts"]:
            path = root / attempt["receipt"]
            assert file_hash(path) == attempt["receipt_sha256"]
            receipt = read_json(path)
            read_rows(root, receipt["chunks"])
            if attempt["kind"] == "main":
                registry["trajectories"][attempt["key"]].update(
                    status="complete",
                    receipt=attempt["receipt"],
                    receipt_sha256=attempt["receipt_sha256"],
                )
            else:
                controls["runs"][attempt["key"]] = dict(
                    model=receipt["model"], horizon=receipt["horizon"], chunks=receipt["chunks"]
                )
        atomic_json(root / "run_registry.json", registry)
        atomic_json(root / "controls.json", controls)
        try:
            for arm in ("R", "N"):
                name = f"C-{arm}_seed551"
                for horizon in (150, 600):
                    key = f"env-{arm}-{horizon}"
                    if key not in controls["runs"]:
                        receipt, _ = trajectory(
                            root,
                            p,
                            name,
                            130001,
                            horizon,
                            150,
                            key,
                            "control",
                            budget,
                            models,
                            fixed_actions=True,
                        )
                        controls["runs"][key] = dict(
                            model=name, horizon=horizon, chunks=receipt["chunks"]
                        )
                        atomic_json(root / "controls.json", controls)
                a = read_rows(root, controls["runs"][f"env-{arm}-600"]["chunks"])
                b = read_rows(root, controls["runs"][f"env-{arm}-150"]["chunks"])
                for x, y in zip(a, b):
                    prefix_compare(x, y, {})
                    for field in ("obs", "state", "next_obs", "next_state"):
                        compare(x[field], y[field], {}, field)
            for name in MODELS:
                key = "policy-" + name
                if key not in controls["runs"]:
                    receipt, _ = trajectory(
                        root, p, name, 130001, 150, 150, key, "control", budget, models
                    )
                    controls["runs"][key] = dict(model=name, horizon=150, chunks=receipt["chunks"])
                    atomic_json(root / "controls.json", controls)
            controls["status"] = "complete"
            atomic_json(root / "controls.json", controls)
            registry["status"] = "running"
            atomic_json(root / "run_registry.json", registry)
            for name in MODELS:
                control = read_rows(root, controls["runs"]["policy-" + name]["chunks"])
                for scenario in SCENARIOS:
                    key = f"{name}/{scenario}"
                    entry = registry["trajectories"][key]
                    if entry["status"] == "complete":
                        path = root / entry["receipt"]
                        assert file_hash(path) == entry["receipt_sha256"]
                        read_rows(root, read_json(path)["chunks"])
                        continue
                    entry["status"] = "running"
                    atomic_json(root / "run_registry.json", registry)
                    reference = load_trace(root / "references" / name / f"{scenario}.json.gz")[
                        "trace"
                    ]
                    paired = None
                    if name != next(iter(MODELS)):
                        other = registry["trajectories"][f"{next(iter(MODELS))}/{scenario}"]
                        paired = read_rows(root, read_json(root / other["receipt"])["chunks"])
                    receipt, path = trajectory(
                        root,
                        p,
                        name,
                        scenario,
                        600,
                        600,
                        key,
                        "main",
                        budget,
                        models,
                        reference=reference,
                        control=control if scenario == 130001 else None,
                        paired=paired,
                    )
                    entry.update(
                        status="complete",
                        receipt=path.relative_to(root).as_posix(),
                        receipt_sha256=file_hash(path),
                    )
                    atomic_json(root / "run_registry.json", registry)
                    print(
                        json.dumps(dict(trajectory=key, status="complete", steps=600)), flush=True
                    )
            registry["status"] = "complete"
            atomic_json(root / "run_registry.json", registry)
            return export(root)
        except BaseException as exc:
            registry["status"] = "incomplete"
            registry["error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(root / "run_registry.json", registry)
            atomic_json(
                root / "verification.json",
                dict(
                    accepted=False, status="INCOMPLETE", error=registry["error"], training_steps=0
                ),
            )
            raise


def export(root):
    root = Path(root).resolve()
    dest = root.parent / (root.name + "_analysis")
    target = root.parent / (root.name + "_audit_light.zip")
    receipt_path = root.parent / (root.name + "_export_receipt.json")

    def evidence_hash():
        return canonical_hash(
            {
                f.relative_to(root).as_posix(): file_hash(f)
                for f in root.rglob("*")
                if f.is_file() and f.name != "run.lock"
            }
        )

    assert source_identity()["sha256"] == read_json(root / "protocol.json")["eval_source_sha256"], (
        "use frozen evaluation source"
    )
    if receipt_path.exists():
        old = read_json(receipt_path)
        assert target.is_file() and file_hash(target) == old["sha256"] and old["accepted"]
        assert evidence_hash() == old["evidence_sha256"], "evidence changed after final export"
        return read_json(dest / "verification.json")
    timer = Timing()
    with timer.phase("analysis"):
        result = analyze(root, dest, require_complete=True)
    atomic_json(root / "verification.json", result)
    shutil.copyfile(dest / "REPORT_VI.md", root / "REPORT_VI.md")
    # No accumulated archives during simulation. Final export is explicit/idempotent.
    tmp = target.with_suffix(".zip.tmp")
    files = {
        f.relative_to(root).as_posix(): f
        for f in root.rglob("*")
        if f.is_file() and f.name != "run.lock"
    }
    files.update(
        {"analysis/" + f.relative_to(dest).as_posix(): f for f in dest.rglob("*") if f.is_file()}
    )
    files.update({n: ROOT / "scripts/analysis" / n for n in ANALYZERS})
    sums = {n: file_hash(f) for n, f in files.items()}
    with timer.phase("export"):
        with zipfile.ZipFile(tmp, "w") as z:
            for n, f in sorted(files.items()):
                z.write(
                    f,
                    n,
                    compress_type=zipfile.ZIP_STORED
                    if f.suffix in (".gz", ".zip")
                    else zipfile.ZIP_DEFLATED,
                )
            z.writestr(
                "SHA256SUMS.json", json.dumps(sums, indent=2), compress_type=zipfile.ZIP_DEFLATED
            )
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None
        tmp.replace(target)
    # Audit the bytes actually packaged using standalone stdlib modules in a clean extraction.
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory(prefix="bs5_continuous_audit_") as scratch:
        folder = Path(scratch) / "evidence"
        with zipfile.ZipFile(target) as z:
            z.extractall(folder)
        with timer.phase("packaged_offline_reaudit"):
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(folder / "analyze_bs5_continuous.py"),
                    str(folder),
                    "--output",
                    str(Path(scratch) / "analysis"),
                    "--require-complete",
                ],
                check=True,
            )
    atomic_json(
        receipt_path,
        dict(
            path=str(target),
            sha256=file_hash(target),
            bytes=target.stat().st_size,
            timing=timer.snapshot(),
            accepted=result["accepted"],
            evidence_sha256=evidence_hash(),
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "export"])
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-map", type=Path)
    parser.add_argument(
        "--fresh-root",
        type=Path,
        help="explicit new authorized fresh-training lineage; never substitutes historical weights",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=[600], default=600)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if args.command == "prepare":
        assert args.result_root
        result = prepare(
            args.result_root,
            args.checkpoint_root or args.result_root,
            args.output,
            checkpoint_map=args.checkpoint_map,
            device=args.device,
            fresh_root=args.fresh_root,
        )
        print(
            json.dumps(
                dict(status=result["status"], missing_weights=result["missing_weights"]), indent=2
            )
        )
    elif args.command == "run":
        run(args.output, checkpoint_root=args.checkpoint_root)
    else:
        export(args.output)


if __name__ == "__main__":
    main()
