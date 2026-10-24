"""Bounded observation of the two real runners; never invokes their matrices."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import zipfile

# Child processes receive these before interpreter/Torch/CUDA initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analysis.analyze_bs5_prefix_verify import compare, lines, read, write
from scripts.analysis.bs5_bundle import canonical_hash, file_hash


def digest(value):
    """Exact typed, structured hash; reading state never advances a generator."""
    h = hashlib.sha256()

    def visit(x):
        if torch.is_tensor(x):
            visit(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=str):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        else:
            h.update((type(x).__name__ + ":" + repr(x)).encode())

    visit(value)
    return h.hexdigest()


def numeric_error(reference, value, tolerance):
    a, b = np.asarray(reference), np.asarray(value)
    if a.shape != b.shape:
        raise ValueError(f"numeric shape differs: {a.shape}, {b.shape}")
    a, b = a.astype(np.float64), b.astype(np.float64)
    delta = np.abs(a - b)
    return dict(
        max_abs=float(delta.max(initial=0)),
        max_rel=float((delta / np.maximum(np.abs(a), 1e-12)).max(initial=0)),
        within_tolerance=bool(
            np.isfinite(a).all()
            and np.isfinite(b).all()
            and (delta <= tolerance["atol"] + tolerance["rtol"] * np.abs(a)).all()
        ),
    )


class Budget:
    """Reserve entire trials durably before launching; abandoned reservations count."""

    def __init__(self, root):
        self.path = Path(root) / "budget_ledger.json"

    def reserve(self, label, steps, implementation, kind):
        data = read(self.path)
        assert label not in data["trials"], "trial exists; never reset a budget reservation"
        assert 0 < steps <= 3000
        assert (
            sum(t["reserved_steps"] for t in data["trials"].values()) + steps
            <= data["hard_cap_train_steps"]
        ), "hard budget cap"
        data["trials"][label] = dict(
            reserved_steps=steps,
            consumed_steps=0,
            implementation=implementation,
            kind=kind,
            status="reserved",
            reserved_utc=time.time(),
        )
        write(self.path, data)

    def claim_step(self, label):
        data = read(self.path)
        trial = data["trials"][label]
        assert trial["consumed_steps"] < trial["reserved_steps"], "trial budget exhausted"
        assert (
            sum(t["consumed_steps"] for t in data["trials"].values()) < data["hard_cap_train_steps"]
        )
        # Charge BEFORE the environment step; interrupted in-flight calls remain charged.
        trial["consumed_steps"] += 1
        trial["status"] = "running"
        write(self.path, data)

    def finish(self, label, status):
        data = read(self.path)
        data["trials"][label]["status"] = status
        write(self.path, data)


def configure(device):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA unavailable; explicitly select CPU for CPU-only equivalence"
        )
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        torch.cuda.init()
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        driver = None
    return dict(
        python=sys.version,
        numpy=np.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        os=platform.platform(),
        device=device,
        dtype="float32",
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
        driver=driver,
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        tf32=False,
        matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        relative_error_denominator_floor=1e-12,
    )


class Observer:
    """Wrap original calls once; never samples/recomputes a training forward."""

    def __init__(self, root, label, module, runtime, reference=None):
        self.root, self.label, self.module = Path(root), label, module
        self.directory = self.root / "trials" / label / "diagnostic"
        self.directory.mkdir(parents=True)
        self.config = read(self.root / "config.json")
        self.protocol = read(self.root / "trials" / label / "protocol.json")
        self.runtime, self.reference = runtime, reference
        self.budget = Budget(root)
        self.step, self.update, self.phase = 0, 0, "setup"
        self.learner = self.replay = self.exploration = None
        self.pending, self.tensors, self.eval_rows = {}, {}, []
        self.syncs, self.clears = [], 0
        self.streams = {
            n: gzip.open(self.directory / f"{n}.jsonl.gz", "wt", encoding="utf-8")
            for n in ("steps", "updates", "evals")
        }
        self.reference_steps = list(lines(Path(reference) / "steps.jsonl.gz")) if reference else []
        self.reference_updates = (
            list(lines(Path(reference) / "updates.jsonl.gz")) if reference else []
        )
        self.started = time.monotonic()

    def emit(self, stream, row):
        self.streams[stream].write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.streams[stream].flush()

    def rng(self):
        return dict(
            python=digest(random.getstate()),
            numpy=digest(np.random.get_state()),
            torch=digest(torch.get_rng_state()),
            cuda=digest(torch.cuda.get_rng_state_all())
            if self.runtime["device"] == "cuda"
            else None,
            exploration=digest(self.exploration.bit_generator.state)
            if self.exploration is not None
            else None,
            replay=digest(self.replay.rng.getstate()) if self.replay is not None else None,
        )

    def state(self):
        learner = self.learner
        return dict(
            models={
                k: digest(getattr(learner, k).state_dict())
                for k in ("agent", "mixer", "target_agent", "target_mixer")
            },
            optimizer=digest(learner.optimizer.state_dict()),
            rng=self.rng(),
            replay_rng=digest(self.replay.rng.getstate()),
            replay_size=len(self.replay),
            lineage=0,
        )

    def numeric(self, arrays, row, kind):
        row["numeric_hashes"] = {k: digest(v) for k, v in arrays.items()}
        row["errors"] = {}
        if kind == "step":
            row["q_values"] = arrays["q"].tolist()
            if self.reference:
                values = self.reference_steps[self.step - 1]["q_values"]
                row["errors"]["q"] = numeric_error(
                    values, arrays["q"], self.config["numerical_tolerance"]
                )
            return
        number = self.update
        path = self.directory / "numeric" / f"{number:06d}.npz"
        if self.reference:
            with np.load(
                Path(self.reference) / "numeric" / f"{number:06d}.npz", allow_pickle=False
            ) as previous:
                assert set(previous.files) == set(arrays)
                for key, value in arrays.items():
                    row["errors"][key] = numeric_error(
                        previous[key], value, self.config["numerical_tolerance"]
                    )
            # Full tensors only at a mismatch and milestones; no optimizer in review.
            changed = row["numeric_hashes"] != self.reference_updates[number - 1]["numeric_hashes"]
            keep = (changed and not hasattr(self, "witness_saved")) or self.step in (
                1000,
                1004,
                1500,
                3000,
            )
            if changed:
                self.witness_saved = True
        else:
            keep = True  # Reference for B and the one permitted self-repeat.
        if keep:
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **arrays)

    def install(self):
        from src.rl.bs5_curriculum_qmix import BS5CurriculumQMIX
        from src.rl.bs5_curriculum_replay import CurriculumReplay
        from scripts.training import run_bs5_curriculum_pilot as old
        from scripts.training import run_bs5_direct_replay_reset as new

        observer = self
        init = BS5CurriculumQMIX.__init__

        def initialize(learner, *args, **kwargs):
            init(learner, *args, **kwargs)
            observer.learner = learner
            for name in ("agent", "mixer", "target_agent", "target_mixer"):

                def capture(module, inputs, output, key=name):
                    if observer.phase in ("act", "update"):
                        observer.tensors.setdefault(key, []).append(output.detach())

                getattr(learner, name).register_forward_hook(capture)
            original_step = learner.optimizer.step

            def optimizer_step(*a, **kw):
                params = list(learner.agent.parameters()) + list(learner.mixer.parameters())
                observer.grad_after = np.concatenate(
                    [p.grad.detach().cpu().numpy().ravel() for p in params if p.grad is not None]
                )
                return original_step(*a, **kw)

            learner.optimizer.step = optimizer_step

        BS5CurriculumQMIX.__init__ = initialize
        replay_init, add, sample = (
            CurriculumReplay.__init__,
            CurriculumReplay.add,
            CurriculumReplay.sample,
        )

        def initialize_replay(replay, *a, **kw):
            replay_init(replay, *a, **kw)
            if observer.replay is not None:
                observer.clears += 1
                raise AssertionError("replay clear forbidden in prefix")
            observer.replay = replay

        CurriculumReplay.__init__ = initialize_replay
        draw = random.Random.sample

        def observed_draw(rng, population, size, **kwargs):
            selected = draw(rng, population, size, **kwargs)
            if observer.replay is not None and rng is observer.replay.rng:
                observer.sampled = selected
            return selected

        # Class wrapper leaves replay.rng picklable and preserves original draws.
        random.Random.sample = observed_draw

        def add_observed(replay, transition):
            add(replay, transition)
            observer.pending.update(
                next_obs=digest(transition.next_obs),
                next_state=digest(transition.next_state),
                next_mask=transition.next_action_mask.tolist(),
                reward=float(transition.reward),
                done=float(transition.done),
                replay_size=len(replay),
                lineage=transition.lineage,
            )

        CurriculumReplay.add = add_observed

        def sample_observed(replay, size):
            before = digest(replay.rng.getstate())
            batch = sample(replay, size)  # exactly one original sample
            observer.batch = dict(
                ids=[x.collection_step for x in observer.sampled],
                current_masks=digest(np.stack([x.current_action_mask for x in observer.sampled])),
                batch=digest(batch),
                batch_fields={k: digest(v) for k, v in batch.items()},
                rng_before=before,
                rng_after=digest(replay.rng.getstate()),
            )
            return batch

        CurriculumReplay.sample = sample_observed
        original_act = old.act

        def act(learner, out, epsilon, rng):
            if observer.phase == "eval":
                return original_act(learner, out, epsilon, rng)
            observer.budget.claim_step(observer.label)
            observer.step += 1
            observer.exploration = rng
            observer.pending = dict(
                step=observer.step,
                episode=(observer.step - 1) // observer.protocol["horizon"],
                slot=(observer.step - 1) % observer.protocol["horizon"] + 1,
                epsilon=epsilon,
                obs=digest(out["obs"]),
                state=digest(out["state"]),
                mask=out["action_mask"].tolist(),
                rng_before=observer.rng(),
            )
            observer.phase, observer.tensors = "act", {}
            result = original_act(learner, out, epsilon, rng)
            q = observer.tensors["agent"][0].cpu().numpy()
            observer.pending["actions"] = result[0]
            observer.numeric({"q": q}, observer.pending, "step")
            margins = []
            for values, mask in zip(q, out["action_mask"]):
                ordered = np.sort(values[mask])
                margins.append(float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else None)
            observer.pending["argmax_margins"] = margins
            observer.phase = "train"
            return result

        old.act = new.act = act
        original_record = old.record

        def record(*args, **kwargs):
            row = original_record(*args, **kwargs)
            if kwargs.get("evaluation", False):
                observer.eval_rows.append(
                    canonical_hash(
                        {k: row[k] for k in ("raw_actions", "reward", "service", "post", "decision")}
                    )
                )
            else:
                observer.pending.update(
                    exogenous=row["exogenous_sha256"],
                    decision=canonical_hash(row["decision"]),
                    post=canonical_hash(row["post"]),
                    service=canonical_hash(row["service"]),
                    rng_after=observer.rng(),
                    replay_rng=digest(observer.replay.rng.getstate()),
                    target_sync=observer.step in observer.syncs,
                )
                observer.emit("steps", observer.pending)
                if observer.step in (1000, 1004, 1500, 3000):
                    write(observer.directory / f"state_{observer.step:06d}.json", observer.state())
                if observer.step % 150 == 0:
                    print(
                        json.dumps(
                            dict(trial=observer.label, step=observer.step, updates=observer.update)
                        ),
                        flush=True,
                    )
            return row

        old.record = new.record = record
        original_update = BS5CurriculumQMIX.train_batch
        clip = torch.nn.utils.clip_grad_norm_

        def clip_observed(parameters, *a, **kw):
            parameters = list(parameters)
            observer.grad_before = np.concatenate(
                [p.grad.detach().cpu().numpy().ravel() for p in parameters if p.grad is not None]
            )
            return clip(parameters, *a, **kw)

        torch.nn.utils.clip_grad_norm_ = clip_observed
        huber = torch.nn.functional.smooth_l1_loss

        def loss_observed(input, target, *a, **kw):
            result = huber(input, target, *a, **kw)
            if observer.phase == "update":
                observer.tensors["td_target"] = [target.detach()]
                observer.tensors["loss"] = [result.detach()]
            return result

        torch.nn.functional.smooth_l1_loss = loss_observed

        def update(learner, batch):
            observer.phase, observer.tensors = "update", {}
            metrics = original_update(learner, batch)
            observer.update += 1
            arrays = {
                f"forward_{key}_{i}": x.cpu().numpy().copy()
                for key, values in observer.tensors.items()
                for i, x in enumerate(values)
            }
            arrays.update(grad_before=observer.grad_before, grad_after=observer.grad_after)
            for key in ("agent", "mixer", "target_agent", "target_mixer"):
                arrays["weights_" + key] = np.concatenate(
                    [
                        t.detach().cpu().numpy().ravel()
                        for _, t in sorted(getattr(learner, key).state_dict().items())
                    ]
                )
            row = dict(
                step=observer.step,
                update=observer.update,
                target_sync=observer.step % observer.protocol["target_every"] == 0,
                metrics=metrics,
                **observer.batch,
            )
            observer.numeric(arrays, row, "update")
            observer.emit("updates", row)
            observer.phase = "train"
            return metrics

        BS5CurriculumQMIX.train_batch = update
        target = BS5CurriculumQMIX.update_targets

        def update_targets(learner, *a, **kw):
            result = target(learner, *a, **kw)
            if observer.phase == "train":
                observer.syncs.append(observer.step)
            return result

        BS5CurriculumQMIX.update_targets = update_targets
        original_evaluate = observer.module.evaluate

        def evaluate(*a, **kw):
            if observer.exploration is None:
                # The runners create this generator before eval0; it is still unconsumed.
                # Use the actual generator captured in the caller's locals, read-only.
                observer.exploration = sys._getframe(1).f_locals["exploration"]
            before = observer.rng()
            if observer.step == 0:
                meta = read(
                    next((observer.root / "trials" / observer.label / "runs").glob("*/meta.json"))
                )
                write(
                    observer.directory / "initial.json",
                    dict(
                        state=observer.state(),
                        resolved_config=meta["resolved_config"],
                        initial_weights_sha256=meta["initial_weights_sha256"],
                        runtime=observer.runtime,
                        schedule=observer.protocol["scenario_schedule"]["551"][:20],
                    ),
                )
            observer.phase, observer.eval_rows = "eval", []
            result = original_evaluate(*a, **kw)
            after = observer.rng()
            assert before == after, "evaluation mutated training RNG"
            observer.emit(
                "evals",
                dict(
                    step=observer.step, rows=observer.eval_rows, rng_before=before, rng_after=after
                ),
            )
            observer.phase = "train"
            return result

        observer.module.evaluate = evaluate

    def close(self):
        for stream in self.streams.values():
            stream.close()


def worker(root, label):
    from scripts.training import run_bs5_curriculum_pilot as old
    from scripts.training import run_bs5_direct_replay_reset as new

    config = read(root / "config.json")
    runtime = configure(config["device"])
    trial = read(root / "budget_ledger.json")["trials"][label]
    module = old if trial["implementation"] == "A" else new
    trial_root = root / "trials" / label
    reference = root / "trials/A/diagnostic" if label != "A" else None
    observer = Observer(root, label, module, runtime, reference)
    observer.install()
    p, registry = read(trial_root / "protocol.json"), read(trial_root / "run_registry.json")
    stop = trial["reserved_steps"]
    status, reason = "failed", None
    try:
        try:
            if trial["implementation"] == "A":
                old.run_one(trial_root, p, registry, "D-R", 551, export=False, stop_after=stop)
            else:
                timers = new.Timers(trial_root, "prefix_only", device=config["device"])
                new.run_one(trial_root, p, registry, 551, timers, stop_at=(stop, "before_clear"))
        except KeyboardInterrupt:
            # Only the known durable stop at the reserved endpoint counts as success.
            commit_path = next((trial_root / "runs").glob("*/commit.json"))
            commit = read(commit_path)
            assert observer.step == stop and commit["steps"] == stop
            assert commit["updates"] == len(range(p["warmup"], stop + 1, p["update_every"]))
            assert observer.clears == 0 and len(observer.replay) == stop and commit["lineage"] == 0
            assert observer.syncs == list(range(p["target_every"], stop + 1, p["target_every"]))
            if trial["implementation"] == "B":
                assert commit["applied_reset_steps"] == []
                if stop == 3000:
                    assert commit["reset_phase"] == "before_clear" and commit["resets"] == [
                        dict(kind="reset_requested", step=3000)
                    ]
            assert observer.update == len(range(p["warmup"], stop + 1, p["update_every"]))
            status = "complete_stop_before_clear"
        else:
            raise AssertionError("real runner returned without the bounded stop hook")
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        observer.close()
        if observer.learner is not None and observer.replay is not None:
            write(
                observer.directory / "endpoint.json",
                dict(
                    state=observer.state(),
                    steps=observer.step,
                    updates=observer.update,
                    replay_size=len(observer.replay),
                    lineage=0,
                    reset_applied=observer.clears,
                    target_sync_steps=observer.syncs,
                    runtime_s=time.monotonic() - observer.started,
                    status=status,
                    reason=reason,
                ),
            )
        Budget(root).finish(label, status)


def prepare(root, config_path, device):
    from scripts.training.run_bs5_launch_ab import source_identity, freeze_source
    from scripts.training.run_bs5_direct_replay_reset import resolved_protocol

    config = read(config_path)
    assert (
        config["seed"] == 551
        and config["prefix_steps"] == 3000
        and config["hard_cap_train_steps"] == 15000
    )
    assert config["numerical_tolerance"] == dict(atol=1e-7, rtol=1e-6)
    identity = source_identity()
    # Validate science/protocol before either prefix can consume budget.
    resolved_protocol(
        read(ROOT / config["reset_config"]),
        identity,
        read(ROOT / config["baseline"]),
        device=device,
    )
    root.mkdir(parents=True, exist_ok=False)
    config.update(
        device=device,
        config_input_sha256=file_hash(config_path),
        scope="prefix_diagnostic_not_pilot",
    )
    write(root / "config.json", config)
    write(
        root / "budget_ledger.json",
        dict(
            hard_cap_train_steps=config["hard_cap_train_steps"],
            charging="reserve full trial before launch; charge before each environment step; interrupted reservations retained",
            trials={},
        ),
    )
    freeze_source(root, identity)


def make_trial(root, label, implementation, stop):
    from scripts.training.run_bs5_launch_ab import source_identity
    from scripts.training.run_bs5_direct_replay_reset import resolved_protocol

    config = read(root / "config.json")
    identity = read(root / "source_identity.json")
    assert source_identity() == identity, "source drift: keep this diagnostic source fixed"
    baseline = read(ROOT / config["baseline"])
    p = (
        deepcopy(baseline["protocol"])
        if implementation == "A"
        else resolved_protocol(
            read(ROOT / config["reset_config"]), identity, baseline, device=config["device"]
        )
    )
    p.update(
        scope="prefix_diagnostic_not_pilot",
        source_sha256=identity["sha256"],
        device=config["device"],
        training_seeds=[551],
        evaluation_steps=[0, stop],
        common_evaluation_steps=[0, stop],
        train_trace_every_episodes=1,
    )
    trial_root = root / "trials" / label
    trial_root.mkdir(parents=True)
    if implementation == "B":
        # Fresh current-runtime initialization is paired to A, not a 24k checkpoint.
        # Historical input file stays immutable; record the precise diagnostic override.
        initial = read(root / "trials/A/diagnostic/initial.json")
        meta = baseline["metas"]["D-R_seed551"]
        write(
            trial_root / "initial_pairing_override.json",
            dict(
                historical=deepcopy(meta),
                reason="pair fresh initialization in the diagnostic runtime; no claim of historical bitwise equivalence",
                diagnostic_initial_weights_sha256=initial["initial_weights_sha256"],
                historical_prefix_input_refs_sha256=canonical_hash(
                    baseline["training_references"]["551"][:20]
                ),
            ),
        )
        meta["initial_weights_sha256"] = initial["initial_weights_sha256"]
        meta["resolved_config"] = initial["resolved_config"]
        a_run = root / "trials/A/runs/D-R_seed551"
        for eid in range(stop // p["horizon"]):
            episode = read(a_run / "episodes" / f"{eid:06d}.json")
            with gzip.open(
                a_run / "train_traces" / f"{eid:06d}.json.gz", "rt", encoding="utf-8"
            ) as stream:
                trace = json.load(stream)["trace"]
            reference = baseline["training_references"]["551"][eid]
            reference["residual"] = episode["residual"]
            reference["exogenous"] = [r["exogenous_sha256"] for r in trace]
        p["baseline_sha256"] = canonical_hash(baseline)
    write(trial_root / "protocol.json", p)
    write(trial_root / "baseline_refs.json", baseline)
    arm = "D-R" if implementation == "A" else "D-R-reset"
    write(
        trial_root / "run_registry.json",
        dict(
            status="prepared",
            attempts=[],
            runs={f"{arm}_seed551": dict(status="not_started", steps=0)},
        ),
    )


def launch(root, label, implementation, stop, kind):
    Budget(root).reserve(label, stop, implementation, kind)
    make_trial(root, label, implementation, stop)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="551",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with (root / f"{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--output",
                str(root),
                "--worker",
                label,
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(
            f"trial {label} failed with exit {code}; see {label}.log; reservation retained"
        )


def export_review(root):
    """Export once at the end, including compressed raw 20-episode traces."""
    payload = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(root).as_posix()
        if (
            path.suffix == ".pt"
            or "numeric" in path.parts
            or path.name.endswith(".tmp")
            or path.name == "active.lock"
        ):
            continue
        if name.startswith("exports/") or name == "source_snapshot.zip":
            continue
        payload[name] = path.read_bytes()
    payload["analyze_bs5_prefix_verify.py"] = (
        ROOT / "scripts/analysis/analyze_bs5_prefix_verify.py"
    ).read_bytes()
    payload["README.md"] = (ROOT / "scripts/notebooks/BS5_PREFIX_VERIFY_KAGGLE.md").read_bytes()
    payload["REVIEW_SHA256SUMS.json"] = (
        json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in sorted(payload.items())}, indent=2)
        + "\n"
    ).encode()
    target = root / "exports" / f"bs5_prefix_verify_review_{time.time_ns()}.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for n, value in sorted(payload.items()):
            z.writestr(n, value)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(payload)
    print(f"REVIEW_ZIP={target}", flush=True)
    return target


def finish(root, status, reason=None):
    ledger = read(root / "budget_ledger.json")
    verification = dict(
        status=status,
        reason=reason,
        seed=551,
        train_steps_charged=sum(t["consumed_steps"] for t in ledger["trials"].values()),
        reserved_train_steps=sum(t["reserved_steps"] for t in ledger["trials"].values()),
        hard_cap_train_steps=ledger["hard_cap_train_steps"],
        no_24k_training=True,
        scope="same-runtime diagnostic only; not causal reset attribution",
        tolerance=read(root / "config.json")["numerical_tolerance"],
    )
    write(root / "verification.json", verification)
    text = [
        "# Kiểm tra prefix BS5",
        "",
        f"Trạng thái: `{status}`.",
        "",
        f"Seed 551; số bước train đã tính budget: {verification['train_steps_charged']}; trần 15000.",
        "",
        "Gọi hai run_one gốc: A dùng stop_after, B dùng stop_at before_clear. Không sao chép vòng train hoặc thay learner.",
        "",
        "Tolerance chốt trước rollout: atol=1e-7, rtol=1e-6. Protocol, actions, masks, scenario, replay sample và RNG so chính xác. MATCH còn yêu cầu hashes số học khớp chính xác.",
        "",
        "Xem prefix_comparison.json, initial/endpoint, log từng bước và update. Eval 0/3000 dùng evaluator gốc và phải giữ nguyên RNG train.",
        "",
        "Không dùng checkpoint lịch sử để khởi tạo. Pass CPU không giải thích sai khác GPU lịch sử; pass GPU với cấu hình mới cũng không chứng minh +10,2% lịch sử là do reset.",
        "",
        "Review bỏ tensor số học đầy đủ và checkpoint optimizer. State đầy đủ giữ trong thư mục output; analyzer offline không cần inference/checkpoint.",
    ]
    if reason:
        text += ["", "Reason: " + reason]
    (root / "report_vi.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    export_review(root)
    print(json.dumps(verification), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bs5_prefix_verify_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed prefixes only; never silently replay failed trials",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker:
        worker(root, args.worker)
        return
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if not root.exists():
        prepare(root, args.config, device)
    else:
        assert args.resume, "output exists: use --resume; never overwrite a diagnostic or ledger"
        assert read(root / "config.json")["device"] == device
    if args.prepare_only:
        print(f"PREPARED_NO_TRAINING={root}")
        return
    lock = root / "active.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        for label in ("A", "B"):
            trials = read(root / "budget_ledger.json")["trials"]
            if label in trials:
                assert trials[label]["status"] == "complete_stop_before_clear", (
                    "incomplete reservation retained; inspect before any new trial"
                )
                continue
            launch(root, label, label, 3000, "main_pair")
        result = compare(root / "trials/A/diagnostic", root / "trials/B/diagnostic")
        write(root / "prefix_comparison.json", result)
        status = result["status"]
        if result["first_divergence"]:
            write(root / "first_divergence.json", result["first_divergence"])
        if status != "MATCH_SAME_RUNTIME":
            # At most one repeat. Existing repeat is reused, never re-reserved.
            first = result["first_divergence"]["step"]
            stop = min(3000, max(150, ((first + 150 + 149) // 150) * 150))
            trials = read(root / "budget_ledger.json")["trials"]
            if "A_repeat" not in trials:
                launch(root, "A_repeat", "A", stop, "one_self_repeat")
            else:
                assert trials["A_repeat"]["status"] == "complete_stop_before_clear"
            repeat = compare(
                root / "trials/A/diagnostic", root / "trials/A_repeat/diagnostic", limit=stop
            )
            write(root / "self_repeat_comparison.json", repeat)
            if repeat["status"] != "MATCH_SAME_RUNTIME":
                status = "SELF_NONREPRODUCIBLE"
            # A stable self-repeat alone is insufficient to assert a concrete code bug.
        finish(root, status)
    except BaseException as exc:
        finish(root, "BLOCKED_OR_UNRESOLVED", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
