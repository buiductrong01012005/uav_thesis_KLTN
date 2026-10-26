# Return-v4: training and benchmark are separate

Use `BS5_RETURN_V4_KAGGLE_DATASET.zip` as the single source Kaggle Dataset.
It contains source, configurations, notebooks and diagnostic tests. It contains
no previous checkpoints, large result archives or pre-trained initialization.
No historical asset Dataset is needed for the two new entrypoints below.

## 1. Train first

Import `bs5_return_v4_fresh_train_kaggle.ipynb`, Add Input the source Dataset,
choose a GPU if available, Run All. The notebook uses installed Kaggle
Torch/NumPy/Matplotlib without replacing Torch, and records the exact runtime.
Internet is unnecessary when those standard packages are present.

- Fresh C-R/C-N x train seeds551/552; 60,000 steps each.
- S1:0–3k, S2:3–6k, S3:6–12k, S4:12–60k.
- S1–S3 game horizon150; S4 horizon1200; S4 bootstraps at the time limit.
- Validation at0,6k,...,60k: five fixed scenario seeds130001–130005,
  1200 uninterrupted steps each, epsilon0. Best is mean utility; ties earliest.
- Total main compute:240k training +264k validation. No final test in training.
- LR3e-4, gamma.99, batch64, warmup1000, replay30k, update every4,
  target every1000, epsilon1→.05 over18k, same learner as source.
- Same train seed => identical initial weights, exogenous scenario schedule,
  residual battery/assignment draws and UE tapes between arms. Replay,
  optimizer and exploration then evolve independently.
- S1/S2 fixture is inherited and favors Relay coordination. This is the same
  curriculum ablation, not a separately tuned no-Relay training recipe.

The notebook runs bounded diagnostic seeds first (not551/552 or final test),
then main training. Diagnostics verify exact CPU resume including optimizer,
target models, replay, sample sequence and RNG. GPU exactness is protected by
runtime/hardware locking; CUDA execution cannot be validated on a CPU host.

Learning curves retain raw per-slot NPZ, bins up to100 slots, episode summaries,
and every batch update. Plots show raw bins and trailing5-bin smoothing,
reset at curriculum substage changes. Validation markers remain actual data.

## 2. Resume across sessions

The soft deadline is10.5h from notebook setup, with30min reserved for export.
The runner estimates unit time from completed games and checkpoints and stops
before starting another game when time is insufficient. There is no guarantee
that all four runs finish within a single12h allocation.

`paused_at_game_boundary` is safe to resume. Save notebook output. Next session,
Add Input `training_resume.zip` (or the previous notebook output) plus the same
source Dataset and Run All. Auto-discovery works with one candidate; otherwise
set `RESUME_INPUT_DIR` to the exact archive or run directory.

Resume files include online/target agent+mixer, Adam, full replay and its RNG,
exploration/Python/NumPy/Torch/CUDA RNG, schedule progress and budget ledger.
Resume generations have unique step-based filenames and immutable progress
commits; the latest two full replay snapshots are retained to bound disk use.
All inference checkpoints, raw games and commit receipts remain retained.
Validation commits each game; training commits each completed game. No game
resets mid-horizon to meet a chunk size. State cannot be silently resumed with
another Torch/NumPy/Python build, device or GPU model.

An abrupt process kill inside a game can leave charged but uncommitted compute;
the runner blocks silent repeats and preserves evidence. This differs from a
planned soft stop. Keep the error/output/ledger for inspection. Export failures
still produce an explicitly uncertified raw evidence archive.

## 3. Outputs

- `review_small.zip`: report, learning curve PNG/PDF, bins/episodes, full update
  metrics, validation/cycles tables. Send this first for review.
- `training_resume.zip`: full resumable/evidence run, including all checkpoints.
- `training_for_benchmark.zip`: small selected best/last checkpoint handoff,
  protocol, frozen test plan, selection receipts/progress. Produced only when
  all four runs and all validation games have completed. Cannot resume training.
- Full raw datasets belong outside the git source repository.

## 4. Benchmark later

Import `bs5_return_v4_benchmark_kaggle.ipynb`. Add Input the same source Dataset
and `training_for_benchmark.zip`. CPU recommended: heuristic previews are CPU
Python/NumPy and cannot be assumed to benefit from a GPU.

Final test scenarios260001–260010 are separate from training/validation and
the previously inspected130xxx/190xxx traces. Two trained seeds per learned arm
plus Random-Relay and four Recovery greedies => nine configurations/seven
methods. Each configuration runs the same ten10k traces:90 games/900k steps.
Random is ONE configuration, with declared per-scenario algorithm RNG seeds.

Optional20-scenario plan uses the additional locked260011–260020 for EVERY
configuration. Select10/20 before prepare for compute reasons, not rankings.
No repeated nonlearning runs just to mirror RL training seeds.

Greedy wrappers use privileged current exact UE/network state, search roles
then one coordinate movement pass, and apply Recovery after3 executed zero
utility slots. Candidate actions are scored by an isolated one-slot env step;
the immutable tape is shared read-only while all mutable env state is copied.
This safely includes Return-v4 route cache and energy-shortfall overrides.
These are not full-horizon oracles or information/compute-matched comparators.
Candidate preview calls and decision latency are reported separately from
executed physical-slot budgets.

Tables retain per-seed/per-scenario results, service/QoS/instant and accumulated
Jain, outages, charging cycles and censoring, Relay traffic/empty fractions and
decision cost. Main comparison uses only complete common scenario blocks.
Bootstrap intervals resample scenarios after averaging learned seeds within
each scenario; two training seeds are still limited replication. Never treat
slots or cycles as independent experimental scenarios.

`benchmark_resume.zip` resumes benchmark only. No retraining is needed when
splitting benchmark over sessions. Checkpoints cannot be reselected on test.

## CLI and offline audit

```
python -B scripts/verification/preflight_bs5_v4_fresh.py --output PREFLIGHT
python -B scripts/training/run_bs5_v4_fresh_train.py prepare --output OUT --device cuda
python -B scripts/training/run_bs5_v4_fresh_train.py run --output OUT --max-seconds 36000
python -B scripts/training/run_bs5_v4_fresh_train.py export --output OUT
python -B scripts/evaluation/run_bs5_v4_benchmark.py prepare --output BENCH --training HANDOFF
python -B scripts/evaluation/run_bs5_v4_benchmark.py run --output BENCH --max-seconds 36000
python -B scripts/evaluation/run_bs5_v4_benchmark.py export --output BENCH --deep-planner
```

`--deep-planner` reconstructs the Return route controller offline on export;
it adds no policy training or evaluation rollout. Default export checks hashes,
masks, continuity, energy, collision occupancy, terminal semantics and rates.
Deep independent route audit costs more CPU time and is optional for full data.
