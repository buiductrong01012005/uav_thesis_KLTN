# UAV Thesis – Spatial-QMIX source package

This package contains the current research code for the two-stage UAV recovery project.
The recommended long-run experiment is the frozen **10x10 Stage-B random-dynamic Spatial-QMIX** setup.

## 1. Requirements

Recommended Python: **3.11 or 3.12**.

Dependencies:
- NumPy
- PyTorch
- pytest

The current protocol is CPU-safe and explicitly uses one PyTorch CPU thread for stable/reproducible runs.

## 2. Windows PowerShell setup

From the folder that contains the ZIP:

```powershell
Expand-Archive .\uav_thesis_src.zip -DestinationPath .\uav_thesis_run
cd .\uav_thesis_run\uav_thesis

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use a Python 3.10+ interpreter you already have, for example:

```powershell
python -m venv .venv
```

## 3. Linux/macOS setup

```bash
unzip uav_thesis_src.zip -d uav_thesis_run
cd uav_thesis_run/uav_thesis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verify the project first

Run the test suite:

```bash
python -m pytest -q
```

The packaged version should pass all tests before you start a long run.

## 5. Short smoke test

Do this before any multi-hour experiment:

```bash
python scripts/run_g10_longrun.py --seed 401 --max-transitions 1000 --eval-every 500 --heldout 10 --patience 99
```

This is only a pipeline check, not a meaningful final training result.

## 6. Recommended 10x10 long run

The current frozen protocol:

```bash
python scripts/run_g10_longrun.py --seed 401 --max-transitions 50000 --eval-every 5000 --heldout 100 --eval-seed 990010 --epsilon-decay-transitions 20000 --min-delta 0.02 --patience 3
```

Meaning:
- maximum budget: 50,000 environment transitions;
- deterministic held-out evaluation approximately every 5,000 transitions;
- 100 held-out episodes at each evaluation;
- automatic early stop when three consecutive checkpoint-to-checkpoint gains are each below 0.02 absolute mean-service improvement.

The run may overshoot a requested transition milestone by at most one episode because checkpoints are saved only at episode boundaries. This makes resume state consistent.

## 7. Resume an interrupted run

Use the same seed and add `--resume`:

```bash
python scripts/run_g10_longrun.py --seed 401 --max-transitions 50000 --eval-every 5000 --heldout 100 --eval-seed 990010 --epsilon-decay-transitions 20000 --min-delta 0.02 --patience 3 --resume
```

The resume file is:

```text
results/longrun_g10/longrun_s401_state.pkl
```

Model checkpoints are written to:

```text
checkpoints/longrun_g10/
```

Learning-curve metadata are written to:

```text
results/longrun_g10/longrun_s401_curve.json
```

## 8. View the learning curve

```bash
python scripts/show_longrun_curve.py --seed 401
```

It prints a compact table containing:
- transitions;
- held-out mean service;
- held-out final service;
- policy decision latency;
- epsilon.

## 9. What to watch during a 3–4 hour run

Do **not** judge the run using individual training-episode rewards; the random dynamic scenarios make those highly noisy.

Use the deterministic held-out checkpoints instead.

A run is worth continuing when the held-out mean-service curve is still making meaningful gains. A rough interpretation for the current 10x10 setup is:

- ~40%: already reproduces the current medium-run region;
- 50–60%: promising long-run improvement;
- 60%+: strong evidence that more training is closing the gap to the fast heuristic;
- three successive gains below ~2 percentage points: reasonable signal to stop and avoid wasting compute.

The fast heuristic is currently stronger in solution quality, so the purpose of this run is to measure how much of that gap Spatial-QMIX can close with more samples, not to force a particular target result.

## 10. Important methodology note

For the current thesis audit, keep these fixed during a long-run comparison:
- reward;
- Spatial-QMIX architecture;
- action space;
- random-dynamic environment distribution;
- evaluation seed/protocol.

Do not tune a single seed mid-run. If a new design is tested later, run it as a separate experiment.

## 11. Main folders

```text
src/envs/       environments
src/rl/         QMIX / Spatial-QMIX / audit algorithms
src/baselines/  random, greedy, oracle and coverage baselines
scripts/        training/evaluation runners
tests/          unit tests
configs/        retained configs
notes/          research/audit notes
```

`results/` and `checkpoints/` are intentionally not bundled with the source ZIP; the runners create them automatically.
