# Episode-level random single-BS failure, v1

This module adds a new opt-in environment and learner. Existing environment
classes, runners, checkpoint layouts, radio, reward, launch cells and battery
rules are unchanged. This is not a curriculum implementation.

Exactly one of the six fixed BS coordinates fails per episode. `failure_mode`
is `fixed` or `random`; `failure_candidates` controls the random subset. IDs
are zero-based: BS5 is `(925 m, 925 m)`. The env deep-copies caller config,
selects the failure from a separate RNG, rebuilds all topology/region caches
before initializing UEs, and appends the six-way failure one-hot to the legacy
state. State/observation dimensions are 111/116 for this six-BS map.

```python
import json
from src.envs.episode_random_failure_env import (
    EpisodeFailureCartesianConfig,
    EpisodeFailureCartesianEnv,
)
from src.rl.episode_failure_spatial_qmix import EpisodeFailureSpatialQMIX

with open("configs/episode_random_failure_v1.json", encoding="utf-8") as f:
    cfg = EpisodeFailureCartesianConfig(**json.load(f))
env = EpisodeFailureCartesianEnv(cfg)
out = env.reset(seed=123)
learner = EpisodeFailureSpatialQMIX(env)

# Explicit stratified evaluation. Override is allowed outside the training
# candidate subset, is labeled in metadata, and doesn't consume failure RNG.
out = env.reset(seed=9401, options={"failed_bs_idx": 5})
```

`EpisodeFailureV9Env` and `EpisodeFailureMovementEnv` use
`EpisodeFailureV9Config`, preserving their original 10-action RelayTo and
5-action movement semantics respectively. The Cartesian variant remains pure
movement × role, with the original opportunistic router. The earlier router
admission problem is NOT repaired by this failure-randomization change.

Do not modify `env.cfg.failed_bs_idx` to switch episodes. Use reset options or
construct a new config. This field reports the active episode's failure;
`configured_failed_bs_idx` records the originally configured fixed failure.
Calling `reset(seed=s)` repeatedly gives the SAME failure/scenario by design.
Vary the scenario seed or use `reset()` to advance the independent failure
stream. Each failure stays fixed until the next reset. The exposed ID resolves
the omitted-failure ambiguity, not the pre-existing partial UE observation.

The legacy Spatial-QMIX parser cannot read the new context layout. Use the
new learner. Old checkpoints must not be interpreted as failure-aware models.
`expose_failure_context=False` exists only for explicit diagnostic/legacy
ablations. There is no automatic weight migration. The old full-task launcher
still constructs the old fixed env; these new JSON files are not magically
read by it. Use the new pilot entry point or integrate this explicit factory
and learner into a separately versioned full training protocol.

## Reproduce checks (from repo root)

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
PYTHONDONTWRITEBYTECODE=1 python -m scripts.analysis.audit_episode_random_failure --output results/random_bs_v1/audit
PYTHONDONTWRITEBYTECODE=1 python -m scripts.analysis.pilot_episode_random_failure --output results/random_bs_v1/pilot
python -m scripts.analysis.analyze_episode_random_failure_bundle --results results/random_bs_v1 --output results/random_bs_v1/offline_analysis.json
```

The pilot defaults to 2 seeds × 2 treatments × 6,000 transitions = 24,000 new
training transitions. It uses all original full dynamics, no curriculum, no
pretraining, original launch cells, no shaping, and last-checkpoint reporting.
Both treatments have the same new model architecture. Validation is stratified
over all six failures with a held-out UE seed, and traced at steps 0/3000/6000.
This small diagnostic validation set is NOT a final research test set.

Traces capture service after movement and before battery/mobility updates.
All validation slots are recorded; training has a summary every episode and
update, plus full traces of the first encountered episode per failed BS.
The standalone analyzer needs only Python/NumPy. It never imports the env or
loads a model, validates rates against rewards, reconstructs summaries, checks
paired UE trajectory hashes and requires every registered pilot run.
Pilot checkpoints contain agent/mixer weights, not exact optimizer/replay
resume state. Re-running the registered pilot is supported; exact resume is
not claimed.
