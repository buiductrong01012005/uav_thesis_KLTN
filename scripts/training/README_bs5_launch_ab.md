# Fixed BS5 launch A/B pilot

For Kaggle, use the [four-cell notebook](../notebooks/bs5_launch_ab_kaggle.ipynb)
or [copyable cells](../notebooks/README.md). Kaggle extracts Dataset ZIPs;
the notebook discovers the read-only source, writes a device override outside
the source before preparation, and keeps all results in `/kaggle/working`.
It performs a fresh preparation on the selected device, then trains/resumes
and exports download bundles. Previous local preparation is not reused.

This opt-in runner uses BS ID **5**, the audited failure-context layout
state/obs **111/116**, full battery/mobility and ten Cartesian actions in all
four arms. Historical environment/router/runner files are unchanged.

The main experiment is A-R/A-N/B-R/B-N × training seeds **451/452**.
Seed 351 appeared in an earlier experiment; 451/452 had no references in the
local source/docs at preparation. Train scenarios are 120001–120040 and
validation scenarios 130001–130010. These are validation, never final tests.

From the repository root, preparation is a separate, bounded operation:

```powershell
python -B -m scripts.training.run_bs5_launch_ab --prepare --outdir results/bs5_launch_ab_20261022
```

It executes physics probes, eight **300-step debug runs** (2,400 training
transitions, outside the main budget), validates every debug evaluation trace,
tests checkpoint-free offline analysis with torch/src imports blocked, and
freezes `protocol.json`. Runtime includes diagnostic logging, shadow/oracle
scoring and an export allowance, with a 1.5 safety factor. The common main
budget is 24k/run, reduced uniformly to 12k/run when the estimate exceeds four
hours. This estimate is approximate, not a promise. No main training occurs
under `--prepare`.

After successful preparation, run (also the restart command):

```powershell
python -B -m scripts.training.run_bs5_launch_ab --run --resume --outdir results/bs5_launch_ab_20261022
```

The runner automatically iterates all eight runs, evaluates at step 0 and every
6,000 transitions, chooses best validation utility including step 0 (earlier
exact tie), exports bundles, and skips completed runs on restart. The LR and
epsilon schedule do not reset between evaluation blocks. No final thesis test,
curriculum, random failure, prior checkpoint or reward shaping is activated.

`--resume` restores online/target networks, Adam, replay and all policy/global
RNG states at the most recent committed evaluation boundary. Episodes start
from the frozen scenario/residual-battery schedule. Work after that boundary
is archived in `aborted_attempts`, then rerun. An interrupted in-flight episode
is not an exact continuation from its last slot. Do not edit the source after
preparation: its fingerprint is verified before training. An OS lock prevents
two processes from writing the same experiment.

Outputs under the chosen directory:

- `protocol.json`, `schema.json`, `source_identity.json`, `run_registry.json`.
- `preflight/`: source-derived topology SVG, geometry, matched-action probes,
  physical relay control and offline acceptance evidence.
- `runs/<arm>_seed<seed>/`: every train-episode metric, every learner-update
  diagnostic, scheduled full train traces, every full validation trace,
  all evaluation weights, best/last selection in `commit.json`, and `resume.pt`.
- `analysis/`: per-seed run table, five contrasts, paired scenario differences,
  service/relay/lifecycle metrics, learning curves and report.
- `exports/audit_light.zip`: all committed evidence and standalone analysis;
  no weights, replay, optimizer or torch/env dependency.
- `exports/repro_full.zip`: the above plus exact source ZIP, checkpoint weights,
  current full resume states and archived aborted artifacts.

The exports are replaced atomically at evaluations and completion. Original
results outside this experiment directory are not modified. A new preparation
refuses an existing directory. Final analysis can be rebuilt without rollout:

```powershell
python -B -m scripts.analysis.bs5_bundle --bundle results/bs5_launch_ab_20261022/exports/audit_light.zip --output results/bs5_launch_ab_20261022/offline_rebuilt --require-complete
```

Both bundles also contain `analyze_bundle.py` and `validate_bundle.py`, runnable
with `--root <extracted-directory> --output <different-output-directory>`.
Use an output outside the extracted bundle so its SHA256 manifest stays intact.

Router v2 preserves association, access sharing, radio formulas and utility.
After the original candidate route choice, it checks shared relay allocation
against **same-Access direct fallback**. A losing client falls back to direct;
shares are recomputed until no remaining client loses direct service. It is a
deterministic admission fix, not a globally optimal router. Different Access
sets and relay opportunity cost can still make all-Access better.
