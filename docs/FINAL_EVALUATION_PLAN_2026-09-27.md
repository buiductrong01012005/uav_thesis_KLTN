# Final evaluation plan — locked

## 1. Main learned result: A0

Train Spatial-QMIX on the exact locked A0 task with at least three independent training seeds using identical hyperparameters and 100k transitions per seed.

Recommended training seeds: 81, 82, 83.

For each seed report:
- best validation mean utility,
- final20 utility,
- best-checkpoint step,
- runtime,
- held-out evaluation on unseen static UE layouts.

Aggregate mean ± std across training seeds. Preserve best validation checkpoint; never report latest-only if training collapses after a better checkpoint.

Main comparisons:
- learned Spatial-QMIX,
- frozen initial deployment,
- reachable coordinate-ascent reference on the canonical layout,
- Relay-movement freeze ablation on the canonical layout.

## 2. A1 battery/dynamic-role experiment

Report as a diagnostic/negative result, not as a positive main claim.

Use the clean identity-randomized scratch variant and matched low-battery/full-battery evaluation. The key question is whether early role handover differs between otherwise matched battery conditions.

Do not spend more training budget trying to force a positive result.

## 3. A2 mobility necessity

Use the already completed seeds 11-13 staged audit:
- adaptive reposition vs frozen deployment,
- report mean ± std / per-seed gains.

## 4. A3 combined robustness

Use the already completed seeds 11-13 staged audit and report the negative role result honestly.

## 5. Held-out policy check

A0 seed81 pilot held-out evaluation is stored under results/final_a0_heldout3.json. This is a pilot because it uses one trained checkpoint over multiple environment seeds, not multiple training seeds.

## 6. Final acceptance rule

The thesis may proceed to writing once:
- A0 has >=3 completed independent training seeds,
- held-out A0 evaluation is run for each best checkpoint,
- A2/A3 staged results are preserved,
- A1 is clearly labeled negative/diagnostic,
- no additional environment tuning is introduced.
