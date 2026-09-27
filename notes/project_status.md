# Project status

## V0 environment
- 5x5 grid, 3 homogeneous UAVs
- Known fixed UE map
- Roles: Access / Relay
- Action: movement + role
- Reward: served-user ratio
- Direct or one-relay-hop backhaul

## Sanity tests
- 4/4 environment tests passing
- Manual configuration: AAA=2/8, RRR=0/8, AAR=8/8, ARR=6/8

## Baselines
- Random mean served-user ratio: 0.063775
- Scripted Greedy mean served-user ratio: 0.995

## QMIX preliminary
- Feed-forward shared agent Q-network + monotonic mixer
- Fixed-map overfit experiment demonstrates learning above Random
- Corrected epsilon schedule reached ~0.71 evaluation served ratio transiently and ~0.61-0.62 later in the interrupted run
- Not yet considered V0 pass because it remains below the scripted Greedy reference and shows instability

## Next
- Stabilize V0 QMIX and save best checkpoints during training
- Inspect deterministic trajectories and failure modes
- Only after V0 passes, add Scout for V1

## V0 oracle baseline
- Added `StaticOraclePolicy` with privileged full UE-map knowledge.
- Exhaustively searches all `(position, role)` assignments for all 3 UAVs:
  `(25 cells × 2 roles)^3 = 125,000` static configurations.
- It gives an **exact steady-state served-user ceiling** for V0.
- Important wording: it is not claimed to be an exact finite-horizon trajectory oracle because it does not optimize cumulative transient reward along the path to the target configuration.
- Greedy remains a separate scripted expert baseline.

## V0 stabilization diagnostics
- Best service-only/potential checkpoint can reach full service but may oscillate between 8/8 and 6/8.
- A movement/role-switch penalty probe underperformed (~0.33 mean served at 250 episodes), so this direction was rejected.
- Added optional Polyak/soft target updates to QMIX (`update_targets(tau)`). Existing hard-update behavior remains the default (`tau=1`).
- Soft-target probe (seed 61, lr=2e-4, tau=0.01, 300 episodes) converged stably to 7/8 served = 0.875 and then selected Stay/Access, Stay/Access, Stay/Relay repeatedly.
- This removes the oscillation failure mode but exposes a local optimum: stable coordination is learned, yet one UE remains uncovered.
- Current interpretation: coordination and stability are both learnable; the remaining issue is escaping a spatial local optimum, not learning the Access/Relay semantics.

## V0 coordination-barrier diagnosis
- At the stable 7/8 state `[(2,1) A, (3,2) A, (1,3) R]`, an immediate 8/8 joint action exists in one step.
- Learned QMIX greedy action is `Stay/A, Stay/A, Stay/R` with immediate service 0.875.
- Exact one-step move `South/A, Stay/A, West/R` reaches 1.0 service but is ranked ~97/1000 by the learned joint Q estimate; the best immediate-optimal action is ranked ~86/1000.
- Local reward structure exposes a coordination barrier: moving the first Access south alone reduces service from 0.875 to 0.75, while moving the Relay jointly west/south with it raises service to 1.0.
- Joint-epsilon exploration probe did not improve over the stable QMIX checkpoint.
- Centralized Joint-DQN probe performed poorly (~0.185 mean served), showing that naive 1000-action centralized exploration is also difficult.
- A simplified parameter-shared MAPPO probe also did not outperform QMIX in the initial run.
- Decision: stop algorithm hopping in V0. V0 is accepted as a sanity stage because stable QMIX learns meaningful Access/Relay coordination at 7/8 service, well above Random (~0.064), while the oracle ceiling remains 1.0.

## V1 scaffold: Scout + unknown UE
- Added `src/envs/v1_env.py`.
- Same 5x5 / 3-UAV fixed map as V0, but UE positions are hidden from the exposed state at reset.
- Roles: Scout / Access / Relay; movement + role gives 15 actions per UAV.
- Scout reveals the exact UE count in the cell occupied after movement (perfect cell-level sensing for V1).
- Fleet shares a global discovered-demand map and known-cell mask.
- Access can serve only UEs in cells that have already been discovered.
- V1 training state intentionally excludes hidden UE ground truth; it contains only fleet-known information.
- Reward = served-user ratio + small discovery shaping for newly explored cells and newly discovered UEs.
- Added V1 unit tests; project test suite currently passes 10/10 tests.

## V1 learnability study (Scout + Access + Relay online)

### Baselines after adding a small Scout footprint (radius = 1 grid cell)
- Random baseline (200 episodes): mean served ratio ~0.0296; final discovered-UE ratio ~0.965.
- Online scripted baseline, using only revealed information: mean served ratio 0.8375; final served ratio 1.0; final discovered-UE ratio 1.0.
- Scripted trajectory discovers 25% of UEs in slot 1, begins service in slot 2, and reaches 100% service around slot 14.
- This confirms the environment is solvable and the Scout abstraction is not itself preventing service.

### Learning probes attempted
1. Shared-policy QMIX, weak discovery shaping (0.1 cell / 0.2 UE):
   - best checkpoint ~0.1175 mean served, 0.125 final served, only 0.25 final UE discovery.
2. Shared-policy QMIX, stronger discovery shaping (1.0 cell / 2.0 UE):
   - can discover ~0.875 of UEs, but best mean service only ~0.0575;
   - deterministic policy tends to collapse to almost all Access and no Relay.
3. Curriculum (service pretraining -> full V1):
   - service pretraining improves partially but does not recover V0-level Access/Relay coordination in the larger V1 network;
   - full V1 remains low-service.
4. Parameter-shared MAPPO:
   - ~0.1825 mean served, 0.25 final served, ~0.75 discovered UE;
   - again collapses toward almost all Access and no Relay.
5. Non-shared/separate-agent QMIX:
   - symmetry breaking does not solve the issue; peaks around ~0.12 mean service in probe evaluations.
6. Structured training reward with separate Access-coverage and backhaul shaping:
   - learns subgoals (coverage/backhaul) separately but still fails to combine them into persistent end-to-end service;
   - best probe ~0.01 mean service despite ~0.875 UE discovery.

### Interpretation
- The main bottleneck is no longer sensing sparsity after introducing the radius-1 Scout footprint.
- The hard part is the conjunction of three coordinated requirements:
  `discover demand + position Access + position/maintain Relay`.
- The same Access/Relay coordination barrier already appeared in V0, and the Scout role compounds it.
- Continued algorithm/reward tuning would make the thesis increasingly about MARL engineering rather than the intended UAV-recovery system question.

### Scope decision / next model
Adopt a two-stage recovery pipeline for the thesis core:
1. **Scout phase:** UAV fleet explores the failed region and builds the discovered demand map. All active UAVs are Scouts; learning/control focuses only on movement (or a deterministic scouting baseline can be used initially).
2. **Recovery phase:** freeze the discovered demand map and solve movement + Access/Relay allocation. Reuse the V0 service model.
3. Add battery/charging and rebalancing only to the recovery phase after the two-stage pipeline is stable.

This preserves the original scenario (unknown UE demand, UAV sensing, Access/Relay recovery, later energy-driven rebalancing) while removing the hardest simultaneous 3-role online coordination loop.

## Two-stage Stage A: Scout-only exploration
- Added `src/envs/stage_a_scout_env.py`.
- All 3 UAVs are Scouts; action space is movement only: `{N,S,E,W,Stay}` = 5 actions/UAV.
- Perfect sensing footprint remains radius 1 cell.
- Hidden UE truth is excluded from observation and centralized training state.
- Default learning reward is **newly revealed cells / 25 only**. Hidden UE count is deliberately *not* used in Stage-A reward, preventing hidden-demand leakage/memorization on the fixed-map sanity task.
- Added online `GreedyCoveragePolicy`; it only reads current positions + known mask and performs a 125-joint-move one-step search with a frontier-distance tie-breaker.
- Test suite now passes 15/15 tests.

### Stage-A baselines (8-slot scout budget)
- Random (500 episodes): final map discovery ~0.8231; final UE discovery ~0.7668.
- Greedy coverage: full map discovery = 1.0 and full UE discovery = 1.0 in 4 slots.
- This establishes a clean coverage ceiling of 1.0 for the fixed 5x5 Stage-A scenario (not claimed as a finite-horizon path oracle).

### Stage-A QMIX fixed-map sanity result
- 600-episode shared-policy QMIX run, seed 71, lr=3e-4, soft target tau=0.01.
- Best deterministic checkpoint: final map discovery = 0.96 (24/25 cells), mean discovery over the 8-slot trajectory = 0.775.
- Final UE discovery = 0.875 (7/8 UEs) because the sole missed cell is `(4,0)`, which contains one UE.
- Random -> QMIX -> heuristic progression is therefore approximately `0.823 -> 0.96 -> 1.0` final map discovery.
- Diagnostic trajectory shows QMIX repeatedly revisits already-known cells at slots 3 and 6 and never commits to the final bottom-left frontier cell. Increasing the scout budget to 10 without retraining does not fix this behavior; this is a learned local exploration policy, not simply insufficient horizon.
- Decision: Stage A is learnable and substantially cleaner than simultaneous Scout/Access/Relay V1. Do not over-tune it yet; next integrate the Stage-A discovered map into Stage-B Access/Relay recovery, then revisit generalization/random maps.

## First end-to-end two-stage handoff sanity check
- Added `scripts/eval_two_stage_pipeline.py` to pass Stage-A output directly into the validated V0 recovery model.
- Stage B starts from the UAV positions at the end of scouting and receives only the UE cells actually discovered in Stage A.
- KPI is reported against all 8 true UEs, so an undiscovered UE creates a hard ceiling on end-to-end service.

### Deterministic fixed-map results
- Greedy Scout -> Greedy Recovery:
  - Stage-A map discovery = 1.0 in 4 slots.
  - UE discovery = 1.0 (8/8).
  - Stage-B mean total served ratio = 0.94; final total served ratio = 1.0.
- QMIX Scout -> Greedy Recovery:
  - Stage-A map discovery = 0.96 in 8 slots.
  - UE discovery = 0.875 (7/8).
  - Stage-B serves 100% of the *discovered* UEs, but only 0.875 of all true UEs in the final state.
  - Stage-B mean total served ratio = 0.8225; final total served ratio = 0.875.

### Interpretation
This cleanly demonstrates the two-stage coupling: Stage-A sensing quality sets an information ceiling for Stage-B recovery. Recovery can be optimal with respect to the discovered map and still fail to restore all users if scouting misses demand. This gives the thesis a simple, interpretable decomposition rather than a simultaneous 3-role MARL problem.

## Two-stage learned pipeline: Stage A QMIX -> Stage B QMIX
- Added `StageBRecoveryEnv`, which receives only the UE map discovered by Stage A and the final Stage-A UAV positions.
- Stage-B training reward is served/discovered UEs; evaluation additionally reports served/true UEs so the Stage-A information ceiling remains visible.
- Test suite: 17/17 passing.
- Stage-B exact static oracle on the QMIX-Scout output can serve 7/7 discovered UEs; because Stage A found 7/8 true UEs, the end-to-end information ceiling is 0.875.
- Training Stage B QMIX from scratch on the Stage-A output performed poorly (~0.314 mean discovered-service at its best checkpoint).
- Transferring the learned V0 Access/Relay skill (`qmix_fixed_map_stable_s31_best.pt`) is substantially better. Fine-tuning reaches ~0.833 mean served among discovered UEs and 6/7 final discovered UEs.
- Learned end-to-end pipeline (QMIX Scout -> QMIX Recovery): Stage A discovers 7/8 UEs; Stage B final service is 6/7 discovered = 6/8 true = 0.75, with mean true-user service ~0.729.
- Reference decomposition:
  - Greedy Scout -> Oracle Recovery: final true-user service 1.0.
  - QMIX Scout -> Oracle Recovery: final true-user service 0.875 (pure Stage-A information loss).
  - QMIX Scout -> Greedy Recovery: final true-user service 0.875.
  - QMIX Scout -> QMIX Recovery: final true-user service 0.75 (one additional UE lost to the known Access/Relay coordination local optimum).
- Interpretation: decomposition exposes two separate bottlenecks quantitatively: scouting information loss (1/8 UE) and recovery coordination loss (1/7 of discovered UEs).

## V2: battery-driven UAV unavailability and reactive rebalancing
- Added `StageBBatteryEnv` on top of the learned two-stage pipeline.
- V2 intentionally isolates rebalancing from energy planning:
  - policy still controls only movement + {Access, Relay};
  - battery drains internally at a fixed rate;
  - UAV 1 has a low initial battery and becomes unavailable after slot 7;
  - the unavailable UAV remains out for the rest of the episode;
  - battery level itself is hidden in V2; the fleet only observes the changed availability after dropout.
- State/observation dimensionality is kept identical to Stage B by encoding an inactive UAV role as `[0,0]`. This permits direct transfer of the previously learned recovery checkpoint.
- Inactive UAVs have only one valid NoOp action and are excluded from access/relay connectivity calculations.
- Test suite now passes 21/21 tests.

### V2 static ceiling
- Stage A QMIX supplies 7 discovered UEs out of 8 true UEs.
- With 3 active UAVs, all 7 discovered UEs are statically serviceable.
- After one UAV drops, exhaustive search shows the exact 2-UAV static ceiling is 6/7 discovered UEs = 0.8571.
- One optimal 2-UAV configuration is Relay at `(1,2)` + Access at `(3,2)`.

### Zero-shot transfer vs fine-tuned rebalancing
- The old Stage-B recovery checkpoint, applied zero-shot to V2, collapses after dropout:
  - pre-drop mean service ≈ 0.7551;
  - post-drop mean service ≈ 0.0062;
  - final discovered-user service = 0;
  - no recovery.
- This confirms that ordinary recovery skill alone does not imply rebalancing skill.

### Fine-tuned QMIX V2 results
Three seeds were fine-tuned from the learned Stage-B recovery checkpoint.
- Seed 131: post-drop mean = 0.8571; final = 0.8571; recovery delay = 1 slot.
- Seed 132: post-drop mean = 0.8571; final = 0.8571; recovery delay = 1 slot.
- Seed 133: post-drop mean ≈ 0.8385; final = 0.8571; recovery delay = 2 slots.
- Multi-seed post-drop mean ≈ 0.8509 ± 0.0088.
- Final service across all seeds = 6/7 discovered UEs, matching the exact 2-UAV static ceiling.
- Recovery success rate = 100%; mean recovery delay ≈ 1.33 slots.

### No-rebalancing ablation
Using the same fine-tuned seed-131 policy before the dropout but freezing the two surviving UAVs afterward:
- post-drop mean service = 0;
- final service = 0.
With learned rebalancing enabled:
- post-drop mean service = 6/7 = 0.8571;
- final service = 6/7;
- recovery delay = 1 slot.
Trajectory interpretation: immediately before dropout the surviving Relay is one cell too far from the BS. After the low-battery UAV disappears, QMIX moves that Relay one cell east and restores the chain `Access -> Relay -> BS`, then holds the configuration.

### Interpretation / scope
V2 is a clean demonstration of **reactive role-aware fleet rebalancing** under a changing set of available UAVs. It does not yet solve anticipatory energy management because battery level is hidden and there is no charging/re-entry. The next V3 extension should add `Returning -> Charging -> Active` and expose only the minimal information needed (availability/return countdown, optionally normalized battery) while keeping the action space unchanged.

## V3: automatic return -> charging -> rejoin
- Added `StageBChargingEnv` in `src/envs/stage_b_charging_env.py`.
- Action space remains unchanged: `movement + {Access, Relay}`.
- Environment automatically handles `Active -> Returning -> Charging -> Active`.
- Return is triggered at a reserved-energy threshold; Returning UAV follows a deterministic shortest Manhattan path to one charging station; charging lasts a fixed number of slots; UAV then rejoins at full battery.
- V3 deliberately remains *reactive*: raw battery, charging countdown, and station information are hidden from the learned policy; a non-active UAV is represented by the same `[0,0]` role encoding used in V2, preserving checkpoint compatibility.
- Test suite after adding V3: 26/26 passing.

### V3 zero-shot and fine-tuning
- Zero-shot transfer of the V2 rebalancing checkpoint survives dropout better than the original recovery checkpoint but is not robust through the whole charging cycle: unavailable mean ~0.619, post-rejoin mean ~0.552, final 6/7.
- Fine-tuning on the full cycle reliably relearns the 2-UAV outage configuration; all four tested seeds recover after dropout.
- Hidden-countdown V3 seeds 141-144:
  - mean unavailable-period service ~0.757 ± 0.102;
  - mean post-rejoin service ~0.804 ± 0.138;
  - full 7/7 reintegration occurred in 1/4 seeds (seed 142).
- Seed 142 demonstrates the desired complete behavior:
  - 2-UAV period reaches the exact 6/7 static ceiling;
  - after rejoin, service reaches 7/7 one slot later;
  - post-rejoin mean ~0.932 and final service = 1.0 among discovered UEs.
- Rejoin ablation on seed 142:
  - normal rejoin: post-rejoin mean ~0.932, final 7/7;
  - preventing the UAV from rejoining: mean/final remain 6/7 = 0.857;
  - therefore the returning UAV provides measurable recovery capacity when the learned policy actually reintegrates it.

## V3.1 probe: expose ETA-to-active only
- Added `StageBChargingObservableEnv` as a minimal observability probe.
- Each UAV gets one additional normalized `ETA-to-active` feature for every teammate:
  - active -> 0;
  - returning -> Manhattan distance to station + fixed charging duration;
  - charging -> remaining charging slots.
- Raw battery remains hidden; therefore this is still not proactive energy optimization.
- Added a transfer utility that expands the V2 input layers while preserving all old state/agent-ID weights and initializes the new ETA weights at zero.
- Test suite: 28/28 passing.
- ETA-visible seeds 145-147:
  - unavailable-period service = 0.857 in all three seeds;
  - post-rejoin mean ~0.859 ± 0.003;
  - final service mean ~0.905 ± 0.067;
  - full 7/7 reintegration occurred in 1/3 seeds (seed 145).
- Interpretation: ETA visibility makes the outage phase much more consistent but does **not yet clearly solve the Access/Relay coordination barrier after rejoin**. With only three seeds, there is not enough evidence to justify adding ETA as a thesis requirement. Prefer the simpler hidden-countdown V3 unless later random-map experiments show a consistent benefit.

### Current V3 interpretation
- The full charging-cycle environment is learnable enough to demonstrate both directions of fleet-size change: `3 -> 2 -> 3`.
- Rebalancing after losing a UAV is robust; reintegration of the returning UAV remains a harder coordination problem and exhibits seed sensitivity, consistent with the local coordination barrier already observed in V0/Stage B.
- Do not add proactive battery scheduling yet. The next scientifically useful step is generalization/randomization (UE maps, failure UAV, charging timing) rather than more reward/algorithm tuning on the single fixed map.

## Generalization experiment G1/G2: random UE hotspot and random UAV starts
- Added `RandomizedStageAScoutEnv` and `RandomizedStageBRecoveryEnv` while preserving the original action/state dimensions where possible.
- Test suite after randomization scaffolding: 36/36 passing.

### Stage A zero-shot generalization (500-episode evaluation for Random/QMIX)
G1 = random hidden UE hotspot, fixed UAV starts:
- Random: final map ~0.831, final UE discovery ~0.866.
- Fixed-map QMIX: final map ~0.816, final UE discovery ~0.895.
- The fixed-map learner therefore does not improve spatial coverage over Random, although its fixed route happens to intersect hotspot samples well enough to discover somewhat more UEs.

G2 = random hidden UE hotspot + random UAV starts:
- Random: final map ~0.793, final UE discovery ~0.822, full-map rate ~0.024.
- Fixed-map QMIX: final map ~0.684, final UE discovery ~0.709, full-map rate ~0.040.
- QMIX trained directly on random hotspot + random starts improves over fixed-map transfer: final map ~0.776, final UE discovery ~0.803, full-map rate ~0.144, full-UE rate ~0.414.
- However, its average map coverage is still slightly below Random. This means Stage-A random-start exploration is not yet a robust learned skill; the policy gains some coordinated full-coverage episodes but has poor average robustness.
- A coverage-only observation probe (positions + known mask, without discovered-demand values) did not improve random-start learning, so simply removing the demand channel is not sufficient.
- Greedy Coverage remains a strong online reference and reaches 100% map/UE discovery in the sampled G1/G2 episodes (about 4--4.5 slots).

### Stage B random-demand generalization
- A fixed-map Recovery QMIX checkpoint almost completely fails zero-shot on random UE hotspots: mean service ~0.048, final ~0.045 over 300 maps.
- Added random-map Stage-B training with fixed handoff geometry. Starting from the old Recovery skill and fine-tuning across random UE hotspots substantially improves held-out performance:
  - random-map-trained QMIX: mean service ~0.395, final ~0.406 over 200 held-out maps;
  - Random baseline: mean ~0.140, final ~0.154 over 300 maps;
  - Greedy expert: mean ~0.895, final ~0.988 over 30 maps.
- Interpretation: random-map training converts Recovery from a fixed-layout memorized policy into a partially transferable skill, but a large gap to the expert remains. This is a useful generalization result rather than evidence that the current learner is production-ready.

### Scope implication
- Do not randomize everything at once. Use staged domain randomization:
  1. random UE hotspot with fixed infrastructure/start geometry;
  2. then random UAV starts;
  3. only later randomize failing-UAV identity / charging timing if needed.
- The current thesis core can still use the fixed-map environment for controlled ablations, while random-map experiments serve as a robustness/generalization section.

## Frozen-method audit: curriculum vs from-scratch on randomized C3
- Added `RandomizedStageBChargingEnv`: random UE hotspot, random UAV initial positions, random low-battery UAV, random dropout time, automatic return/charge/rejoin, while keeping the Stage-B observation/action dimensionality unchanged.
- Added `configs/final_curriculum_v1.json` and a controlled curriculum-vs-scratch script. Reward and QMIX architecture are fixed across all stages; the scratch control receives the same episode budget, epsilon resets, and replay resets.
- Three paired probe seeds (311-313), 480 training episodes each:
  - curriculum held-out mean service: ~0.2589 ± 0.0334;
  - from-scratch held-out mean service: ~0.2561 ± 0.0315;
  - paired delta: only ~+0.00275;
  - final service is effectively tied (~0.2642 curriculum vs ~0.2660 scratch).
- Conclusion: the proposed C1->C2->C3 curriculum has **no meaningful empirical advantage** under the current formulation. It should not be part of the core thesis method merely as a rescue technique. Keep it as an audit result unless later larger-scale experiments materially change the conclusion.

## Architecture audit: flat MLP-QMIX vs spatial-CNN-QMIX
- Motivation: random-map experiments showed strong fixed-geometry overfitting. The flat MLP ignores the 2-D inductive structure of UE, BS and UAV positions.
- Added `SpatialAgentQNetwork` as a drop-in QMIX agent encoder. It reconstructs six spatial channels from the existing observation: UE density, BS, Access UAVs, Relay UAVs, unavailable UAVs, and the controlled UAV position. Reward, QMIX mixer, action space, optimizer settings, and final C3 environment remain unchanged.
- Three paired seeds (323-325), only 100 training episodes each:
  - MLP mean service: ~0.1910 ± 0.0563;
  - Spatial mean service: ~0.3234 ± 0.0750;
  - paired improvement: ~+0.1324 absolute mean service.
  - MLP final service: ~0.1700 ± 0.0561;
  - Spatial final service: ~0.3400 ± 0.0726;
  - paired final-service improvement: ~+0.1700.
- Spatial encoder improves both unavailable-period and post-rejoin service in all three probe seeds. This is a much cleaner improvement than reward shaping/curriculum because it changes only the representation of an inherently spatial problem.
- Caveat: CNN training is slower on CPU, so longer multi-seed runs should be done only after this architecture decision is accepted.

## 2026-09-25 — Spatial-QMIX scale/generalization audit

### Frozen method
- Two-stage framework retained: Scout policy -> Recovery/Rebalancing policy.
- Stage-B learner: Spatial-QMIX (CNN map encoder + unchanged QMIX mixer).
- Reward/action space unchanged.
- Curriculum audit was not adopted: over 3 seeds it was essentially tied with direct final-task training.

### Longer 5x5 training
- 200-episode Spatial-QMIX held-out mean service across seeds 333/334/335: roughly 0.55, with very consistent results.
- A separate 300-episode run (seed 351) achieved mean service 0.538, suggesting that simply extending the same training budget does not automatically close the gap to a strong heuristic.

### Scalable fast baseline
Added `CoordinateGreedyPolicy`, a polynomial coordinate-local-search planner. It does not enumerate joint UAV placements and replans only when fleet availability changes.
100-episode benchmark:
- 5x5: random 0.137 mean service, coordinate-greedy 0.759; greedy latency ~1.27 ms/joint action.
- 7x7: random 0.174, coordinate-greedy 0.796; ~1.48 ms.
- 10x10: random 0.127, coordinate-greedy 0.795; ~1.84 ms.

### Resolution-scalability test
To avoid making large maps physically infeasible under one-relay-hop backhaul, 5x5/7x7/10x10 are treated as different grid resolutions of the same physical area. Access/backhaul radii, hotspot sigma, horizon, dropout timing and charging duration are scaled in cell units.

With an approximately equal 2160-transition training budget and two training seeds, evaluated on a common held-out distribution:
- 5x5 Spatial-QMIX: mean service ~0.343, final ~0.356, inference ~0.485 ms/joint action.
- 7x7 Spatial-QMIX: mean service ~0.459, final ~0.463, inference ~0.478 ms.
- 10x10 Spatial-QMIX: mean service ~0.233, final ~0.208, inference ~0.507 ms.

Interpretation:
- Inference latency remains essentially flat around 0.5 ms as grid resolution increases from 5x5 to 10x10.
- Training/sample efficiency degrades on 10x10 under the same transition budget.
- 7x7 outperforming 5x5 in this small 2-seed audit should not be interpreted as monotonic improvement; seed variance remains large.
- Spatial-QMIX consistently beats Random but remains substantially below the fast coordinate-greedy benchmark, so solution quality—not online inference speed—is currently the main bottleneck.

### Runtime engineering note (not a method contribution)
- Spatial map construction was vectorized; tests unchanged.
- Replay buffer changed from deque-backed random sampling to list-backed ring storage to keep random access O(1).
- Long larger-grid runs exposed a long-lived PyTorch/CPU backend stall. A restart-safe chunked trainer was added that preserves learner, target networks, optimizer, replay buffer, and RNG states exactly across process chunks. This is infrastructure only and does not alter the learning algorithm.

### Current decision
Do not add battery planning, extra reward shaping, or more environment features yet. The clean next methodological audit is whether another standard MARL family without QMIX's monotonic factorization (e.g. spatial MAPPO) closes the quality gap under the same final C3 task. Only one such alternative should be tested to avoid over-engineering.

## 2026-09-25 — Algorithm-family audit: Spatial-QMIX vs Spatial-MAPPO
- Added `SpatialMAPPO`, reusing the exact same six-channel CNN per-agent spatial encoder as Spatial-QMIX. The C3 random-dynamic environment, reward, action space, and observations are unchanged; MAPPO uses a centralized critic on global state.
- Added tests for masked actions, update dimensions, and correct GAE bootstrap at time-limit truncation. Test suite: 48/48 passing.
- Important correctness fix during the audit: C3 episodes end by horizon truncation, so MAPPO GAE now bootstraps from the final critic value rather than incorrectly treating the horizon as a terminal absorbing state. This did not reverse the result.
- Frozen 200-episode comparison, seeds 333-335, 100 held-out C3 episodes per seed:
  - Spatial-QMIX mean service: ~0.5114 ± 0.0488; final service ~0.5404 ± 0.0459.
  - Spatial-MAPPO mean service: ~0.2171 ± 0.0518; final service ~0.1892 ± 0.0588.
  - Spatial-QMIX unavailable-period service: ~0.4987; MAPPO ~0.1981.
  - Spatial-QMIX post-rejoin service: ~0.5324; MAPPO ~0.2107.
  - Spatial-QMIX inference: ~0.46 ms/joint action; Spatial-MAPPO ~1.01 ms/joint action on the same CPU setup.
- A 1000-episode MAPPO long-run was attempted only as a sample-efficiency sanity check but exceeded the current execution window before producing a complete auditable result; it is not used for conclusions.
- Decision: keep Spatial-QMIX as the thesis learner. MAPPO did not close the coordination/quality gap under a clean controlled comparison and adds no empirical justification for replacing QMIX. Do not continue algorithm shopping unless later evidence creates a specific need.
- Next focus: improve/scalably evaluate the frozen Spatial-QMIX solution on larger/random environments and benchmark against fast online heuristics; do not add battery planning or further reward/training tricks at this stage.

## 2026-09-25 — Official Spatial-QMIX resolution scalability experiment
Frozen final audit protocol:
- Stage-B C3 random-dynamic task (random UE hotspot, random UAV starts, random low-battery UAV, automatic return/charge/rejoin).
- Learner: Spatial-QMIX only; reward/action/architecture frozen.
- Training budget normalized to ~4,800 environment transitions per seed: 5x5=4,800, 7x7=4,788, 10x10=4,806.
- Three independent training seeds: 361, 362, 363.
- Deterministic evaluation: 100 held-out episodes on the same held-out scenario sequence for all training seeds within each grid.
- Online comparators: Random and polynomial Coordinate-Greedy. Exact brute-force oracle is intentionally excluded from large-grid online comparison.

Official mean-service results:
- 5x5: Spatial-QMIX 0.4881 ± 0.0892; Random 0.1544; Coordinate-Greedy 0.7638.
- 7x7: Spatial-QMIX 0.4981 ± 0.0471; Random 0.1541; Coordinate-Greedy 0.8084.
- 10x10: Spatial-QMIX 0.3043 ± 0.0394; Random 0.1082; Coordinate-Greedy 0.7693.

Final-service results:
- 5x5: Spatial-QMIX 0.4896 ± 0.1170; Greedy 0.8550.
- 7x7: Spatial-QMIX 0.5238 ± 0.0356; Greedy 0.8888.
- 10x10: Spatial-QMIX 0.3067 ± 0.0656; Greedy 0.8563.

Decision latency (joint action, CPU):
- Spatial-QMIX: ~0.479 ms (5x5), ~0.478 ms (7x7), ~0.531 ms (10x10).
- Coordinate-Greedy: ~1.214 ms, ~1.537 ms, ~1.977 ms respectively.

Interpretation:
- Spatial-QMIX consistently outperforms Random and preserves sub-millisecond inference across 5x5->10x10.
- With fixed interaction budget, 5x5 and 7x7 are similar (~0.49-0.50 mean service), while 10x10 drops to ~0.30, indicating sample-efficiency/generalization degradation at higher spatial resolution.
- Coordinate-Greedy remains substantially stronger in solution quality while still being fast, so the current RL advantage is latency/constant-time inference rather than service quality.
- Do not add battery planning or new reward tricks. The clean next question is sample complexity: whether 10x10 closes the quality gap when its training budget scales with state-space/resolution, versus fixed-budget training.

## 2026-09-25 — 10x10 sample-complexity probe
- Ran one frozen-method Spatial-QMIX long trajectory on the 10x10 C3 random-dynamic task with a single epsilon schedule defined for 178 episodes (~9.6k transitions). No reward/architecture/curriculum changes.
- Deterministic evaluation used the same held-out seed at intermediate checkpoints (30 episodes):
  - 2,430 transitions: mean service 0.2360; final 0.3042.
  - 4,806 transitions: mean service 0.2393; final 0.2417.
  - 7,236 transitions: mean service 0.3010; final 0.3500.
  - 9,612 transitions: mean service 0.3701; final 0.3917.
- Re-evaluating the 9,612-transition checkpoint on 100 held-out episodes gave mean service 0.4181 and final service 0.4438, with ~0.526 ms/joint-action inference.
- Training-window mean service also rose from ~0.165 (episodes 1-30) to ~0.35-0.38 in later 30-episode windows, though it remained noisy due to domain randomization and exploration.
- Interpretation: learning on 10x10 is delayed rather than immediately saturated. The curve is nearly flat through ~4.8k transitions, then improves materially between ~4.8k and ~9.6k. This is enough evidence to justify a longer checkpointed run on a user's local machine, but not to claim convergence or predict continued monotonic gains.
- Recommended long-run protocol: keep method frozen, checkpoint/evaluate every ~5k transitions, use a fixed held-out set (>=100 episodes), and stop if mean service improves <1-2 absolute percentage points across three consecutive checkpoints. Do not run for hours without intermediate evaluation.

### Long-run execution policy prepared
- Added a frozen-method 10x10 long-run driver (infrastructure only) with fixed transition-based epsilon decay, checkpoint/evaluation every 5k transitions, 100 held-out episodes on a fixed evaluation seed, and optional early stop after three consecutive checkpoint gains below 0.02 absolute mean service.
- Default maximum is 50k transitions. The purpose is to avoid multi-hour blind runs: inspect the learning curve at 5k intervals and continue only while held-out performance materially improves.
