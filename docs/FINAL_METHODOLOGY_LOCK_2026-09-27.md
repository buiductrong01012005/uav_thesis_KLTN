# Final methodology lock — 2026-09-27

This file locks the thesis methodology after the A1 identity-randomization audit. No further environment/reward tuning should be used to force a desired Access/Relay story.

## Core system model kept

- Fixed 1 km x 1 km, 20x20 physical grid, 50 m/cell.
- Five UAVs, one failed terrestrial BS, surviving BSs as gateways.
- UAV action remains movement x role: 5 movements x {Access, Relay} = 10 actions.
- One-hop relay only.
- Reward remains mean UE satisfaction min(R_u / d_u, 1).
- Traffic observation remains 5x5 aggregate + exact fleet state + agent ID.
- Automatic battery lifecycle remains Active -> Returning -> Charging -> Active.
- No multi-hop, proactive-return action, altitude action, reward shaping, or deterministic auto-Relay mechanism is added.

## Locked experimental interpretation

### A0 — static UE, no battery: positive learned movement result

A0 keeps roles fixed at 4 Access + 1 Relay and learns movement with Spatial-QMIX.

- Seed81 best validation mean utility: 0.69108.
- Reachable reference episode mean: 0.71097.
- Reference steady-state utility: 0.72074.
- Learned Relay movement is useful; freezing Relay movement reduced performance by about 10.47 percentage points in the prior ablation.

Interpretation: coordinated spatial repositioning is learnable and is the strongest positive RL result.

### A1 — static UE + battery + dynamic roles: negative battery-specific re-role result

Several audits showed that Relay can be structurally useful, especially with backhaul radius 4 cells. However, the learned policy did not provide clean evidence that temporal role reassignment was triggered specifically by battery depletion.

The final clean A1 audit randomizes UAV identity-position mapping, pins the low-battery residual UAV to the useful Relay anchor, trains from scratch, and keeps the same reward/radio/action model.

- Best seed81 validation utility: 0.38181 at about 86k transitions.
- In held-out matched low-battery/full-battery mappings, early handover occurred in both conditions.
- Therefore the handover decision is not uniquely attributable to battery depletion.

Interpretation: battery lifecycle works, dynamic roles are available, and reconfiguration occurs, but the thesis must NOT claim that MARL reliably learned battery-conditioned temporal role reassignment.

### A2 — mobile UE, no battery: positive movement-necessity result

Across seeds 11-13 in the staged necessity audit:

- Adaptive repositioning mean utility: 0.685.
- Frozen deployment mean utility: 0.584.
- Mean movement gain: +10.09 percentage points.
- Mobility did not independently create a need to change Relay count in the tested trajectories.

Interpretation: UE mobility justifies continued adaptive movement.

### A3 — mobile UE + battery: combined robustness / negative role result

Across seeds 11-13 in the staged audit:

- Adaptive movement remained useful (about +15.8 pp in the matched pre-event mobility branch).
- Dynamic role changes during the Relay-unavailable interval did not improve utility on average (about -3.1 pp with the tested controller/trajectory audit).

Interpretation: movement adaptation remains useful; combined mobility+battery does not provide robust evidence that extra role switching is beneficial under the current BS3 topology.

## Final thesis claim boundary

Supported:

1. Spatial-QMIX can learn useful coordinated UAV repositioning in the fixed recovery topology.
2. Relay usefulness depends on geometry/backhaul conditions; a Relay is not universally optimal.
3. Slow UE mobility creates a measurable need for continued repositioning.
4. The automatic battery return/charge/rejoin lifecycle is implemented and can be evaluated without adding it to the action space.

Not supported / must not be claimed:

1. Reliable battery-triggered temporal Access/Relay reassignment learned by MARL.
2. A3 being strictly better because of dynamic role switching.
3. Any result obtained by reward shaping or topology tuning solely to force Relay handover.

## Stop rule

A1 is closed. Do not add more tuning variants. Final work is evaluation, multi-seed confirmation of the positive A0 result, held-out checks, plots/tables, and thesis writing.
