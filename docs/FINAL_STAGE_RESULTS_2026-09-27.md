# Final staged results — 2026-09-27

This file records the locked staged evidence used in the thesis. A2/A3 are staged necessity/robustness audits, not learned-policy final-performance tables.

## A0 — static UE, no battery

Main learned result. Spatial-QMIX seed81 best validation = 0.69108. Final acceptance still requires independent training seeds 82 and 83 plus held-out evaluation per best checkpoint.

Pilot held-out check with the seed81 best checkpoint over three unseen layouts:
- RL: 0.3798 ± 0.0818
- frozen deployment: 0.1984 ± 0.0000
- gain: +18.14 ± 8.18 percentage points

## A1 — static UE + battery + dynamic roles

Closed as a negative battery-specific re-role result.

The clean variant trains from scratch with randomized UAV identity-position mapping and backhaul radius 4. The best seed81 validation utility is 0.38181. In matched low-battery/full-battery held-out audits, early handover occurs in both conditions, so temporal handover cannot be attributed specifically to battery depletion.

Do not tune A1 further.

## A2 — mobile UE, no battery

Seeds: 11, 12, 13.

- Adaptive repositioning: **0.6847 ± 0.0136**
- Frozen deployment: **0.5837 ± 0.0361**
- Movement gain: **+10.09 ± 2.89 pp**
- Relay-count changes along audited trajectories: **0.0 ± 0.0**

Interpretation: UE mobility clearly motivates continued UAV repositioning. It does not independently motivate temporal changes in Relay count in these trajectories.

## A3 — mobile UE + battery

Seeds: 11, 12, 13.

Mobility branch:
- Adaptive repositioning: **0.6722 ± 0.0313**
- Frozen deployment: **0.5143 ± 0.0580**
- Movement gain: **+15.78 ± 7.28 pp**

Battery/role branch:
- Fixed-role unavailable utility: **0.6375 ± 0.0511**
- Dynamic-role unavailable utility: **0.6065 ± 0.0190**
- Dynamic re-role gain: **-3.10 ± 5.36 pp**
- Dynamic post-rejoin utility: **0.6670 ± 0.0717**

Interpretation: movement adaptation remains useful when mobility and battery are combined. Extra dynamic role switching is not robustly beneficial under the locked BS3 topology.

## Locked claim boundary

Supported:
1. Spatial-QMIX learns useful coordinated repositioning in A0.
2. UE mobility creates a measurable need for continued repositioning.
3. Relay usefulness is geometry/backhaul dependent.
4. Return -> Charging -> Rejoin lifecycle works and can be evaluated without adding proactive return to the action space.

Not supported:
1. Battery-specific temporal Access/Relay reassignment learned reliably by MARL.
2. Dynamic role switching improves the combined A3 scenario on average.
3. Any stronger claim obtained by further reward/topology tuning to force a positive result.

## Remaining final-evaluation work

Only the main A0 learned result remains incomplete for final acceptance:
- train independent seeds 82 and 83 with the locked A0 runner;
- preserve each best validation checkpoint;
- run held-out static-layout evaluation for each best checkpoint;
- report training-seed mean ± std and held-out mean ± std.

No further A1/A2/A3 tuning is permitted under the methodology lock.
