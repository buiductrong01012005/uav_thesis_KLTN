# V1 Two-Stage Design

## Stage A: Scouting
- 5x5 failed-region grid, 3 UAVs, altitude fixed.
- Hidden UE distribution; simulator knows truth, UAVs do not.
- Every UAV is in Scout mode during this stage.
- Local action: movement only `{N,S,E,W,Stay}` (5 actions).
- Perfect sensing within Euclidean radius 1 grid cell.
- Shared discovered-demand map + known-cell mask.
- Primary objective: discover useful demand quickly.
- Initial implementation: fixed scout budget `T_scout`; later compare with an early-stop rule if all known M users are discovered.

## Stage B: Communication recovery
- Scouting output is frozen as the available demand map.
- Local action: movement + `{Access, Relay}` (10 actions).
- Direct backhaul or one Relay hop to surviving BS.
- Primary objective: served-user ratio; throughput can be added later.
- Reuse the validated V0 environment/learner structure.

## Later V2
- Add battery state, automatic return/charge/rejoin.
- Rebalancing occurs only in Stage B via new movement + Access/Relay decisions as UAV availability changes.

## Why this split
It removes the empirically difficult simultaneous conjunction
`Scout + Access + Relay` from each slot while preserving the thesis scenario and making every component independently testable.
