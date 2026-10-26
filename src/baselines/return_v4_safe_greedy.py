"""Isolated one-slot previews for Return-v4; never mutate the executed env.

Legacy preview helpers only restore positions/roles, which is insufficient for
Return-v4 route memory, energy overrides and status. These new wrappers score
the actual immediate service reward on an isolated clone, with candidate slot
actions. The immutable UE tape is shared read-only to avoid copying a 10k tape
for every candidate. No future multi-step trajectory is searched.
"""
from copy import deepcopy

from src.baselines.recovery_aware_all_access_greedy import RecoveryAwareAllAccessGreedy
from src.baselines.recovery_aware_role_greedy import RecoveryAwareExhaustiveGreedy, RecoveryAwareFixedMixGreedy


class IsolatedPreview:
    preview_calls = 0

    def _score(self, role_vec, moves=None):
        env = self.env
        tape = getattr(env, "_ue_tape", None)
        memo = {id(tape): tape} if tape is not None else {}
        clone = deepcopy(env, memo)
        moves = moves or {}
        actions = [int(clone.encode_action(moves.get(i, 4), int(role_vec[i])))
                   if clone.status[i] == 0 else int(clone.encode_action(4, 0))
                   for i in range(clone.cfg.num_uavs)]
        mask = clone.get_action_mask()
        assert all(mask[i, a] for i, a in enumerate(actions))
        self.preview_calls += 1
        return float(clone.step(actions)["reward"])


class V4AllAccessRecovery(IsolatedPreview, RecoveryAwareAllAccessGreedy):
    pass


class V4FixedMixRecovery(IsolatedPreview, RecoveryAwareFixedMixGreedy):
    pass


class V4DynamicRecovery(IsolatedPreview, RecoveryAwareExhaustiveGreedy):
    pass


def make_greedy(label, env, patience=3):
    if label == "5A-Recovery":
        return V4AllAccessRecovery(env, zero_service_patience=patience)
    if label in ["4A1R-Recovery", "3A2R-Recovery"]:
        return V4FixedMixRecovery(env, composition=label.split("-")[0], zero_service_patience=patience)
    if label == "Dynamic-Recovery":
        return V4DynamicRecovery(env, zero_service_patience=patience)
    raise ValueError(label)
