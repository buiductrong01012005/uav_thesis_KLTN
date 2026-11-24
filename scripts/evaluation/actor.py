"""Strict checkpoint actor loading; independent of replication plans and training budget."""

import hashlib
import numpy as np
from scripts.source_integrity import digest
from scripts.evaluation import run_fourarm400k_heuristics as core


class FrozenActor:
    def __init__(self, checkpoint_bytes, train_protocol, model):
        import io
        import torch
        from src.rl.fourarm_qmix import FourArmActor

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        expected = dict(
            arm=train_protocol["arm"],
            selected_step=model["selected_step"],
            selection_metric="reward_total",
            protocol_sha256=digest(train_protocol),
            reward_version=train_protocol["reward_version"],
            training_env_version=train_protocol["training_env_version"],
            obs_dim=173,
            state_dim=156,
            n_actions=train_protocol["policy_actions"],
            fair_config=train_protocol["fair_config"],
        )
        expected.update(
            architecture=train_protocol["architecture"],
            resolved_config_sha256=train_protocol["resolved_config_sha256"],
        )
        if any(saved.get(k) != v for k, v in expected.items()) or not np.isclose(
            saved["validation_score"], model["validation_score"], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Checkpoint payload does not match frozen validation selection")
        self.agent = FourArmActor(train_protocol["policy_actions"], train_protocol["architecture"])
        self.agent.load_state_dict(saved["agent"], strict=True)
        self.agent.eval()
        self.n_actions = train_protocol["policy_actions"]

    @property
    def last_stats(self):
        return {}

    def act(self, obs, mask):
        import torch

        obs = np.asarray(obs)
        mask = np.asarray(mask, dtype=bool)
        if (
            obs.shape != (5, 173)
            or mask.shape != (5, self.n_actions)
            or not np.isfinite(obs).all()
            or not mask.any(1).all()
        ):
            raise ValueError("Invalid actor observation or mask")
        with torch.no_grad():
            q = self.agent(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
        if not np.isfinite(q).all():
            raise FloatingPointError("Nonfinite Q values")
        q = np.where(mask, q, -1e9)
        return np.argmax(q, axis=1).astype(int).tolist()


def runtime():
    import torch

    return dict(
        **core.runtime(),
        torch=str(torch.__version__),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
    )
