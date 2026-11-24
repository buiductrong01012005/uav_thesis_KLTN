"""Atomic checkpoint/protocol writes shared by the final training runner."""

import hashlib
import json
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def model_payload(learner):
    return dict(
        agent=learner.agent.state_dict(),
        mixer=learner.mixer.state_dict(),
        target_agent=learner.target_agent.state_dict(),
        target_mixer=learner.target_mixer.state_dict(),
        optimizer=learner.optimizer.state_dict(),
    )
