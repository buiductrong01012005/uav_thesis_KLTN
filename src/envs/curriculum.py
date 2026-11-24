"""Fixed curriculum conditions shared by training and the environment."""


def condition_for(step, arm, protocol):
    """All boundaries are episode boundaries; gates never control the schedule."""
    ends = protocol["stage_ends"]
    stage = next((f"S{i + 1}" for i, end in enumerate(ends) if step < end), "S4")
    if arm.startswith("D"):
        stage = "D"
    condition = dict(
        stage=stage,
        substage=stage,
        active_count=5,
        mobility=True,
        battery=True,
        starts="B",
        radius=None,
        jitter=0,
        stay_only=False,
    )
    if stage == "S1":
        condition.update(
            active_count=2, mobility=False, battery=False, starts="fixture", stay_only=True
        )
    elif stage == "S2":
        second = step >= (ends[0] + ends[1]) // 2
        condition.update(
            substage="S2b" if second else "S2a",
            active_count=2,
            mobility=False,
            battery=False,
            starts="fixture",
            radius=3 if second else 1,
            jitter=2 if second else 1,
        )
    elif stage == "S3":
        part = min(3, (step - ends[1]) // ((ends[2] - ends[1]) // 4))
        condition.update(
            substage=f"S3{chr(97 + part)}",
            active_count=2 if part == 0 else 5,
            mobility=part >= 2,
            battery=part >= 3,
            starts="intermediate" if part == 0 else "B",
        )
    return condition


def task_signature(condition):
    # S3d and S4 are the same physical task: a label change does not clear replay.
    return tuple(
        (key, value) for key, value in sorted(condition.items()) if key not in {"stage", "substage"}
    )


FULL_CONDITION = {
    "stage": "S4",
    "substage": "S4",
    "active_count": 5,
    "mobility": True,
    "battery": True,
    "starts": "B",
    "radius": None,
    "jitter": 0,
    "stay_only": False,
}
