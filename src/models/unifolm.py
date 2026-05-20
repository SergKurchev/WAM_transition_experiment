"""UnifoLM-WMA-0 adapter.

Repo: https://github.com/unitreerobotics/unifolm-world-model-action
TODO: implement after studying the model API.
"""

from dds_interface import RobotState


class UnifoLMModel:
    def __init__(self, checkpoint: str | None):
        if checkpoint is None:
            raise ValueError("WAM_CHECKPOINT must be set for UnifoLM")
        # TODO: load model from checkpoint
        raise NotImplementedError("UnifoLM integration not yet implemented")

    def __call__(self, state: RobotState) -> tuple[float, float, float]:
        """Run inference. Returns (vx, vy, wz)."""
        raise NotImplementedError
