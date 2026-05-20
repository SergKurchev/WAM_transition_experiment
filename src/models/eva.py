"""EVA model adapter.

Project: https://eva-project-page.github.io/
TODO: implement after studying the model API.
"""

from dds_interface import RobotState


class EVAModel:
    def __init__(self, checkpoint: str | None):
        if checkpoint is None:
            raise ValueError("WAM_CHECKPOINT must be set for EVA")
        # TODO: load model from checkpoint
        raise NotImplementedError("EVA integration not yet implemented")

    def __call__(self, state: RobotState) -> tuple[float, float, float]:
        """Run inference. Returns (vx, vy, wz)."""
        raise NotImplementedError
