"""Deterministic mid-episode pursuer online/offline interventions."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .stage_env import StageCoalitionEnv


@dataclass
class TeamSizeChange:
    mode: str
    scenario_seed: int
    event_step: int = 150
    member_ids: List[int] = field(default_factory=list)
    applied: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"control", "drop1", "drop2", "join1", "hold1"}:
            raise ValueError(f"unsupported team-size change mode {self.mode!r}")
        if self.event_step < 0:
            raise ValueError("event_step must be non-negative")

    def prepare(self, env: StageCoalitionEnv) -> None:
        """Choose intervention members and establish the pre-event team."""
        rng = np.random.default_rng(int(self.scenario_seed) ^ 0x5A17)
        count = {
            "control": 0, "drop1": 1, "drop2": 2,
            "join1": 1, "hold1": 1,
        }[self.mode]
        self.member_ids = (
            sorted(int(i) for i in rng.choice(env.num_pursuers, count, replace=False))
            if count else []
        )
        if self.mode in {"join1", "hold1"}:
            env.set_pursuer_offline(self.member_ids[0])

    def apply_if_due(self, env: StageCoalitionEnv, step: int) -> bool:
        if self.applied or int(step) < self.event_step:
            return False
        if self.mode.startswith("drop"):
            for member in self.member_ids:
                env.set_pursuer_offline(member)
        elif self.mode == "join1":
            env.set_pursuer_online(self.member_ids[0])
        self.applied = True
        return True
