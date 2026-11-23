"""Shared configuration for 6-pursuer/2-evader training and real flight."""

from dataclasses import dataclass
from typing import Tuple

TRAINING_ARENA_SIZE = 8.0
TRAINING_NUM_OBSTACLES = 10
TRAINING_EPISODES = 20000
TRAINING_MAX_STEPS = 1000
TRAINING_BATCH_SIZE = 256
TRAINING_SEEDS = (0, 1, 2, 3, 4)


@dataclass
class SixVTwoConfig:
    """Deployment contract in metres, seconds, and world-frame velocities."""

    num_pursuers: int = 6
    num_evaders: int = 2
    group_size: int = 3
    num_obstacles: int = TRAINING_NUM_OBSTACLES
    control_hz: float = 50.0
    arena_size: float = TRAINING_ARENA_SIZE
    flight_height: float = 1.0
    max_pursuer_speed: float = 0.35
    max_evader_speed: float = 0.35
    max_pursuer_accel: float = 0.15
    max_evader_accel: float = 0.15
    max_vertical_speed: float = 0.25
    height_kp: float = 1.5
    height_kd: float = 0.25
    hard_separation: float = 0.28
    boundary_margin: float = 0.30
    mocap_timeout: float = 0.25
    hidden_dim: int = 64
    observation_dim: int = 45
    future_steps: int = 10
    sequence_length: int = 10
    assignment_interval: int = 5
    assignment_switch_penalty: float = 0.05
    assignment_hysteresis: float = 0.02
    assignment_min_dwell_steps: int = 5
    pursuer_names: Tuple[str, ...] = (
        "cf1", "cf2", "cf3", "cf4", "cf5", "cf6"
    )
    evader_names: Tuple[str, ...] = ("cf7", "cf8")

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def all_names(self) -> Tuple[str, ...]:
        return self.pursuer_names + self.evader_names

    def validate(self) -> None:
        if len(self.all_names) != self.num_pursuers + self.num_evaders:
            raise ValueError(
                f"the Crazyflie name list must contain exactly "
                f"{self.num_pursuers + self.num_evaders} names"
            )
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
