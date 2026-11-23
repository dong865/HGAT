"""Conservative world-frame safety filter for eight Crazyflies."""

import numpy as np

from .config import SixVTwoConfig


def filter_velocity(
    positions: np.ndarray,
    velocity: np.ndarray,
    origin: np.ndarray,
    config: SixVTwoConfig,
) -> np.ndarray:
    """Suppress closing/boundary velocities before sending radio commands."""
    safe = np.asarray(velocity, dtype=np.float32).copy()
    half = config.arena_size / 2.0
    for index, position in enumerate(positions):
        relative = position[:2] - origin[:2]
        for axis in range(2):
            if relative[axis] < -half + config.boundary_margin:
                safe[index, axis] = max(safe[index, axis], 0.05)
            elif relative[axis] > half - config.boundary_margin:
                safe[index, axis] = min(safe[index, axis], -0.05)

    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            delta = positions[first, :2] - positions[second, :2]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-6 or distance >= config.hard_separation:
                continue
            direction = delta / distance
            closing = float(np.dot(safe[first] - safe[second], direction))
            if closing < 0.0:
                correction = -0.5 * closing * direction
                safe[first] += correction
                safe[second] -= correction
            push = 0.12 * (config.hard_separation - distance) / config.hard_separation
            safe[first] += push * direction
            safe[second] -= push * direction
    return safe

