"""Fixed-distribution training and deterministic evaluation protocol."""

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

Scale = Tuple[int, int]
DEFAULT_SCALES: Tuple[Scale, ...] = ((6, 2), (4, 2), (5, 2), (6, 3))
DEFAULT_SCALE_WEIGHTS: Tuple[float, ...] = (0.7, 0.1, 0.1, 0.1)
DEFAULT_D_CAP: float = 2.0
DEFAULT_HOLD_STEPS: int = 5
EVAL_SEED_OFFSET: int = 977


def resolve_eval_seed(train_seed: int, eval_seed=None) -> int:
    """Return one fixed validation base seed, independent of training episode."""
    return int(train_seed) + EVAL_SEED_OFFSET if eval_seed is None else int(eval_seed)


def normalize_scale_weights(weights: Sequence[float], count: int) -> np.ndarray:
    """Validate and normalize fixed scale probabilities."""
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError(f"expected {count} scale weights, got shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scale weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("at least one scale weight must be positive")
    return values / total


class FixedScaleSampler:
    """Sample episode scales from one distribution for the full training run."""

    def __init__(
        self,
        scales: Sequence[Scale] = DEFAULT_SCALES,
        weights: Sequence[float] = DEFAULT_SCALE_WEIGHTS,
        seed: int = 0,
    ) -> None:
        self.scales = tuple((int(p), int(e)) for p, e in scales)
        if not self.scales:
            raise ValueError("at least one scale is required")
        self.weights = normalize_scale_weights(weights, len(self.scales))
        self._rng = np.random.default_rng(seed)

    def sample(self) -> Scale:
        index = int(self._rng.choice(len(self.scales), p=self.weights))
        return self.scales[index]


def stratified_seeds(
    scales: Iterable[Scale] = DEFAULT_SCALES,
    episodes_per_scale: int = 10,
    base_seed: int = 0,
) -> Dict[Scale, Tuple[int, ...]]:
    """Return scale-separated deterministic evaluation seeds."""
    if episodes_per_scale <= 0:
        raise ValueError("episodes_per_scale must be positive")
    result: Dict[Scale, Tuple[int, ...]] = {}
    for scale_index, scale in enumerate(scales):
        key = (int(scale[0]), int(scale[1]))
        result[key] = tuple(
            int(base_seed + scale_index * 100000 + episode_index)
            for episode_index in range(episodes_per_scale)
        )
    return result


def checkpoint_score(per_scale_capture_rates: Dict[Scale, float]) -> float:
    """Weight 6v2 performance while retaining macro-scale generalization."""
    missing = [scale for scale in DEFAULT_SCALES if scale not in per_scale_capture_rates]
    if missing:
        raise ValueError(f"missing capture rates for scales: {missing}")
    macro = float(np.mean([per_scale_capture_rates[s] for s in DEFAULT_SCALES]))
    return 0.7 * float(per_scale_capture_rates[(6, 2)]) + 0.3 * macro


def parse_scale_weights(value: str) -> Tuple[float, ...]:
    """Parse the four comma-separated CLI scale weights."""
    try:
        weights = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("scale weights must be comma-separated numbers") from exc
    normalized = normalize_scale_weights(weights, len(DEFAULT_SCALES))
    return tuple(float(item) for item in normalized)
