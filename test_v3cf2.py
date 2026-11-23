import math

import numpy as np
import torch

from .v3cf2_dataset import (
    NearCaptureTracker,
    build_lookahead_features,
    near_capture_potential_score,
)
from .train_v3cf2_value import split_by_seed, train_value_offline


def _near_snapshot():
    pursuers = np.array([
        [0.9, 0.0, 0.0],
        [-0.45, 0.9 * math.sqrt(3.0) / 2.0, 0.0],
        [-0.45, -0.9 * math.sqrt(3.0) / 2.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    tasks = np.array([[0.0, 0.0, 0.0], [4.0, 1.0, 0.0]])
    return {
        "positions": np.vstack([pursuers, tasks]),
        "num_pursuers": 4,
        "num_evaders": 2,
        "task_mask": np.ones(2, dtype=bool),
    }


def test_near_capture_requires_five_consecutive_steps():
    tracker = NearCaptureTracker(radius=1.0, required_hold=5)
    snapshot = _near_snapshot()
    assignment = np.array([0, 0, 0, 1])
    for step in range(1, 5):
        assert tracker.update(snapshot, assignment, step) is None
    assert tracker.near_capture_step is None
    assert tracker.update(snapshot, assignment, 5) == 5
    assert tracker.near_capture_step == 5


def test_near_capture_streak_resets_after_geometry_break():
    tracker = NearCaptureTracker(radius=1.0, required_hold=5)
    snapshot = _near_snapshot()
    assignment = np.array([0, 0, 0, 1])
    for step in range(1, 5):
        tracker.update(snapshot, assignment, step)
    broken = _near_snapshot()
    broken["positions"][0, 0] = 1.1
    assert tracker.update(broken, assignment, 5) is None
    for step in range(6, 10):
        assert tracker.update(snapshot, assignment, step) is None
    assert tracker.update(snapshot, assignment, 10) == 10


def test_near_capture_label_orders_early_late_and_unachieved():
    early = near_capture_potential_score(
        {"near_capture_step": 4, "delta_captured": 0, "full_capture": False},
        horizon=100,
    )
    late = near_capture_potential_score(
        {"near_capture_step": 30, "delta_captured": 2, "full_capture": True},
        horizon=100,
    )
    missing = near_capture_potential_score(
        {"near_capture_step": None, "delta_captured": 2, "full_capture": True},
        horizon=100,
    )
    assert early > late > missing


def test_lookahead_features_match_distance_time_and_margin():
    pursuers = np.array([
        [2.0, 0.0], [-1.0, math.sqrt(3.0)], [-1.0, -math.sqrt(3.0)], [4.0, 0.0],
    ])
    tasks = np.array([[0.0, 0.0], [10.0, 0.0]])
    goals = np.array([[3.0, 0.0], [11.0, 0.0]])
    features = build_lookahead_features(
        pursuers, tasks, goals, np.array([0, 0, 0, 1]),
        v_max=0.5, v_max_e=0.25,
    )
    np.testing.assert_allclose(features[0], [4.0, 12.0, 8.0, 2.0 / 3.0], atol=1e-6)
    np.testing.assert_allclose(features[1], [12.0, 4.0, -8.0, 2.0], atol=1e-6)


def _synthetic_records():
    records = []
    for seed in range(10):
        near_steps = [5 + seed % 3, 25 + seed % 3, None]
        features = np.zeros((3, 2, 81), dtype=np.float32)
        # Earlier near-capture has a shorter estimated closure time.
        features[:, 0, 77] = [0.5, 2.0, 4.0]
        labels = [near_capture_potential_score(
            {"near_capture_step": step, "delta_captured": 0, "full_capture": False},
            horizon=100,
        ) for step in near_steps]
        records.append({
            "state_id": f"synthetic-{seed}", "seed": seed,
            "features": features.tolist(),
            "near_capture_step": near_steps,
            "min_d_max": [0.4, 1.5, 3.0],
            "G_H": labels, "alive": np.ones((3, 2), dtype=bool).tolist(),
            "assignments": [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
            "modes": [[3, 1], [2, 2], [1, 3]],
        })
    return records


def test_small_data_training_exceeds_ninety_percent_ranking_accuracy():
    result = train_value_offline(
        _synthetic_records(), output=None, epochs=80, hidden_dim=32,
        learning_rate=0.03, batch_size=4, device="cpu", seed=7,
    )
    accuracy = result["history"][-1]["validation"]["pairwise_ranking_accuracy"]
    assert accuracy is not None and accuracy > 0.90


def test_seed_partitions_are_disjoint():
    train, validation, validation_seeds = split_by_seed(_synthetic_records(), 0.2)
    train_seeds = {row["seed"] for row in train}
    valid_seeds = {row["seed"] for row in validation}
    assert train_seeds.isdisjoint(valid_seeds)
    assert valid_seeds == set(validation_seeds)
    assert len(validation_seeds) == 2
