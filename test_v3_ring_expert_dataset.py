import json

import numpy as np
import pytest
import torch

from .stage_env import StageCoalitionEnv
from .v3_ring_expert_dataset import (
    assigned_member_distances,
    build_expert_record,
    initial_assignments,
    repair_dead_tasks,
    validate_training_seeds,
    write_shard,
    rollout_assignment,
    validate_shard_manifests,
)


def test_expert_observation_and_actions_share_hard_assignment():
    """Catches soft or stale membership in Actor observations."""
    env = StageCoalitionEnv(max_pursuers=6, max_evaders=3, num_obstacle=10, seed=1)
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=72000)
    assignment = np.array([0, 0, 0, 1])
    encoded = {
        "pursuer_embeddings": torch.zeros(6, 4),
        "task_embeddings": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]]),
        "predictions": torch.zeros(3, 10, 2),
        "uncertainty": torch.zeros(3),
    }
    expert_action = np.zeros((4, 2), dtype=np.float32)
    record = build_expert_record(env, assignment, encoded, lambda obs: torch.zeros((6, 2)), expert_action)
    obs = np.asarray(record["obs"])
    assert record["assignment"] == [0, 0, 0, 1]
    assert env.assignment[:4].tolist() == [0, 0, 0, 1]
    assert np.allclose(obs[:3, 29], 0.75)
    assert obs[3, 29] == pytest.approx(0.25)
    assert np.asarray(record["expert_action"]).shape == (4, 2)
    assert np.asarray(record["original_action"]).shape == (4, 2)


def test_farthest_member_uses_assigned_coalition_even_if_global_nearest_differs():
    """Catches global nearest-three selection instead of assigned-member selection."""
    pursuers = np.array([[0.1, 0], [0.2, 0], [5.0, 0], [10.1, 0]])
    evaders = np.array([[0., 0.], [10., 0.]])
    distances, farthest = assigned_member_distances(pursuers, evaders, [0, 0, 0, 1], [True, True])
    assert distances == pytest.approx([0.1, 0.2, 5.0, 0.1])
    assert farthest == 2


def test_searches_all_sixteen_initial_assignments_and_repairs_dead_task():
    candidates = initial_assignments()
    assert len(candidates) == 16
    assert len({tuple(row) for row in candidates}) == 16
    assert tuple(candidates[0]) == (0, 0, 0, 0)
    assert tuple(candidates[-1]) == (1, 1, 1, 1)
    assert repair_dead_tasks([0, 1, 0, 1], [False, True]).tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("seed_record", [70086, 70087, 73000, 74000, {"seed": 72000, "parent_seed": 70086}])
def test_training_seed_validator_rejects_sealed_derived_and_nontraining(seed_record):
    with pytest.raises(ValueError):
        validate_training_seeds([seed_record], assigned_seeds=[72000])


def test_training_seed_validator_rejects_unassigned_and_duplicates():
    with pytest.raises(ValueError):
        validate_training_seeds([72001], assigned_seeds=[72000])
    with pytest.raises(ValueError):
        validate_training_seeds([72000, 72000], assigned_seeds=[72000])
    assert validate_training_seeds([72000], assigned_seeds=[72000]) == [72000]


def test_shard_saves_only_successful_finite_actions_and_failure_metadata(tmp_path):
    sample = {
        "seed": 72000, "assignment": [0, 0, 0, 1], "obs": [[0.0]],
        "expert_action": [[0.5, -0.5]] * 4,
        "original_action": [[0.0, 0.0]] * 4,
        "farthest_member": 2, "assigned_distance": [1., 2., 3., 4.],
        "captured": 1, "full_capture": False,
    }
    trial = {"seed": 72000, "initial_assignment": [0, 0, 0, 1], "captured": 1,
             "full_capture": False, "outcome": "timeout", "steps": 1, "records": [sample]}
    failed = {"seed": 72000, "initial_assignment": [0, 0, 0, 0], "captured": 0,
              "full_capture": False, "outcome": "timeout", "steps": 1, "records": []}
    manifest = write_shard(tmp_path / "shard", seeds=[72000], assigned_seeds=[72000],
                           trials=[trial, failed], source_sha256="abc")
    saved = json.loads((tmp_path / "shard" / "trajectories.jsonl").read_text())
    failure_rows = [json.loads(line) for line in (tmp_path / "shard" / "failures.jsonl").read_text().splitlines()]
    assert len(saved["records"]) == 1
    assert len(failure_rows) == 1
    assert manifest["successful_seeds"] == [72000]
    assert manifest["failed_attempts"] == 1
    assert manifest["source_sha256"] == "abc"


def test_shard_rejects_nonfinite_or_unbounded_action(tmp_path):
    record = {"seed": 72000, "assignment": [0, 0, 0, 1], "obs": [[0.0]],
              "expert_action": [[np.nan, 0.0]] * 4, "original_action": [[0., 0.]] * 4,
              "farthest_member": 0, "assigned_distance": [1.] * 4,
              "captured": 1, "full_capture": False}
    trial = {"seed": 72000, "initial_assignment": [0, 0, 0, 1], "captured": 1,
             "full_capture": False, "outcome": "timeout", "steps": 1, "records": [record]}
    with pytest.raises(ValueError):
        write_shard(tmp_path / "bad", seeds=[72000], assigned_seeds=[72000], trials=[trial], source_sha256="abc")
    record["expert_action"] = [[1.01, 0.0]] * 4
    with pytest.raises(ValueError):
        write_shard(tmp_path / "bad", seeds=[72000], assigned_seeds=[72000], trials=[trial], source_sha256="abc")


def test_one_step_rollout_uses_environment_three_value_step_contract():
    class Encoder:
        def reset(self):
            pass

        def encode(self, snap, differentiable=False):
            return {"pursuer_embeddings": torch.zeros(6, 4),
                    "task_embeddings": torch.zeros(3, 4),
                    "predictions": torch.zeros(3, 10, 2),
                    "uncertainty": torch.zeros(3)}

    trial = rollout_assignment(72000, [0, 0, 0, 1], Encoder(),
                               lambda obs: torch.zeros((6, 2)),
                               assigned_seeds=[72000], max_steps=1)
    assert trial["steps"] == 1
    assert trial["outcome"] != "invalid_sample"
    assert len(trial["records"]) == 1
    with pytest.raises(ValueError):
        rollout_assignment(72001, [0, 0, 0, 1], Encoder(),
                           lambda obs: torch.zeros((6, 2)),
                           assigned_seeds=[72000], max_steps=1)


def test_merge_validator_rejects_duplicate_seed_source_mismatch_and_tampering(tmp_path):
    first = write_shard(tmp_path / "a", [72000], [72000, 72001], [], "source-a")
    second = write_shard(tmp_path / "b", [72001], [72000, 72001], [], "source-a")
    paths = [tmp_path / "a" / "artifact_manifest.json", tmp_path / "b" / "artifact_manifest.json"]
    merged = validate_shard_manifests(paths, [72000, 72001], "source-a")
    assert merged["seeds"] == [72000, 72001]
    with pytest.raises(ValueError):
        validate_shard_manifests([paths[0], paths[0]], [72000, 72001], "source-a")
    with pytest.raises(ValueError):
        validate_shard_manifests(paths, [72000, 72001], "source-b")
    (tmp_path / "b" / "trajectories.jsonl").write_text("tamper\n")
    with pytest.raises(ValueError):
        validate_shard_manifests(paths, [72000, 72001], "source-a")


def test_merge_accepts_invalid_sample_failure_metadata_even_after_capture(tmp_path):
    failed = {"seed": 72000, "initial_assignment": [0, 0, 0, 1],
              "captured": 1, "full_capture": False, "outcome": "invalid_sample",
              "steps": 2, "error": "nonfinite action", "records": []}
    write_shard(tmp_path / "shard", [72000], [72000], [failed], "source-a")
    merged = validate_shard_manifests(
        [tmp_path / "shard" / "artifact_manifest.json"], [72000], "source-a",
    )
    assert merged["successful_seeds"] == []
