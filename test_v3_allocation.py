import numpy as np
import pytest
import torch

from .coalition_value import generate_candidate_assignments
from .v3_allocation import (
    form_demand_guided_alliance,
    load_value_checkpoint,
    normalized_demand_from_candidates,
    rank_value_candidates,
)


def test_normalized_demand_conserves_four_agents_and_keeps_tasks_nonempty():
    candidates = np.stack(generate_candidate_assignments(4, 2))
    values = torch.zeros(len(candidates))
    preferred = np.flatnonzero(np.all(candidates == [0, 0, 0, 1], axis=1))[0]
    values[preferred] = 20.0

    demand, capacity = normalized_demand_from_candidates(
        values, candidates, num_tasks=2, temperature=0.1,
    )

    assert torch.allclose(demand.sum(), torch.tensor(1.0))
    assert torch.all(capacity >= 1.0)
    assert torch.allclose(capacity.sum(), torch.tensor(4.0))
    assert capacity[0] > 2.99
    assert capacity[1] < 1.01


def test_equal_candidate_values_produce_symmetric_two_plus_two_demand():
    candidates = np.stack(generate_candidate_assignments(4, 2))

    demand, capacity = normalized_demand_from_candidates(
        torch.zeros(len(candidates)), candidates, num_tasks=2,
    )

    assert torch.allclose(demand, torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(capacity, torch.tensor([2.0, 2.0]), atol=1e-6)


class _PreferTaskZeroMembers(torch.nn.Module):
    def forward(self, features):
        # First geometric feature is coalition-size / total-team-size.
        return features[:, 0, -7]


def test_value_ranking_selects_highest_scoring_nonempty_assignment():
    candidates = np.stack(generate_candidate_assignments(4, 2))
    pursuer_embeddings = torch.zeros(4, 3)
    task_embeddings = torch.zeros(2, 3)
    pursuer_positions = np.array([
        [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 0.0],
    ])
    task_positions = np.array([[0.0, 0.0], [5.0, 0.0]])

    ranked = rank_value_candidates(
        _PreferTaskZeroMembers(), pursuer_embeddings, task_embeddings,
        pursuer_positions, task_positions, candidates, temperature=0.2,
    )

    assert tuple(ranked.task_counts.tolist()) == (3, 1)
    assert np.all(np.bincount(ranked.assignment, minlength=2) > 0)
    assert torch.allclose(ranked.capacity.sum(), torch.tensor(4.0))


def test_value_checkpoint_rejects_non_proxy_schema(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"schema": "wrong", "hidden_dim": 3, "value_head": {}}, path)

    with pytest.raises(ValueError, match="schema"):
        load_value_checkpoint(path, torch.device("cpu"), expected_hidden_dim=3)


def test_demand_guided_alliance_turns_three_one_capacity_into_three_one_membership():
    costs = torch.tensor([
        [0.1, 2.0],
        [0.2, 1.5],
        [0.3, 1.0],
        [1.2, 0.1],
    ])

    assignment = form_demand_guided_alliance(
        costs, torch.tensor([2.9, 1.1]), count_weight=4.0,
    )

    assert tuple(np.bincount(assignment, minlength=2)) == (3, 1)
    assert assignment[3] == 1
