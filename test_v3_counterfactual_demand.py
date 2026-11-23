import numpy as np
import torch

from .v3_counterfactual_demand import (
    fit_counterfactual_demand, mode_targets_from_rollouts,
)
from .v3_sparse_demand import SparseDemandHead


def test_counterfactual_targets_prefer_actual_three_plus_one_winner():
    modes = [(2, 2), (3, 1), (1, 3)]
    rollouts = [
        {"captured": 0, "first_capture_time": None, "reward": -120.0},
        {"captured": 1, "first_capture_time": 3.4, "reward": -80.0},
        {"captured": 0, "first_capture_time": None, "reward": -100.0},
    ]

    target, scores = mode_targets_from_rollouts(
        modes, rollouts, horizon_seconds=10.0, temperature=0.25,
    )

    assert target.shape == scores.shape == (3,)
    assert int(target.argmax()) == 1
    assert target[1] > 0.9
    assert np.isclose(target.sum(), 1.0)


def test_counterfactual_targets_do_not_average_symmetric_winners_into_two_plus_two():
    modes = [(2, 2), (3, 1), (1, 3)]
    rollouts = [
        {"captured": 0, "first_capture_time": None, "reward": -100.0},
        {"captured": 1, "first_capture_time": 4.0, "reward": -80.0},
        {"captured": 1, "first_capture_time": 4.0, "reward": -80.0},
    ]

    target, _ = mode_targets_from_rollouts(
        modes, rollouts, horizon_seconds=10.0, temperature=0.25,
    )

    assert target[1] > 0.49
    assert target[2] > 0.49
    assert target[0] < 0.01


def test_candidate_recall_training_updates_mode_branch_but_freezes_continuous_demand():
    torch.manual_seed(9)
    head = SparseDemandHead(hidden_dim=2, width=8)
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    modes = ((2, 2), (3, 1), (1, 3), (4, 0), (0, 4))
    scene = {
        "task_embeddings": embeddings,
        "alive_mask": torch.tensor([True, True]),
        "task_geometry": torch.zeros(2, 6),
        "num_agents": 4,
        "capture_size": 3,
        "modes": modes,
        # Both asymmetric modes are acceptable candidates; no unique Top-1
        # supervision should be imposed on the demand simplex.
        "good_mode_mask": np.array([False, True, True, False, False]),
    }
    continuous_before = {
        key: value.detach().clone() for key, value in head.net.state_dict().items()
    }
    mode_before = {
        key: value.detach().clone() for key, value in head.mode_net.state_dict().items()
    }

    fit_counterfactual_demand(
        head, [scene], epochs=100, learning_rate=1e-2,
    )
    _, probabilities = head.score_modes(
        embeddings, scene["alive_mask"], 4, modes,
        task_geometry=scene["task_geometry"],
    )

    for key, value in continuous_before.items():
        torch.testing.assert_close(head.net.state_dict()[key], value)
    assert any(not torch.equal(head.mode_net.state_dict()[key], value)
               for key, value in mode_before.items())
    top_three = set(torch.topk(probabilities, 3).indices.tolist())
    assert {1, 2}.issubset(top_three)
