import torch

from .v2_training import actor_los_auxiliary_loss, actor_ring_auxiliary_targets


def _obs(rows=2, dim=128, max_pursuers=6):
    obs = torch.zeros((1, rows, dim), dtype=torch.float32)
    target_offset = 10 + 3 * (max_pursuers - 1)
    obs[0, 0, 0:2] = torch.tensor([0.25, 0.50])
    obs[0, 0, target_offset:target_offset + 2] = torch.tensor([0.75, 0.50])
    obs[0, 0, target_offset + 3] = 1.0
    return obs


def test_los_loss_is_zero_for_correct_direction():
    obs = _obs()
    actions = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    mask = torch.ones((1, 2))
    loss = actor_los_auxiliary_loss(obs, actions, mask, max_pursuers=6)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_los_loss_penalizes_wrong_direction_and_ignores_unassigned_rows():
    obs = _obs()
    mask = torch.ones((1, 2))
    wrong = torch.tensor([[[-1.0, 0.0], [100.0, 100.0]]])
    right = torch.tensor([[[1.0, 0.0], [-100.0, -100.0]]])
    wrong_loss = actor_los_auxiliary_loss(obs, wrong, mask, max_pursuers=6)
    right_loss = actor_los_auxiliary_loss(obs, right, mask, max_pursuers=6)
    assert wrong_loss > 0.5
    assert torch.isclose(right_loss, torch.tensor(0.0))


def test_los_loss_empty_selection_preserves_gradient_graph():
    obs = torch.zeros((1, 2, 128), dtype=torch.float32)
    actions = torch.zeros((1, 2, 2), requires_grad=True)
    loss = actor_los_auxiliary_loss(obs, actions, torch.ones((1, 2)), 6)
    loss.backward()
    assert actions.grad is not None
    assert torch.count_nonzero(actions.grad) == 0


def test_ring_targets_give_three_members_distinct_directions():
    positions = torch.tensor([
        [1.0, 1.0, 0.0], [1.0, 4.0, 0.0], [4.0, 1.0, 0.0],
        [6.0, 6.0, 0.0], [6.0, 4.0, 0.0], [4.0, 6.0, 0.0],
        [3.0, 3.0, 0.0], [5.0, 5.0, 0.0],
    ]).numpy()
    snapshot = {
        "positions": positions,
        "velocities": positions * 0.0,
        "assignment": [0, 0, 0, 1, 1, 1],
        "task_mask": [True, True],
        "pursuer_mask": [True] * 6,
        "num_pursuers": 6,
        "num_evaders": 2,
        "length": 8.0,
    }
    target, valid = actor_ring_auxiliary_targets(
        [snapshot], max_pursuers=6, d_cap=2.0, device=torch.device("cpu"))
    assert target.shape == (1, 6, 2)
    assert valid.all()
    # A collapsed common command cannot satisfy distinct ring slots.
    assert torch.unique(target[0].round(decimals=3), dim=0).shape[0] >= 4
    assert torch.allclose(torch.linalg.vector_norm(target, dim=-1), torch.ones((1, 6)))
