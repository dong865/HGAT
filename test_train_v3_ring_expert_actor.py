import copy

import torch

from .stage_networks import SharedActor
from .train_v3_ring_expert_actor import (
    anchored_actor_loss,
    flatten_trajectories,
    inject_actor_weights,
    select_expert_trajectories,
)


def test_anchored_loss_weights_farthest_member_and_parameter_anchor():
    actor = SharedActor(5, hidden_dim=8)
    original = copy.deepcopy(actor)
    obs = torch.randn(2, 4, 5)
    expert = torch.zeros(2, 4, 2)
    anchor = torch.ones(2, 4, 2) * 0.25
    farthest = torch.tensor([1, 3])
    total, parts = anchored_actor_loss(
        actor, original, obs, expert, anchor, farthest,
        farthest_weight=4.0, anchor_weight=0.2, param_weight=0.1,
    )
    assert torch.isfinite(total)
    assert parts["expert"] > 0
    assert parts["anchor"] > 0
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())


def test_inject_actor_weights_changes_only_actor_and_actor_target():
    actor = SharedActor(5, hidden_dim=8)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(1.0)
    payload = {
        "actor": {"old": torch.tensor(1)},
        "actor_target": {"old": torch.tensor(2)},
        "critic1": {"keep": torch.tensor(3)},
        "demand_head": {"keep": torch.tensor(4)},
    }
    result = inject_actor_weights(payload, actor)
    assert result["critic1"] == payload["critic1"]
    assert result["demand_head"] == payload["demand_head"]
    assert set(result["actor"]) == set(actor.state_dict())
    assert set(result["actor_target"]) == set(actor.state_dict())


def test_select_expert_trajectories_excludes_partial_capture_failures_by_default():
    trajectories = [
        {"seed": 72000, "captured": 1, "full_capture": False, "records": [1]},
        {"seed": 72001, "captured": 2, "full_capture": True, "records": [2]},
        {"seed": 72002, "captured": 2, "full_capture": True, "records": [3]},
    ]

    selected = select_expert_trajectories(trajectories)

    assert [row["seed"] for row in selected] == [72001, 72002]


def test_select_expert_trajectories_can_explicitly_keep_partial_capture_ablation():
    trajectories = [
        {"seed": 72000, "captured": 1, "full_capture": False, "records": [1]},
        {"seed": 72001, "captured": 2, "full_capture": True, "records": [2]},
    ]

    selected = select_expert_trajectories(trajectories, require_full_capture=False)

    assert [row["seed"] for row in selected] == [72000, 72001]


def test_flatten_post_capture_only_starts_at_dead_task_repair():
    def record(assignment, value):
        return {
            "assignment": assignment,
            "obs": [[value] * 5] * 4,
            "expert_action": [[0.0, 0.0]] * 4,
            "original_action": [[0.0, 0.0]] * 4,
            "farthest_member": 0,
        }

    trajectory = {
        "seed": 72001,
        "initial_assignment": [0, 0, 0, 1],
        "records": [
            record([0, 0, 0, 1], 1.0),
            record([0, 0, 0, 1], 2.0),
            record([1, 1, 1, 1], 3.0),
            record([1, 1, 1, 1], 4.0),
        ],
    }

    rows = flatten_trajectories([trajectory], post_capture_only=True)

    assert [float(row[1][0, 0]) for row in rows] == [3.0, 4.0]
