"""Behavior tests for the sparse-demand/value hybrid allocator."""

import numpy as np
import pytest
import torch
from torch import nn

from .v3_hybrid_controller import (
    StableAllianceGate,
    V3HybridAllocator,
    capture_feasibility_penalty,
    score_candidates,
)
from . import v3_hybrid_controller
from .v3_sparse_demand import SparseDemandHead


def test_demand_penalty_prefers_two_plus_two_when_values_tie():
    candidates = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    scores = score_candidates(
        torch.tensor([7.0, 7.0]), candidates, torch.tensor([2.0, 2.0]),
        None, demand_weight=1.0, switch_weight=0.0,
    )
    assert torch.allclose(scores, torch.tensor([7.0, 5.0]))


def test_larger_value_overrides_soft_demand_penalty():
    candidates = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    scores = score_candidates(
        torch.tensor([7.0, 10.0]), candidates, torch.tensor([2.0, 2.0]),
        None, demand_weight=1.0, switch_weight=0.0,
    )
    assert scores.argmax().item() == 1
    assert torch.allclose(scores, torch.tensor([7.0, 8.0]))


def test_capture_feasibility_penalty_only_suppresses_two_plus_two_when_urgent():
    candidates = np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ])
    early = capture_feasibility_penalty(
        candidates, alive=np.array([True, True]), capture_size=3,
        weight=0.8, remaining_fraction=0.9, warmup_fraction=0.5,
        urgency=False,
    )
    urgent = capture_feasibility_penalty(
        candidates, alive=np.array([True, True]), capture_size=3,
        weight=0.8, remaining_fraction=0.4, warmup_fraction=0.5,
        urgency=True,
    )
    assert torch.equal(early, torch.zeros(4))
    assert torch.allclose(urgent, torch.tensor([0.8, 0.0, 0.0, 0.0]))


def test_capture_feasibility_penalty_is_zero_after_one_task_dies():
    candidates = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    penalty = capture_feasibility_penalty(
        candidates, alive=np.array([True, False]), capture_size=3,
        weight=0.8, remaining_fraction=0.1, warmup_fraction=0.5,
        urgency=True,
    )
    assert torch.equal(penalty, torch.zeros(2))


def test_switch_cost_is_changed_agent_fraction_only():
    candidates = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    scores = score_candidates(
        torch.tensor([4.0, 4.0]), candidates, torch.tensor([2.0, 2.0]),
        np.array([0, 0, 1, 1]), demand_weight=0.0, switch_weight=2.0,
    )
    assert torch.allclose(scores, torch.tensor([4.0, 3.0]))


@pytest.mark.parametrize("values,candidates,capacity,previous", [
    (torch.tensor([1.0]), np.array([[0, 1], [0, 0]]), torch.tensor([1., 1.]), None),
    (torch.tensor([1.0]), np.array([[0, 2]]), torch.tensor([1., 1.]), None),
    (torch.tensor([float('nan')]), np.array([[0, 1]]), torch.tensor([1., 1.]), None),
    (torch.tensor([1.0]), np.array([[0, 1]]), torch.tensor([float('inf'), 1.]), None),
    (torch.tensor([1.0]), np.array([[0, 1]]), torch.tensor([1., 1.]), np.array([0])),
])
def test_score_rejects_invalid_shapes_and_nonfinite_inputs(values, candidates, capacity, previous):
    with pytest.raises(ValueError):
        score_candidates(values, candidates, capacity, previous, 1.0, 1.0)


def test_gate_holds_better_candidate_inside_dwell_then_accepts_sufficient_gain():
    gate = StableAllianceGate(min_dwell_steps=3, improvement_margin=0.5)
    candidates = np.array([[0, 0, 1, 1], [0, 0, 0, 1]])
    assignment, info = gate.select(candidates, torch.tensor([2.0, 1.0]), step=5)
    assert assignment.tolist() == [0, 0, 1, 1]
    assert info == {"switched": True, "reason": "initial"}

    assignment, info = gate.select(candidates, torch.tensor([2.0, 4.0]), step=7)
    assert assignment.tolist() == [0, 0, 1, 1]
    assert info == {"switched": False, "reason": "dwell"}

    assignment, info = gate.select(candidates, torch.tensor([2.0, 2.4]), step=8)
    assert assignment.tolist() == [0, 0, 1, 1]
    assert info == {"switched": False, "reason": "margin"}

    assignment, info = gate.select(candidates, torch.tensor([2.0, 2.6]), step=9)
    assert assignment.tolist() == [0, 0, 0, 1]
    assert info == {"switched": True, "reason": "improved"}


def test_gate_force_uses_best_score_immediately_and_reset_clears_history():
    gate = StableAllianceGate(min_dwell_steps=100, improvement_margin=10.0)
    candidates = np.array([[0, 0], [1, 1]])
    gate.select(candidates, torch.tensor([2.0, 1.0]), step=0)
    assignment, info = gate.select(candidates, torch.tensor([0.0, 3.0]), step=1, force=True)
    assert assignment.tolist() == [1, 1]
    assert info == {"switched": True, "reason": "forced"}
    gate.reset()
    assignment, info = gate.select(candidates, torch.tensor([1.0, 0.0]), step=2)
    assert assignment.tolist() == [0, 0]
    assert info == {"switched": True, "reason": "initial"}


def test_gate_keeps_members_of_capture_critical_task_until_freeze_releases():
    """Removing the frozen-task filter must allow the high-score bad switch."""
    gate = StableAllianceGate(min_dwell_steps=0, improvement_margin=0.0)
    candidates = np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ])
    gate.select(candidates, torch.tensor([2.0, 1.0]), step=0)

    frozen, info = gate.select(
        candidates, torch.tensor([1.0, 10.0]), step=1,
        frozen_tasks=np.array([True, False]),
    )
    assert frozen.tolist() == [0, 0, 0, 1]
    assert info == {"switched": False, "reason": "capture_freeze"}

    released, info = gate.select(
        candidates, torch.tensor([1.0, 10.0]), step=2,
        frozen_tasks=np.array([False, False]),
    )
    assert released.tolist() == [1, 1, 1, 1]
    assert info == {"switched": True, "reason": "improved"}


def test_capture_freeze_uses_hold_or_distance_and_releases_with_hysteresis():
    """Dropping either trigger or hysteresis must change this literal sequence."""
    assert hasattr(v3_hybrid_controller, "CaptureFreezeGuard")
    guard = v3_hybrid_controller.CaptureFreezeGuard(
        freeze_distance=1.6, release_distance=1.84, capture_size=3,
    )
    assignment = np.array([0, 0, 0, 1])
    pursuers = np.array([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [8.0, 8.0],
    ])
    tasks = np.array([[0.0, 0.0], [9.0, 9.0]])
    alive = np.array([True, True])

    assert guard.update(assignment, pursuers, tasks, alive, [0, 0]).tolist() == [True, False]

    # Between freeze and release radii, the task remains frozen.
    pursuers[:3] *= 1.7
    assert guard.update(assignment, pursuers, tasks, alive, [0, 0]).tolist() == [True, False]

    pursuers[:3] *= 2.0 / 1.7
    assert guard.update(assignment, pursuers, tasks, alive, [0, 0]).tolist() == [False, False]

    # A live capture hold freezes even when geometry is outside the distance trigger.
    assert guard.update(assignment, pursuers, tasks, alive, [1, 0]).tolist() == [True, False]
    assert guard.update(assignment, pursuers, tasks, [False, True], [1, 0]).tolist() == [False, False]


class _CountFirstTaskValue(nn.Module):
    def forward(self, features, task_mask=None):
        # The first scale feature is the number of agents assigned to task 0.
        return features[:, 0, -6] * 10.0


class _ScheduledDemandHead(nn.Module):
    def __init__(self, demands):
        super().__init__()
        self.demands = [torch.tensor(demand, dtype=torch.float32) for demand in demands]
        self.calls = 0

    def forward(self, task_embeddings, alive_mask, num_agents, capture_size=3,
                task_geometry=None):
        demand = self.demands[min(self.calls, len(self.demands) - 1)]
        self.calls += 1
        return demand.to(task_embeddings.device)


class _MutableCountFirstTaskValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0

    def forward(self, features, task_mask=None):
        return features[:, 0, -6] * (10.0 * self.scale)


def _allocator_fixture(num_tasks=2):
    demand_head = SparseDemandHead(hidden_dim=2, width=4)
    for parameter in demand_head.parameters():
        nn.init.zeros_(parameter)
    allocator = V3HybridAllocator(
        demand_head=demand_head, value_head=_CountFirstTaskValue(), map_size=10.0,
        mode_limit=15, max_candidates=32, demand_weight=0.1,
        switch_weight=0.0, min_dwell_steps=5, improvement_margin=0.1,
    )
    encoded = {
        "pursuer_embeddings": torch.zeros(4, 2),
        "task_embeddings": torch.zeros(num_tasks, 2),
        "relation_costs": torch.zeros(4, num_tasks),
    }
    snapshot = {"positions": np.array(
        [[0., 0.], [1., 0.], [0., 1.], [1., 1.]]
        + [[5. + i, 5.] for i in range(num_tasks)]
    )}
    return allocator, encoded, snapshot


def test_allocator_can_select_four_plus_zero_from_bounded_candidates():
    allocator, encoded, snapshot = _allocator_fixture()
    result = allocator.allocate(encoded, snapshot, torch.tensor([True, True]), step=0)
    assert result["assignment"].tolist() == [0, 0, 0, 0]
    assert len(result["assignment"]) == 4
    assert (4, 0) in result["modes"]
    assert result["candidate_count"] <= 32
    assert result["raw_values"].shape == result["unified_scores"].shape == (result["candidate_count"],)
    assert result["switched"] is True
    assert result["reason"] == "initial"
    assert torch.allclose(result["demand"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(result["capacity"], torch.tensor([2.0, 2.0]))


def test_allocator_reports_mode_probabilities_aligned_with_modes():
    allocator, encoded, snapshot = _allocator_fixture()
    result = allocator.allocate(
        encoded, snapshot, torch.tensor([True, True]), step=0,
    )

    assert result["mode_probabilities"].shape == (len(result["modes"]),)
    assert torch.allclose(result["mode_probabilities"].sum(), torch.tensor(1.0))
    assert result["candidate_mode_probabilities"].shape == (
        result["candidate_count"],
    )


def test_allocator_forces_reassignment_when_previous_task_dies():
    allocator, encoded, snapshot = _allocator_fixture(num_tasks=3)
    first = allocator.allocate(encoded, snapshot, torch.tensor([True, True, True]), step=0)
    assert first["assignment"].tolist() == [0, 0, 0, 0]
    second = allocator.allocate(encoded, snapshot, torch.tensor([False, True, True]), step=1)
    assert set(second["assignment"].tolist()).issubset({1, 2})
    assert second["capacity"][0].item() == 0.0
    assert second["demand"][0].item() == 0.0
    assert second["reason"] == "forced"


def test_allocator_assigns_all_agents_to_only_live_task():
    allocator, encoded, snapshot = _allocator_fixture(num_tasks=3)
    result = allocator.allocate(encoded, snapshot, torch.tensor([False, True, False]), step=0)
    assert result["assignment"].tolist() == [1, 1, 1, 1]
    assert result["modes"] == [(0, 4, 0)]
    assert result["candidate_count"] == 1


def test_allocator_uses_active_counts_from_padded_encoder_output():
    allocator, encoded, snapshot = _allocator_fixture()
    encoded["pursuer_embeddings"] = torch.zeros(6, 2)
    encoded["task_embeddings"] = torch.zeros(3, 2)
    encoded["relation_costs"] = torch.zeros(6, 3)
    encoded["num_pursuers"] = 4
    encoded["num_evaders"] = 2
    result = allocator.allocate(encoded, snapshot, torch.tensor([True, True, False]), step=0)
    assert result["assignment"].tolist() == [0, 0, 0, 0]
    assert len(result["demand"]) == len(result["capacity"]) == 2
    encoded["num_pursuers"] = 3
    smaller_snapshot = {"positions": np.array(
        [[0., 0.], [1., 0.], [0., 1.], [5., 5.], [6., 5.]]
    )}
    resized = allocator.allocate(
        encoded, smaller_snapshot, torch.tensor([True, True, False]), step=1
    )
    assert len(resized["assignment"]) == 3
    assert resized["reason"] == "forced"


def test_allocator_drops_stale_assignment_when_active_agent_count_shrinks():
    allocator, encoded, snapshot = _allocator_fixture()
    encoded["pursuer_embeddings"] = torch.zeros(6, 2)
    encoded["task_embeddings"] = torch.zeros(3, 2)
    encoded["relation_costs"] = torch.zeros(6, 3)
    encoded["num_pursuers"] = 4
    encoded["num_evaders"] = 2
    allocator.allocate(
        encoded, snapshot, torch.tensor([True, True, False]), step=0
    )

    encoded["num_pursuers"] = 3
    resized_snapshot = {"positions": np.array([
        [0., 0.], [1., 0.], [0., 1.], [5., 5.], [6., 5.],
    ])}
    result = allocator.allocate(
        encoded, resized_snapshot, torch.tensor([True, True, False]), step=1
    )

    assert result["assignment"].shape == (3,)
    assert set(result["assignment"].tolist()).issubset({0, 1})
    assert result["reason"] == "forced"


def test_gate_compares_changed_mode_candidates_with_current_incumbent_score():
    allocator, encoded, snapshot = _allocator_fixture()
    allocator.demand_head = _ScheduledDemandHead([[0.5, 0.5], [1.0, 0.0]])
    allocator.value_head = _MutableCountFirstTaskValue()
    allocator.gate = StableAllianceGate(min_dwell_steps=0, improvement_margin=0.5)
    allocator.mode_limit = 2
    allocator.max_candidates = 2
    allocator.demand_weight = 0.1

    first = allocator.allocate(
        encoded, snapshot, torch.tensor([True, True]), step=0
    )
    assert first["assignment"].tolist() == [0, 0, 1, 1]
    assert first["candidate_count"] == 2

    allocator.value_head.scale = 0.4
    second = allocator.allocate(
        encoded, snapshot, torch.tensor([True, True]), step=1
    )

    assert second["modes"] == [(4, 0), (2, 2)]
    assert second["candidate_count"] == 2
    assert second["assignment"].tolist() == [0, 0, 0, 0]
    assert second["reason"] == "improved"


def test_hybrid_module_has_no_sinkhorn_reference():
    import inspect
    from . import v3_hybrid_controller

    assert "sinkhorn" not in inspect.getsource(v3_hybrid_controller).lower()
