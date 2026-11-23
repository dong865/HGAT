import pytest
import torch

from .v3_sparse_demand import (
    SparseDemandHead,
    capacities_from_sparse_demand,
    masked_sparsemax,
    neighbor_capacity_modes,
)


def test_masked_sparsemax_is_sparse_and_normalized():
    logits = torch.tensor([3.0, 1.0, -4.0])
    result = masked_sparsemax(logits, torch.tensor([True, True, True]))

    assert torch.all(result >= 0)
    assert torch.allclose(result.sum(), torch.tensor(1.0))
    assert torch.count_nonzero(result == 0).item() >= 1


def test_masked_sparsemax_sets_masked_entries_exactly_zero():
    logits = torch.tensor([1.0, 8.0, 2.0])
    result = masked_sparsemax(logits, torch.tensor([True, False, True]))

    assert result[1].item() == 0.0
    assert torch.allclose(result[[0, 2]].sum(), torch.tensor(1.0))


def test_masked_sparsemax_one_live_entry_is_one():
    result = masked_sparsemax(
        torch.tensor([2.0, 9.0]), torch.tensor([True, False])
    )

    assert torch.equal(result, torch.tensor([1.0, 0.0]))


def test_masked_sparsemax_all_false_mask_returns_zeros():
    result = masked_sparsemax(torch.tensor([2.0, 9.0]), torch.tensor([False, False]))

    assert torch.equal(result, torch.zeros(2))


def test_masked_sparsemax_rejects_nonfinite_active_logits():
    with pytest.raises(ValueError, match="finite"):
        masked_sparsemax(torch.tensor([float("nan"), 0.0]), torch.tensor([True, False]))


def test_masked_sparsemax_has_finite_gradients():
    logits = torch.tensor([3.0, 1.0, -4.0], requires_grad=True)
    masked_sparsemax(logits, torch.ones(3, dtype=torch.bool)).square().sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_masked_sparsemax_is_translation_invariant_at_large_float32_logits():
    logits = torch.tensor([1e8, 1e8], dtype=torch.float32)
    result = masked_sparsemax(logits, torch.tensor([True, True]))

    assert torch.allclose(result, torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.sum(), torch.tensor(1.0))


def test_masked_sparsemax_one_live_entry_is_one_at_large_logit_scale():
    result = masked_sparsemax(
        torch.tensor([1e8, 1e8]), torch.tensor([True, False])
    )

    assert torch.equal(result, torch.tensor([1.0, 0.0]))


def test_masked_sparsemax_uses_uniform_fallback_for_degenerate_projection(monkeypatch):
    monkeypatch.setattr(torch, "relu", lambda value: torch.zeros_like(value))

    result = masked_sparsemax(
        torch.tensor([0.0, 2.0, -1.0]), torch.tensor([True, True, False])
    )

    assert torch.equal(result, torch.tensor([0.5, 0.5, 0.0]))


def test_sparse_demand_head_outputs_masked_sparse_simplex():
    torch.manual_seed(3)
    head = SparseDemandHead(hidden_dim=4, width=8, max_agents=6, max_tasks=3)
    embeddings = torch.randn(3, 4)
    geometry = torch.randn(3, 6)
    alive = torch.tensor([True, True, False])

    demand = head(embeddings, alive, num_agents=4, task_geometry=geometry)

    assert head.max_agents == 6
    assert head.max_tasks == 3
    assert torch.all(demand >= 0)
    assert torch.allclose(demand[:2].sum(), torch.tensor(1.0))
    assert demand[2].item() == 0.0


def test_sparse_demand_head_no_live_task_returns_zeros():
    head = SparseDemandHead(hidden_dim=2, width=4)

    demand = head(torch.randn(2, 2), torch.tensor([False, False]), num_agents=4)

    assert torch.equal(demand, torch.zeros(2))


def test_sparse_demand_head_can_return_logits_for_nonzero_supervision_gradient():
    head = SparseDemandHead(hidden_dim=2, width=4)
    for parameter in head.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    embeddings = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    alive = torch.tensor([True, True])

    demand, logits = head(
        embeddings, alive, num_agents=4, return_logits=True,
    )
    target = torch.tensor([0.75, 0.25])
    loss = -(target * torch.log_softmax(logits, dim=0)).sum()
    loss.backward()

    assert demand.shape == logits.shape == (2,)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in head.parameters())


def test_sparse_demand_head_scores_capacity_modes_as_distribution():
    torch.manual_seed(4)
    head = SparseDemandHead(hidden_dim=2, width=8)
    embeddings = torch.randn(2, 2)
    geometry = torch.randn(2, 6)
    modes = [(2, 2), (3, 1), (1, 3)]

    logits, probabilities = head.score_modes(
        embeddings, torch.tensor([True, True]), num_agents=4,
        capacity_modes=modes, task_geometry=geometry,
    )

    assert logits.shape == probabilities.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.all(probabilities > 0)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))


def test_mode_score_branch_can_prefer_three_plus_one_without_changing_simplex():
    head = SparseDemandHead(hidden_dim=2, width=8)
    embeddings = torch.zeros(2, 2)
    alive = torch.tensor([True, True])
    modes = [(2, 2), (3, 1), (1, 3)]
    optimizer = torch.optim.Adam(head.mode_net.parameters(), lr=0.05)
    demand_before = head(embeddings, alive, num_agents=4).detach().clone()

    for _ in range(80):
        logits, _ = head.score_modes(
            embeddings, alive, num_agents=4, capacity_modes=modes,
        )
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([1])
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, probabilities = head.score_modes(
        embeddings, alive, num_agents=4, capacity_modes=modes,
    )
    demand_after = head(embeddings, alive, num_agents=4)
    assert int(probabilities.argmax()) == 1
    assert probabilities[1] > 0.9
    assert torch.equal(demand_before, demand_after)


def test_capacities_equal_agent_count_times_sparse_demand():
    demand = torch.tensor([0.75, 0.25, 0.0])
    alive = torch.tensor([True, True, False])

    capacity = capacities_from_sparse_demand(demand, alive, num_agents=4)

    assert torch.allclose(capacity, torch.tensor([3.0, 1.0, 0.0]))
    assert torch.allclose(capacity.sum(), torch.tensor(4.0))


def test_capacities_are_zero_with_no_live_tasks():
    capacity = capacities_from_sparse_demand(
        torch.tensor([0.2, 0.8]), torch.tensor([False, False]), num_agents=4
    )

    assert torch.equal(capacity, torch.zeros(2))


def test_four_agents_two_live_tasks_have_all_five_labelled_modes():
    modes = neighbor_capacity_modes(
        torch.tensor([2.0, 2.0]), torch.tensor([True, True]),
        num_agents=4, limit=5,
    )

    assert set(modes) == {(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)}


def test_neighbor_capacity_modes_respect_limit():
    modes = neighbor_capacity_modes(
        torch.tensor([2.0, 2.0]), torch.tensor([True, True]),
        num_agents=4, limit=2,
    )

    assert modes == [(2, 2), (1, 3)]


def test_neighbor_capacity_modes_use_lexicographic_tie_breaking():
    modes = neighbor_capacity_modes(
        torch.tensor([2.0, 2.0]), torch.tensor([True, True]),
        num_agents=4, limit=3,
    )

    assert modes == [(2, 2), (1, 3), (3, 1)]


def test_six_agents_three_tasks_modes_are_unique_feasible_and_include_zero_neighbor():
    modes = neighbor_capacity_modes(
        torch.tensor([0.0, 3.0, 3.0]), torch.tensor([True, True, True]),
        num_agents=6, limit=6,
    )

    assert len(modes) <= 6
    assert len(modes) == len(set(modes))
    assert all(len(mode) == 3 and all(x >= 0 for x in mode) for mode in modes)
    assert all(sum(mode) == 6 for mode in modes)
    assert any(0 in mode for mode in modes)


def test_neighbor_modes_keep_dead_positions_zero_and_empty_mask_is_empty():
    modes = neighbor_capacity_modes(
        torch.tensor([0.0, 4.0, 2.0]), torch.tensor([False, True, True]),
        num_agents=6, limit=4,
    )
    empty = neighbor_capacity_modes(
        torch.zeros(2), torch.tensor([False, False]), num_agents=6, limit=4
    )

    assert modes
    assert all(mode[0] == 0 for mode in modes)
    assert empty == []


def test_one_live_task_gets_all_agents():
    modes = neighbor_capacity_modes(
        torch.tensor([0.0, 5.0]), torch.tensor([False, True]),
        num_agents=5, limit=4,
    )

    assert modes == [(0, 5)]


def test_large_capacity_mode_search_is_bounded_and_keeps_incumbent():
    incumbent = (8, 3, 3, 3, 3)
    modes = neighbor_capacity_modes(
        torch.tensor([4.0] * 5), torch.tensor([True] * 5),
        num_agents=20, limit=6, current_mode=incumbent,
    )

    assert len(modes) == 6
    assert modes[0] == (4, 4, 4, 4, 4)
    assert incumbent in modes
    assert all(sum(mode) == 20 for mode in modes)
