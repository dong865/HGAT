import numpy as np
import pytest

from .v3_candidate_generator import generate_bounded_candidates


def _counts(row, num_tasks):
    return tuple(np.bincount(row, minlength=num_tasks).tolist())


def test_4v2_covers_each_labelled_mode_and_respects_limit():
    costs = np.array([
        [0.0, 5.0],
        [1.0, 4.0],
        [4.0, 1.0],
        [5.0, 0.0],
    ])
    modes = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]

    candidates = generate_bounded_candidates(costs, modes, [True, True], max_candidates=32)

    assert candidates.ndim == 2
    assert 1 <= len(candidates) <= 32
    assert candidates.shape[1] == 4
    assert np.issubdtype(candidates.dtype, np.integer)
    assert np.all((0 <= candidates) & (candidates < 2))
    assert len({tuple(row) for row in candidates}) == len(candidates)
    assert {_counts(row, 2) for row in candidates} == set(modes)


def test_dead_task_never_appears_in_6v3_candidates():
    costs = np.arange(18, dtype=float).reshape(6, 3)
    modes = [(3, 3, 0), (4, 2, 0), (2, 4, 0)]

    candidates = generate_bounded_candidates(
        costs, modes, [True, True, False], max_candidates=3
    )

    assert candidates.shape == (3, 6)
    assert np.all(candidates != 2)
    assert {_counts(row, 3) for row in candidates} == set(modes)


def test_matching_current_assignment_is_included_and_swaps_preserve_counts():
    costs = np.array([[0, 9], [1, 8], [8, 1], [9, 0]], dtype=float)
    current = np.array([0, 1, 0, 1])
    modes = [(2, 2)]

    candidates = generate_bounded_candidates(
        costs, modes, [True, True], current_assignment=current, max_candidates=20
    )

    rows = {tuple(row) for row in candidates}
    assert tuple(current) in rows
    assert all(_counts(row, 2) == modes[0] for row in candidates)
    assert len(rows) > 1


def test_matching_current_assignment_is_preserved_at_minimum_limit():
    costs = np.array([[0, 9], [1, 8], [8, 1], [9, 0]], dtype=float)
    current = np.array([0, 1, 0, 1])

    candidates = generate_bounded_candidates(
        costs, [(2, 2)], [True, True], current_assignment=current, max_candidates=1
    )

    assert candidates.tolist() == [current.tolist()]


@pytest.mark.parametrize(
    "costs,modes,alive,limit",
    [
        (np.zeros((2, 2)), [(1, 0)], [True, True], 1),  # mode total differs from N
        (np.zeros((2, 2)), [(-1, 3)], [True, True], 1),  # negative capacity
        (np.zeros((2, 2)), [(1.5, 0.5)], [True, True], 1),  # noninteger capacity
        (np.zeros((2, 2)), [(1, 1)], [True, False], 1),  # dead task capacity
        (np.zeros((2, 2)), [(2, 0), (0, 2)], [True, True], 1),  # too few slots
        (np.zeros((2, 2)), [(2, 0)], [False, False], 1),  # no live task
        (np.zeros((2, 2)), [(2, 0)], [True, True], 0),  # invalid limit
        (np.zeros((2, 2)), [(2, 0)], [True], 1),  # bad mask shape
    ],
)
def test_rejects_invalid_inputs(costs, modes, alive, limit):
    with pytest.raises(ValueError):
        generate_bounded_candidates(costs, modes, alive, max_candidates=limit)


def test_rejects_wrong_cost_rank_and_current_assignment_shape():
    with pytest.raises(ValueError):
        generate_bounded_candidates(np.zeros(4), [(2, 0)], [True, True])
    with pytest.raises(ValueError):
        generate_bounded_candidates(
            np.zeros((2, 2)), [(2, 0)], [True, True], current_assignment=[0]
        )


def test_candidate_set_is_equivariant_to_agent_permutation_with_ties():
    costs = np.array([
        [1.0, 1.0, 3.0],
        [1.0, 1.0, 3.0],
        [2.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
    ])
    modes = [(2, 2, 0), (2, 1, 1), (1, 2, 1), (1, 1, 2)]
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)

    original = generate_bounded_candidates(costs, modes, [True] * 3, max_candidates=200)
    permuted = generate_bounded_candidates(
        costs[permutation], modes, [True] * 3, max_candidates=200
    )[:, inverse]

    assert {tuple(row) for row in original} == {tuple(row) for row in permuted}


def test_candidate_set_is_equivariant_with_extreme_costs():
    costs = np.array([
        [-1e16, -1e16],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1e16],
    ])
    mode = [(2, 2)]
    permutation = np.array([1, 2, 3, 0])
    inverse = np.argsort(permutation)

    original = generate_bounded_candidates(costs, mode, [True, True], max_candidates=200)
    permuted = generate_bounded_candidates(
        costs[permutation], mode, [True, True], max_candidates=200
    )[:, inverse]

    assert {tuple(row) for row in original} == {tuple(row) for row in permuted}


def test_large_mode_generation_returns_fixed_budget_without_enumeration():
    costs = np.arange(60, dtype=float).reshape(12, 5)
    candidates = generate_bounded_candidates(
        costs, [(3, 3, 2, 2, 2)], [True] * 5, max_candidates=12,
    )

    assert 1 <= len(candidates) <= 12
    assert all(_counts(row, 5) == (3, 3, 2, 2, 2) for row in candidates)
