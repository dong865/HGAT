"""Bounded, count-preserving task assignments for sparse allocation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _as_numpy(value: torch.Tensor | np.ndarray, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _assignment_cost(row: tuple[int, ...], costs: np.ndarray) -> float:
    # `math.fsum` avoids losing small terms based on the arbitrary agent order.
    return math.fsum(costs[agent_id, task_id]
                     for agent_id, task_id in enumerate(row))


def generate_bounded_candidates(
    relation_costs: torch.Tensor | np.ndarray,
    capacity_modes: Sequence[Sequence[int]],
    alive_mask: torch.Tensor | np.ndarray,
    current_assignment: np.ndarray | None = None,
    max_candidates: int = 32,
) -> np.ndarray:
    """Generate a bounded set of live-task assignments for capacity modes.

    Each mode contributes its minimum-cost assignment(s); one-swap neighbors
    add alternatives without changing that mode's task counts.
    """
    costs = _as_numpy(relation_costs, dtype=np.float64)
    alive = _as_numpy(alive_mask, dtype=bool)
    if costs.ndim != 2:
        raise ValueError("relation_costs must have shape [num_agents, num_tasks]")
    num_agents, num_tasks = costs.shape
    if num_agents < 1 or num_tasks < 1:
        raise ValueError("relation_costs must include at least one agent and task")
    if alive.shape != (num_tasks,):
        raise ValueError("alive_mask must have shape [num_tasks]")
    if not alive.any():
        raise ValueError("at least one task must be alive")
    if not np.isfinite(costs[:, alive]).all():
        raise ValueError("relation costs for live tasks must be finite")

    modes = []
    for mode in capacity_modes:
        values = tuple(mode)
        if any(not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError("capacity counts must be integers")
        modes.append(tuple(int(value) for value in values))
    if not modes:
        raise ValueError("capacity_modes must contain at least one mode")
    if any(len(mode) != num_tasks for mode in modes):
        raise ValueError("every capacity mode must have one count per task")
    for mode in modes:
        if any(count < 0 for count in mode):
            raise ValueError("capacity counts must be nonnegative")
        if sum(mode) != num_agents:
            raise ValueError("each capacity mode must sum to num_agents")
        if any(mode[task_id] != 0 for task_id in range(num_tasks) if not alive[task_id]):
            raise ValueError("dead tasks must have zero capacity")

    try:
        limit = int(max_candidates)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_candidates must be an integer") from exc
    if limit != max_candidates or limit < len(modes):
        raise ValueError("max_candidates must be at least the number of modes")

    current = None
    if current_assignment is not None:
        current_array = _as_numpy(current_assignment)
        if current_array.shape != (num_agents,):
            raise ValueError("current_assignment must have shape [num_agents]")
        if not np.issubdtype(current_array.dtype, np.integer):
            raise ValueError("current_assignment must contain integer task IDs")
        current_tuple = tuple(int(value) for value in current_array)
        if all(0 <= value < num_tasks and alive[value] for value in current_tuple):
            current = current_tuple

    per_mode: list[list[tuple[int, ...]]] = []
    all_candidates: set[tuple[int, ...]] = set()
    for mode in modes:
        slots = np.repeat(np.arange(num_tasks, dtype=np.int64), mode)
        row_indices, slot_indices = linear_sum_assignment(costs[:, slots])
        assignment = np.empty(num_agents, dtype=np.int64)
        assignment[row_indices] = slots[slot_indices]
        best_row = tuple(int(value) for value in assignment)
        mode_candidates = {best_row}
        frontier = {best_row}
        # Explore only a fixed-size swap neighborhood. With a generous test
        # budget this recovers tie orbits; deployment remains capped at 32.
        while frontier and len(mode_candidates) < limit:
            next_frontier = set()
            for row in frontier:
                for left in range(num_agents):
                    for right in range(left + 1, num_agents):
                        if row[left] == row[right]:
                            continue
                        swapped = list(row)
                        swapped[left], swapped[right] = swapped[right], swapped[left]
                        swapped = tuple(swapped)
                        if swapped not in mode_candidates:
                            mode_candidates.add(swapped)
                            next_frontier.add(swapped)
            frontier = next_frontier

        if current is not None and tuple(np.bincount(current, minlength=num_tasks)) == mode:
            mode_candidates.add(current)

        # Retain one minimum-cost representative per supplied mode before
        # filling any remaining capacity with globally ranked alternatives.
        representative = best_row
        if current is not None and tuple(np.bincount(current, minlength=num_tasks)) == mode:
            representative = current
        per_mode.append([representative])
        all_candidates.update(mode_candidates)

    reserved = []
    for rows in per_mode:
        row = rows[0]
        if row not in reserved:
            reserved.append(row)
    remaining = all_candidates.difference(reserved)
    ranked = sorted(remaining, key=lambda row: (_assignment_cost(row, costs), row))
    selected = reserved + ranked[:max(0, limit - len(reserved))]
    return np.asarray(selected, dtype=np.int64)
