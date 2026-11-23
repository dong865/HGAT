"""Pure V3 helpers for value-ranked coalition candidates and task demand.

This module consumes the independently trained geometric-proxy value head but
does not alter it or install it into the online controller.  Demand is an
identifiable share of the resources left after reserving one member per active
task, rather than an unconstrained absolute head followed by Sinkhorn scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn

from .coalition_value import (
    VALUE_SCHEMA,
    CoalitionValueHead,
    build_candidate_features,
)


@dataclass(frozen=True)
class ValueRankedAllocation:
    assignment: np.ndarray
    task_counts: np.ndarray
    predicted_value: float
    candidate_values: torch.Tensor
    candidate_weights: torch.Tensor
    demand: torch.Tensor
    capacity: torch.Tensor
    best_index: int


def form_demand_guided_alliance(
    relation_costs: torch.Tensor,
    capacity: torch.Tensor,
    count_weight: float = 2.0,
) -> np.ndarray:
    """Sequentially form non-empty alliances from demand and relation costs.

    One distinct member is first seeded into every task.  Remaining members
    are added by the smallest relation-plus-marginal-capacity cost, avoiding
    enumeration of complete assignments.
    """
    costs = relation_costs.detach().cpu().numpy().astype(np.float64)
    target = capacity.detach().cpu().numpy().astype(np.float64)
    if costs.ndim != 2 or target.shape != (costs.shape[1],):
        raise ValueError("relation_costs [N,M] and capacity [M] are required")
    num_agents, num_tasks = costs.shape
    if num_agents < num_tasks:
        raise ValueError("every active task requires one member")
    best_seed = None
    best_seed_cost = float("inf")
    for agents in itertools.permutations(range(num_agents), num_tasks):
        value = sum(costs[agent, task] for task, agent in enumerate(agents))
        if value < best_seed_cost:
            best_seed_cost = float(value)
            best_seed = agents
    assignment = np.full(num_agents, -1, dtype=np.int64)
    counts = np.zeros(num_tasks, dtype=np.int64)
    for task, agent in enumerate(best_seed):
        assignment[agent] = task
        counts[task] += 1
    unassigned = set(np.flatnonzero(assignment < 0).tolist())
    while unassigned:
        best = None
        best_cost = float("inf")
        for agent in sorted(unassigned):
            for task in range(num_tasks):
                marginal = (
                    (counts[task] + 1.0 - target[task]) ** 2
                    - (counts[task] - target[task]) ** 2
                )
                value = costs[agent, task] + float(count_weight) * marginal
                if value < best_cost:
                    best_cost = float(value)
                    best = agent, task
        agent, task = best
        assignment[agent] = task
        counts[task] += 1
        unassigned.remove(agent)
    return assignment


def _candidate_counts(candidates: np.ndarray, num_tasks: int) -> np.ndarray:
    array = np.asarray(candidates, dtype=np.int64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("candidates must be a non-empty [K,N] array")
    counts = np.stack([
        np.bincount(row, minlength=num_tasks)[:num_tasks] for row in array
    ])
    if np.any(counts <= 0):
        raise ValueError("every candidate must keep every active task non-empty")
    return counts


def normalized_demand_from_candidates(
    values: torch.Tensor,
    candidates: np.ndarray,
    num_tasks: int,
    temperature: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(rho, capacity)` from a soft distribution over candidates.

    One member is reserved for every active task.  Candidate counts above that
    reserve define a distribution over the remaining `N-M` resources.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scores = values.reshape(-1)
    counts_np = _candidate_counts(candidates, int(num_tasks))
    if len(scores) != len(counts_np):
        raise ValueError("one value is required for every candidate")
    num_agents = int(np.asarray(candidates).shape[1])
    residual = num_agents - int(num_tasks)
    if residual < 0:
        raise ValueError("num_agents must be at least num_tasks")
    weights = torch.softmax(scores / float(temperature), dim=0)
    if residual == 0:
        demand = torch.full(
            (num_tasks,), 1.0 / num_tasks, dtype=scores.dtype,
            device=scores.device,
        )
        capacity = torch.ones_like(demand)
        return demand, capacity
    counts = torch.as_tensor(counts_np, dtype=scores.dtype, device=scores.device)
    excess_share = (counts - 1.0) / float(residual)
    demand = (weights[:, None] * excess_share).sum(dim=0)
    demand = demand / demand.sum().clamp_min(torch.finfo(demand.dtype).eps)
    capacity = 1.0 + float(residual) * demand
    return demand, capacity


@torch.no_grad()
def rank_value_candidates(
    model: nn.Module,
    pursuer_embeddings: torch.Tensor,
    task_embeddings: torch.Tensor,
    pursuer_positions: np.ndarray,
    task_positions: np.ndarray,
    candidates: np.ndarray,
    temperature: float = 0.25,
) -> ValueRankedAllocation:
    """Score all supplied candidates and return their Top-1 plus demand."""
    candidates = np.asarray(candidates, dtype=np.int64)
    _candidate_counts(candidates, int(task_embeddings.shape[0]))
    features = torch.stack([
        build_candidate_features(
            pursuer_embeddings, task_embeddings, pursuer_positions,
            task_positions, assignment,
        )
        for assignment in candidates
    ])
    values = model(features).reshape(-1)
    if len(values) != len(candidates):
        raise ValueError("value model must return one scalar per candidate")
    demand, capacity = normalized_demand_from_candidates(
        values, candidates, int(task_embeddings.shape[0]), temperature,
    )
    weights = torch.softmax(values / float(temperature), dim=0)
    best_index = int(values.argmax().item())
    assignment = candidates[best_index].copy()
    counts = np.bincount(
        assignment, minlength=int(task_embeddings.shape[0])
    )[:int(task_embeddings.shape[0])]
    return ValueRankedAllocation(
        assignment=assignment,
        task_counts=counts,
        predicted_value=float(values[best_index]),
        candidate_values=values.detach().cpu(),
        candidate_weights=weights.detach().cpu(),
        demand=demand.detach().cpu(),
        capacity=capacity.detach().cpu(),
        best_index=best_index,
    )


def load_value_checkpoint(
    path: str | Path,
    device: torch.device,
    expected_hidden_dim: int | None = None,
) -> Tuple[CoalitionValueHead, Mapping]:
    """Load a geometric-proxy value checkpoint with strict schema checks."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("schema") != VALUE_SCHEMA:
        raise ValueError(
            f"unexpected value checkpoint schema {payload.get('schema')!r}; "
            f"expected {VALUE_SCHEMA!r}"
        )
    if payload.get("supervision") != "geometric_proxy_not_rollout_return":
        raise ValueError("value checkpoint supervision metadata is missing or unsafe")
    hidden_dim = int(payload.get("hidden_dim", -1))
    if expected_hidden_dim is not None and hidden_dim != int(expected_hidden_dim):
        raise ValueError(
            f"hidden_dim mismatch: value={hidden_dim}, policy={expected_hidden_dim}"
        )
    model = CoalitionValueHead(hidden_dim).to(device)
    model.load_state_dict(payload["value_head"])
    model.eval()
    return model, payload
