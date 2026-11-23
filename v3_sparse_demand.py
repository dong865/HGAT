"""Sparse relative demand prediction and integer capacity neighbors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .v3_demand_distillation import DEMAND_GEOMETRY_DIM


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor,
                     dim: int = -1) -> torch.Tensor:
    """Project active logits onto the simplex, leaving masked entries at zero."""
    active = mask.to(device=logits.device, dtype=torch.bool)
    if active.shape != logits.shape:
        active = torch.broadcast_to(active, logits.shape)
    if torch.any(active & ~torch.isfinite(logits)):
        raise ValueError("active logits must be finite")

    dim = dim % logits.ndim
    # Sparsemax is translation invariant. Centering before sorting avoids
    # cancellation in the threshold arithmetic for large common offsets.
    compute_dtype = (torch.float64 if logits.dtype in
                     (torch.float16, torch.bfloat16, torch.float32) else logits.dtype)
    work_logits = logits.to(dtype=compute_dtype)
    active_count = active.sum(dim=dim, keepdim=True)
    max_active = work_logits.masked_fill(~active, -torch.inf).max(
        dim=dim, keepdim=True
    ).values
    max_active = torch.where(active_count > 0, max_active, torch.zeros_like(max_active))
    safe_logits = torch.where(
        active, work_logits - max_active, torch.zeros_like(work_logits)
    )
    sorted_logits, _ = torch.sort(
        safe_logits.masked_fill(~active, -torch.inf), dim=dim, descending=True
    )
    ranks = torch.arange(
        1, logits.shape[dim] + 1, dtype=compute_dtype, device=logits.device
    )
    shape = [1] * logits.ndim
    shape[dim] = logits.shape[dim]
    ranks = ranks.reshape(shape)
    cumulative = torch.cumsum(sorted_logits, dim=dim)
    sorted_active = torch.arange(
        logits.shape[dim], device=logits.device
    ).reshape(shape) < active_count
    support_sorted = sorted_active & (1 + ranks * sorted_logits > cumulative)
    support_count = support_sorted.sum(dim=dim, keepdim=True)
    tau = ((sorted_logits.masked_fill(~support_sorted, 0.0).sum(dim=dim, keepdim=True)
            - 1.0) / support_count.clamp_min(1).to(compute_dtype))
    projected = torch.relu(safe_logits - tau)
    projected = torch.where(active, projected, torch.zeros_like(projected))

    # A defensive fallback for extreme floating-point inputs where projection
    # fails to produce a finite positive mass.
    mass = projected.sum(dim=dim, keepdim=True)
    live = active_count > 0
    tolerance = torch.finfo(compute_dtype).eps * 16
    degenerate = live & (
        ~torch.isfinite(mass) | (mass <= tolerance) | ((mass - 1.0).abs() > tolerance)
    )
    uniform = active.to(compute_dtype) / active_count.clamp_min(1).to(compute_dtype)
    result = torch.where(degenerate, uniform, projected)
    return result.to(dtype=logits.dtype)


class SparseDemandHead(nn.Module):
    """Predict a masked sparse demand simplex using normalized-demand cues."""

    def __init__(self, hidden_dim: int, width: int = 64,
                 max_agents: int = 6, max_tasks: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim) + 3 + DEMAND_GEOMETRY_DIM, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.max_agents = int(max_agents)
        self.max_tasks = int(max_tasks)
        mode_input_dim = self.hidden_dim + DEMAND_GEOMETRY_DIM + self.max_tasks + 3
        self.mode_net = nn.Sequential(
            nn.Linear(mode_input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1),
        )

    def forward(self, task_embeddings: torch.Tensor, alive_mask: torch.Tensor,
                num_agents: int, capture_size: int = 3,
                task_geometry: torch.Tensor | None = None,
                return_logits: bool = False):
        mask = alive_mask.to(device=task_embeddings.device, dtype=torch.bool)
        num_tasks = int(mask.sum().item())
        if num_tasks == 0:
            zeros = task_embeddings.new_zeros(task_embeddings.shape[0])
            return (zeros, zeros) if return_logits else zeros
        scarcity = float(num_agents) / max(num_tasks * int(capture_size), 1)
        global_features = task_embeddings.new_tensor([
            float(num_agents) / max(self.max_agents, 1),
            float(num_tasks) / max(self.max_tasks, 1),
            scarcity,
        ]).expand(task_embeddings.shape[0], -1)
        if task_geometry is None:
            task_geometry = task_embeddings.new_zeros(
                (task_embeddings.shape[0], DEMAND_GEOMETRY_DIM)
            )
        else:
            task_geometry = task_geometry.to(task_embeddings)
        logits = self.net(torch.cat([
            task_embeddings, global_features, task_geometry
        ], dim=-1)).squeeze(-1)
        demand = masked_sparsemax(logits, mask, dim=0)
        masked_logits = logits.masked_fill(~mask, -torch.inf)
        return (demand, masked_logits) if return_logits else demand

    def score_modes(self, task_embeddings: torch.Tensor,
                    alive_mask: torch.Tensor, num_agents: int,
                    capacity_modes, capture_size: int = 3,
                    task_geometry: torch.Tensor | None = None):
        """Return independent logits/probabilities for feasible count modes.

        Unlike a single demand simplex, this branch can retain several
        asymmetric hypotheses (for example both 3+1 and 1+3) without
        averaging them into 2+2.
        """
        mask = alive_mask.to(device=task_embeddings.device, dtype=torch.bool)
        if task_embeddings.ndim != 2 or mask.shape != (task_embeddings.shape[0],):
            raise ValueError("task embeddings and alive mask must be aligned")
        modes = torch.as_tensor(
            capacity_modes, dtype=task_embeddings.dtype,
            device=task_embeddings.device,
        )
        if modes.ndim != 2 or modes.shape[1] != task_embeddings.shape[0]:
            raise ValueError("capacity modes must have shape [K, num_tasks]")
        if modes.shape[0] == 0 or int(num_agents) <= 0:
            raise ValueError("at least one mode and one agent are required")
        if (modes < 0).any() or not torch.allclose(
                modes.sum(-1), modes.new_full((modes.shape[0],), float(num_agents))):
            raise ValueError("capacity modes must be nonnegative and sum to num_agents")
        if torch.any(modes[:, ~mask] != 0):
            raise ValueError("dead tasks must receive zero members")
        if task_embeddings.shape[0] > self.max_tasks:
            raise ValueError("active task tensor exceeds configured max_tasks")
        if task_geometry is None:
            geometry = task_embeddings.new_zeros(
                (task_embeddings.shape[0], DEMAND_GEOMETRY_DIM)
            )
        else:
            geometry = task_geometry.to(task_embeddings)
            if geometry.shape != (task_embeddings.shape[0], DEMAND_GEOMETRY_DIM):
                raise ValueError("task geometry has invalid shape")

        weights = modes / float(num_agents)
        pooled_tasks = weights @ task_embeddings
        pooled_geometry = weights @ geometry
        padded_counts = task_embeddings.new_zeros((len(modes), self.max_tasks))
        padded_counts[:, :modes.shape[1]] = weights
        live_count = int(mask.sum().item())
        global_features = task_embeddings.new_tensor([
            float(num_agents) / max(self.max_agents, 1),
            float(live_count) / max(self.max_tasks, 1),
            float(num_agents) / max(live_count * int(capture_size), 1),
        ]).expand(len(modes), -1)
        logits = self.mode_net(torch.cat([
            pooled_tasks, pooled_geometry, padded_counts, global_features,
        ], dim=-1)).squeeze(-1)
        return logits, torch.softmax(logits, dim=0)


def capacities_from_sparse_demand(demand: torch.Tensor, alive_mask: torch.Tensor,
                                  num_agents: int) -> torch.Tensor:
    """Convert relative demand into nonnegative capacities summing to N."""
    mask = alive_mask.to(device=demand.device, dtype=torch.bool)
    if mask.shape != demand.shape:
        mask = torch.broadcast_to(mask, demand.shape)
    live_count = int(mask.sum().item())
    if live_count == 0:
        return torch.zeros_like(demand)
    if int(num_agents) < 0:
        raise ValueError("num_agents must be nonnegative")
    active_demand = torch.where(mask, demand, torch.zeros_like(demand))
    mass = active_demand.sum()
    valid_mass = torch.isfinite(mass) & (mass > torch.finfo(demand.dtype).eps)
    normalized = active_demand / mass.clamp_min(torch.finfo(demand.dtype).eps)
    uniform = mask.to(demand.dtype) / float(live_count)
    normalized = torch.where(valid_mass, normalized, uniform)
    return float(int(num_agents)) * normalized


def neighbor_capacity_modes(capacity: torch.Tensor, alive_mask: torch.Tensor,
                            num_agents: int, limit: int,
                            current_mode=None) -> list[tuple[int, ...]]:
    """Build a fixed-budget diverse neighborhood without composition enumeration."""
    if int(num_agents) < 0:
        raise ValueError("num_agents must be nonnegative")
    mask = alive_mask.to(device=capacity.device, dtype=torch.bool)
    if mask.shape != capacity.shape:
        mask = torch.broadcast_to(mask, capacity.shape)
    live_positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not live_positions or int(limit) <= 0:
        return []

    live_count = len(live_positions)
    total = int(num_agents)
    if live_count == 1:
        mode = [0] * capacity.numel()
        mode[live_positions[0]] = total
        return [tuple(mode)]

    target = capacity.detach().to(dtype=torch.float64, device="cpu")
    active_target = np.asarray([float(target[pos]) for pos in live_positions], dtype=float)
    mass = float(active_target.sum())
    if not np.isfinite(mass) or mass <= 0:
        active_target = np.full(live_count, total / live_count, dtype=float)
    else:
        active_target *= total / mass

    base = np.floor(active_target).astype(int)
    residual = total - int(base.sum())
    fractional = active_target - base
    if residual > 0:
        order = sorted(range(live_count), key=lambda i: (-fractional[i], i))
        for index in order[:residual]:
            base[index] += 1

    def expand(active_values):
        full = [0] * capacity.numel()
        for position, value in zip(live_positions, active_values):
            full[position] = int(value)
        return tuple(full)

    candidates = {expand(base): 0}
    if current_mode is not None:
        incumbent = tuple(int(value) for value in current_mode)
        if (len(incumbent) == capacity.numel() and sum(incumbent) == total
                and all(value >= 0 for value in incumbent)
                and all(incumbent[pos] == 0 for pos in range(capacity.numel())
                        if pos not in live_positions)):
            candidates[incumbent] = min(candidates.get(incumbent, 1), 1)

    frontier = {tuple(base.tolist())}
    depth = 1
    budget = max(int(limit) * 4, live_count + 1)
    while len(candidates) < budget and frontier and depth <= total:
        next_frontier = set()
        for values in frontier:
            for donor in range(live_count):
                if values[donor] <= 0:
                    continue
                for receiver in range(live_count):
                    if donor == receiver:
                        continue
                    moved = list(values)
                    moved[donor] -= 1
                    moved[receiver] += 1
                    moved = tuple(moved)
                    mode = expand(moved)
                    candidates[mode] = min(candidates.get(mode, depth + 1), depth + 1)
                    next_frontier.add(moved)
        frontier = next_frontier
        depth += 1

    # Explicit zero-member boundary proposals preserve the sparse-demand
    # semantics even when the local budget fills near a balanced center.
    for starved in range(live_count):
        values = base.copy()
        supply = int(values[starved])
        values[starved] = 0
        receivers = [i for i in range(live_count) if i != starved]
        receivers.sort(key=lambda i: (-active_target[i], i))
        for offset in range(supply):
            values[receivers[offset % len(receivers)]] += 1
        mode = expand(values)
        candidates[mode] = min(candidates.get(mode, total + 1), total + 1)

    def rank(item):
        mode, priority = item
        distance = sum((mode[pos] - float(target[pos])) ** 2
                       for pos in live_positions)
        return priority, distance, mode

    return [mode for mode, _ in sorted(candidates.items(), key=rank)[:int(limit)]]
