"""Scale-aware coalition teacher and normalized-demand student for V3.

The value model is an offline teacher.  Deployment consumes only the demand
student, whose output is a simplex over the resources remaining after one
member has been reserved for every active task.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coalition_value import (
    GEOMETRY_DIM,
    build_candidate_features,
    pairwise_ranking_loss,
)
from .v3_allocation import normalized_demand_from_candidates


V3_TEACHER_SCHEMA = "coalition_value_scale_aware_teacher_v2"
V3_DEMAND_SCHEMA = "normalized_demand_student_v3"
SCALE_FEATURE_DIM = 6
DEMAND_GEOMETRY_DIM = 6


def build_demand_geometry(pursuer_positions: np.ndarray,
                          task_positions: np.ndarray,
                          map_size: float) -> torch.Tensor:
    """Permutation-invariant per-task geometry available during deployment."""
    pursuers = np.asarray(pursuer_positions, dtype=np.float64)[:, :2]
    tasks = np.asarray(task_positions, dtype=np.float64)[:, :2]
    rows = []
    scale = max(float(map_size), 1e-6)
    for task in tasks:
        relative = pursuers - task
        distances = np.linalg.norm(relative, axis=1)
        order = np.argsort(distances)
        near = relative[order[:min(3, len(order))]]
        angles = np.sort(np.arctan2(near[:, 1], near[:, 0]))
        if len(angles) >= 2:
            gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
            max_gap = float(gaps.max()) / (2.0 * math.pi)
        else:
            max_gap = 1.0
        third = distances[order[min(2, len(order) - 1)]]
        rows.append([
            float(distances.mean()) / scale,
            float(distances.min()) / scale,
            float(distances.max()) / scale,
            float(third) / scale,
            max_gap,
            float(np.linalg.norm(near.mean(axis=0))) / scale,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_scale_aware_candidate_features(
    pursuer_embeddings: torch.Tensor,
    task_embeddings: torch.Tensor,
    pursuer_positions: np.ndarray,
    task_positions: np.ndarray,
    assignment: np.ndarray,
    capture_size: int = 3,
) -> torch.Tensor:
    """Extend the original invariant features with resource-scale semantics."""
    base = build_candidate_features(
        pursuer_embeddings, task_embeddings, pursuer_positions,
        task_positions, assignment,
    )
    num_agents = int(len(assignment))
    num_tasks = int(task_embeddings.shape[0])
    scarcity = num_agents / max(num_tasks * int(capture_size), 1)
    rows = []
    for task_id in range(num_tasks):
        count = int(np.count_nonzero(np.asarray(assignment) == task_id))
        scale = torch.as_tensor([
            float(count),
            count / max(float(capture_size), 1.0),
            (count - float(capture_size)) / max(float(capture_size), 1.0),
            float(num_agents),
            float(num_tasks),
            float(scarcity),
        ], dtype=base.dtype, device=base.device)
        rows.append(torch.cat([base[task_id], scale]))
    return torch.stack(rows)


class ScaleAwareCoalitionValueHead(nn.Module):
    """Permutation-invariant offline value teacher with explicit scale cues."""

    def __init__(self, hidden_dim: int, width: int = 64):
        super().__init__()
        feature_dim = 2 * int(hidden_dim) + GEOMETRY_DIM + SCALE_FEATURE_DIM
        self.task_encoder = nn.Sequential(
            nn.Linear(feature_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.hidden_dim = int(hidden_dim)

    def forward(self, features: torch.Tensor,
                task_mask: torch.Tensor | None = None) -> torch.Tensor:
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        encoded = self.task_encoder(features)
        if task_mask is None:
            task_mask = torch.ones(encoded.shape[:2], dtype=torch.bool,
                                   device=encoded.device)
        elif task_mask.ndim == 1:
            task_mask = task_mask.unsqueeze(0)
        mask = task_mask.bool().unsqueeze(-1)
        weights = mask.to(encoded.dtype)
        mean = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        maximum = encoded.masked_fill(
            ~mask, torch.finfo(encoded.dtype).min
        ).max(1).values
        result = self.value(torch.cat([mean, maximum], dim=-1)).squeeze(-1)
        return result.squeeze(0) if unbatched else result


def demand_target_from_values(
    values: torch.Tensor,
    candidates: np.ndarray,
    num_tasks: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    demand, _ = normalized_demand_from_candidates(
        values, candidates, num_tasks, temperature,
    )
    return demand


def best_mode_demand_target(values: torch.Tensor, candidates: np.ndarray,
                            num_agents: int, num_tasks: int):
    """Select the highest-mean integer count mode without multiplicity bias."""
    values = torch.as_tensor(values)
    rows = np.asarray(candidates)
    if (values.ndim != 1 or rows.ndim != 2 or len(rows) != values.numel()
            or rows.shape[1] != int(num_agents) or int(num_tasks) < 1
            or not torch.isfinite(values).all()):
        raise ValueError("values and candidates must be finite and aligned")
    grouped = {}
    for index, row in enumerate(rows):
        mode = tuple(np.bincount(row, minlength=int(num_tasks)).tolist())
        grouped.setdefault(mode, []).append(index)
    mode_scores = []
    for mode, indices in grouped.items():
        score = values[indices].mean()
        mode_scores.append((float(score.detach().cpu()), mode))
    _, best_mode = max(mode_scores, key=lambda item: (item[0], item[1]))
    target = values.new_tensor(best_mode) / float(num_agents)
    return target, best_mode


class NormalizedDemandHead(nn.Module):
    """Online student predicting a masked relative resource-demand simplex."""

    def __init__(self, hidden_dim: int, width: int = 64,
                 max_agents: int = 6, max_tasks: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim) + 3 + DEMAND_GEOMETRY_DIM, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.max_agents = int(max_agents)
        self.max_tasks = int(max_tasks)

    def forward(self, task_embeddings: torch.Tensor, task_mask: torch.Tensor,
                num_agents: int, capture_size: int = 3,
                task_geometry: torch.Tensor | None = None) -> torch.Tensor:
        mask = task_mask.bool()
        num_tasks = int(mask.sum().item())
        if num_tasks == 0:
            return task_embeddings.new_zeros(task_embeddings.shape[0])
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
        ], -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        demand = torch.softmax(logits, dim=0)
        return torch.where(mask, demand, torch.zeros_like(demand))


def capacities_from_demand(demand: torch.Tensor, task_mask: torch.Tensor,
                           num_agents: int) -> torch.Tensor:
    """Reserve one member/task, then distribute all residual supply."""
    mask = task_mask.to(demand.dtype)
    num_tasks = int(task_mask.bool().sum().item())
    if num_tasks == 0:
        return torch.zeros_like(demand)
    residual = int(num_agents) - num_tasks
    if residual < 0:
        raise ValueError("num_agents must be at least the number of active tasks")
    normalized = demand * mask
    normalized = normalized / normalized.sum().clamp_min(
        torch.finfo(demand.dtype).eps
    )
    return mask + float(residual) * normalized


def fit_scale_aware_teacher(
    model: ScaleAwareCoalitionValueHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 200,
    lr: float = 3e-3,
    rank_weight: float = 1.0,
) -> dict:
    """Fit one teacher on candidate rows; encoder features stay detached."""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    target = labels.detach()
    target = (target - target.mean()) / target.std().clamp_min(1e-6)
    last = torch.tensor(float("nan"), device=features.device)
    model.train()
    for _ in range(int(epochs)):
        if features.ndim == 4:
            prediction = model(features.detach().flatten(0, 1)).reshape(target.shape)
            regression = F.smooth_l1_loss(prediction, target)
            ranking = torch.stack([
                pairwise_ranking_loss(pred, truth)
                for pred, truth in zip(prediction, target)
            ]).mean()
        else:
            prediction = model(features.detach()).reshape(-1)
            regression = F.smooth_l1_loss(prediction, target)
            ranking = pairwise_ranking_loss(prediction, target)
        last = regression + float(rank_weight) * ranking
        optimizer.zero_grad(set_to_none=True)
        last.backward()
        optimizer.step()
    model.eval()
    return {"loss": float(last.detach())}


def fit_demand_student(
    model: NormalizedDemandHead,
    task_embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_agents: int,
    epochs: int = 200,
    lr: float = 3e-3,
    capture_size: int = 3,
    task_geometry: torch.Tensor | None = None,
) -> dict:
    """Distil teacher-derived demand distributions into the online student."""
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    mask = torch.ones(
        task_embeddings.shape[1], dtype=torch.bool,
        device=task_embeddings.device,
    )
    last = torch.tensor(float("nan"), device=task_embeddings.device)
    model.train()
    for _ in range(int(epochs)):
        predictions = torch.stack([
            model(
                row.detach(), mask, num_agents=num_agents,
                capture_size=capture_size,
                task_geometry=(None if task_geometry is None
                               else task_geometry[index].detach()),
            )
            for index, row in enumerate(task_embeddings)
        ])
        last = -(targets.detach() * predictions.clamp_min(1e-8).log()).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        last.backward()
        optimizer.step()
    model.eval()
    return {"loss": float(last.detach())}
