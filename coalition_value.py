"""Small, independently trainable value model for complete coalitions.

The first usable version is trained against a deterministic geometric proxy.
It deliberately does not claim to predict long-horizon capture return; the
checkpoint metadata records that limitation so it cannot be confused with a
future rollout-calibrated model.
"""

from __future__ import annotations

import itertools
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VALUE_SCHEMA = "coalition_value_geometric_proxy_v1"
GEOMETRY_DIM = 7


def generate_candidate_assignments(num_agents: int,
                                   num_tasks: int) -> List[np.ndarray]:
    """Return all labelled assignments with every active task non-empty."""
    if num_agents < num_tasks or num_tasks <= 0:
        return []
    result = []
    for values in itertools.product(range(num_tasks), repeat=num_agents):
        assignment = np.asarray(values, dtype=np.int64)
        if np.all(np.bincount(assignment, minlength=num_tasks) > 0):
            result.append(assignment)
    return result


def _max_angular_gap(relative_positions: np.ndarray) -> float:
    if len(relative_positions) < 2:
        return 2.0 * math.pi
    angles = np.sort(np.arctan2(relative_positions[:, 1],
                                relative_positions[:, 0]))
    wrapped = np.r_[angles, angles[0] + 2.0 * math.pi]
    return float(np.diff(wrapped).max())


def build_candidate_features(pursuer_embeddings: torch.Tensor,
                             task_embeddings: torch.Tensor,
                             pursuer_positions: np.ndarray,
                             task_positions: np.ndarray,
                             assignment: np.ndarray) -> torch.Tensor:
    """Build one permutation-invariant feature row per active task.

    Each row contains the task embedding, mean member embedding, and seven
    transparent geometric statistics.  Rows retain task order; member order
    is removed by mean pooling and symmetric statistics.
    """
    num_agents = int(pursuer_embeddings.shape[0])
    rows = []
    for task_id in range(int(task_embeddings.shape[0])):
        member_ids = np.flatnonzero(np.asarray(assignment) == task_id)
        if len(member_ids):
            member_index = torch.as_tensor(
                member_ids, dtype=torch.long, device=pursuer_embeddings.device,
            )
            pooled = pursuer_embeddings[member_index].mean(dim=0)
            relative = np.asarray(pursuer_positions)[member_ids, :2] - \
                np.asarray(task_positions)[task_id, :2]
            distances = np.linalg.norm(relative, axis=1)
            gap = _max_angular_gap(relative)
            centroid_distance = float(np.linalg.norm(relative.mean(axis=0)))
            geometry = [
                len(member_ids) / max(num_agents, 1),
                float(distances.mean()),
                float(distances.min()),
                float(distances.max()),
                gap / (2.0 * math.pi),
                centroid_distance,
                float(len(member_ids) >= 3 and gap < math.pi),
            ]
        else:
            pooled = torch.zeros_like(pursuer_embeddings[0])
            geometry = [0.0, 10.0, 10.0, 10.0, 1.0, 10.0, 0.0]
        geom_t = torch.as_tensor(
            geometry, dtype=pursuer_embeddings.dtype,
            device=pursuer_embeddings.device,
        )
        rows.append(torch.cat([task_embeddings[task_id], pooled, geom_t]))
    return torch.stack(rows)


def geometric_proxy_value(pursuer_positions: np.ndarray,
                          task_positions: np.ndarray,
                          assignment: np.ndarray,
                          capture_size: int = 3) -> float:
    """Fast supervision favouring capture feasibility plus continued tracking.

    A coalition receives a large bonus only once it has enough members to
    capture.  Distance and angular coverage refine the ranking, while every
    non-empty task gets a small tracking value.  This makes the proxy useful
    for candidate screening without pretending it is a rollout return.
    """
    score = 0.0
    for task_id in range(len(task_positions)):
        member_ids = np.flatnonzero(np.asarray(assignment) == task_id)
        if not len(member_ids):
            score -= 4.0
            continue
        relative = np.asarray(pursuer_positions)[member_ids, :2] - \
            np.asarray(task_positions)[task_id, :2]
        distances = np.linalg.norm(relative, axis=1)
        closeness = math.exp(-float(distances.mean()) / 2.0)
        score += 0.25 * closeness
        if len(member_ids) >= capture_size:
            gap = _max_angular_gap(relative)
            coverage = 1.0 - gap / (2.0 * math.pi)
            score += 3.0 + 1.5 * coverage + 1.5 * closeness
    return float(score)


def pairwise_ranking_loss(predictions: torch.Tensor,
                          labels: torch.Tensor,
                          margin: float = 0.1) -> torch.Tensor:
    """Hinge ranking loss over all unequal-label pairs in one candidate set."""
    pred_delta = predictions[:, None] - predictions[None, :]
    label_delta = labels[:, None] - labels[None, :]
    mask = label_delta > 1e-8
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    return F.relu(float(margin) - pred_delta[mask]).mean()


class CoalitionValueHead(nn.Module):
    """Set-invariant value head over task-conditioned coalition summaries."""

    def __init__(self, hidden_dim: int, width: int = 64):
        super().__init__()
        task_feature_dim = 2 * int(hidden_dim) + GEOMETRY_DIM
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1),
        )
        self.hidden_dim = int(hidden_dim)

    def forward(self, task_features: torch.Tensor,
                task_mask: torch.Tensor | None = None) -> torch.Tensor:
        unbatched = task_features.ndim == 2
        if unbatched:
            task_features = task_features.unsqueeze(0)
        encoded = self.task_encoder(task_features)
        if task_mask is None:
            task_mask = torch.ones(encoded.shape[:2], dtype=torch.bool,
                                   device=encoded.device)
        elif task_mask.ndim == 1:
            task_mask = task_mask.unsqueeze(0)
        mask = task_mask.to(encoded.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(~task_mask.bool().unsqueeze(-1),
                                      neg_inf).max(dim=1).values
        output = self.value(torch.cat([mean_pool, max_pool], dim=-1)).squeeze(-1)
        return output.squeeze(0) if unbatched else output
