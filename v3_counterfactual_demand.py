"""Counterfactual capacity-mode supervision for the V3 demand head."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def mode_targets_from_rollouts(capacity_modes, rollouts, *,
                               horizon_seconds: float,
                               temperature: float = 0.25):
    """Convert paired same-state rollout outcomes into a mode distribution."""
    modes = [tuple(int(value) for value in mode) for mode in capacity_modes]
    rows = list(rollouts)
    if not modes or len(modes) != len(rows):
        raise ValueError("capacity modes and rollouts must be nonempty and aligned")
    if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive and finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scores = []
    for row in rows:
        captured = int(row.get("captured", 0))
        reward = float(row.get("reward", 0.0))
        first = row.get("first_capture_time")
        if captured < 0 or not math.isfinite(reward):
            raise ValueError("rollout outcomes must be finite")
        time_bonus = 0.0
        if first is not None:
            first = float(first)
            if not math.isfinite(first) or first < 0:
                raise ValueError("first capture time must be nonnegative and finite")
            time_bonus = max(0.0, 1.0 - first / horizon_seconds)
        # Captures dominate, while timing and bounded raw reward break ties.
        scores.append(4.0 * captured + time_bonus + 0.25 * math.tanh(reward / 100.0))
    scores = np.asarray(scores, dtype=np.float64)
    centered = (scores - scores.max()) / float(temperature)
    weights = np.exp(centered)
    target = weights / weights.sum()
    return target, scores


def fit_counterfactual_demand(head, scenes, *, epochs: int = 100,
                              learning_rate: float = 1e-3,
                              task_loss_weight: float = 0.25):
    """Fit only mode-candidate recall; leave continuous demand untouched."""
    rows = list(scenes)
    if not rows or epochs < 1 or learning_rate <= 0:
        raise ValueError("training requires scenes, epochs and positive learning rate")
    optimizer = torch.optim.Adam(head.mode_net.parameters(),
                                 lr=float(learning_rate))
    history = []
    head.train()
    for _ in range(int(epochs)):
        losses = []
        for row in rows:
            embeddings = row["task_embeddings"]
            alive = row["alive_mask"]
            modes = row["modes"]
            geometry = row["task_geometry"]
            mode_logits, _ = head.score_modes(
                embeddings, alive, row["num_agents"], modes,
                row.get("capture_size", 3), geometry,
            )
            good = torch.as_tensor(
                row["good_mode_mask"], dtype=torch.bool,
                device=embeddings.device,
            )
            if good.shape != mode_logits.shape or not good.any():
                raise ValueError("each scene needs at least one aligned good mode")
            # Independent multi-label supervision lets both symmetric 3+1
            # and 1+3 hypotheses be high without averaging into 2+2.
            losses.append(F.binary_cross_entropy_with_logits(
                mode_logits, good.to(mode_logits.dtype)
            ))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
