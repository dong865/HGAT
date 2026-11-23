"""Dynamic-state counterfactual branches for V3 coalition values."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment


def fork_environment(env):
    """Return an independent exact copy, including RNG and shaping caches."""
    return copy.deepcopy(env)


def summarize_branch(*, rewards, captured_before: int, captured_after: int,
                     elapsed_steps: int, time_step: float,
                     first_capture_step=None, gamma: float = 1.0):
    rewards = [float(value) for value in rewards]
    if (elapsed_steps < 0 or time_step <= 0 or not 0 < gamma <= 1
            or any(not math.isfinite(value) for value in rewards)):
        raise ValueError("invalid branch outcome")
    discounted = sum((float(gamma) ** index) * value
                     for index, value in enumerate(rewards))
    new_captures = max(0, int(captured_after) - int(captured_before))
    delay = (None if first_capture_step is None else
             float(first_capture_step) * float(time_step))
    # Lexicographic semantics encoded at separated numerical scales.
    time_term = (float(elapsed_steps) * float(time_step)
                 if delay is None else delay)
    ranking_key = (1_000_000.0 * new_captures - 1_000.0 * time_term
                   + math.tanh(discounted / 100.0))
    return {
        "new_captures": new_captures,
        "first_capture_delay": delay,
        "discounted_return": float(discounted),
        "ranking_key": float(ranking_key),
    }


@torch.no_grad()
def rollout_forced_assignment(env, encoder, actor, first_encoded,
                              assignment, horizon_steps: int,
                              max_steps: int, gamma: float = 1.0):
    """Fork at the current decision state and hold one alliance for H steps."""
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    branch = fork_environment(env)
    branch_encoder = copy.deepcopy(encoder)
    initial = np.asarray(assignment, dtype=np.int64).copy()
    if initial.shape != (branch.num_pursuers,):
        raise ValueError("assignment must contain every active pursuer")
    captured_before = int(branch.captured[:branch.num_evaders].sum())
    rewards, first_capture_step = [], None
    for offset in range(int(horizon_steps)):
        active = np.flatnonzero(branch.task_mask[:branch.num_evaders])
        if len(active) == 0 or branch.done:
            break
        repaired = initial.copy()
        repaired[~np.isin(repaired, active)] = int(active[0])
        apply_membership(branch, repaired)
        snapshot = branch.snapshot()
        encoded = first_encoded if offset == 0 else branch_encoder.encode(
            snapshot, differentiable=False
        )
        membership = hard_K_from_assignment(
            repaired, branch.task_mask, 6, 3,
            encoded["pursuer_embeddings"].device,
        )
        obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
        mask = torch.as_tensor(
            branch.pursuer_mask, dtype=torch.float32, device=obs.device,
        )
        actions = (actor(obs) * mask.unsqueeze(-1)).cpu().numpy()
        reward, done, info = branch.step(actions, max_steps=max_steps)
        rewards.append(float(reward))
        if first_capture_step is None and info.get("capture_events"):
            first_capture_step = offset + 1
        if done:
            break
    summary = summarize_branch(
        rewards=rewards, captured_before=captured_before,
        captured_after=int(branch.captured[:branch.num_evaders].sum()),
        elapsed_steps=len(rewards), time_step=branch.time_step,
        first_capture_step=first_capture_step, gamma=gamma,
    )
    summary.update({"steps": len(rewards), "assignment": initial.tolist(),
                    "outcome": str(branch.outcome)})
    return summary


def counterfactual_value_loss(predictions: torch.Tensor, branch_results,
                              rank_margin: float = 0.2,
                              regression_weight: float = 0.1):
    """Same-state regression plus pairwise ordering for coalition values."""
    rows = list(branch_results)
    if predictions.ndim != 1 or len(rows) != predictions.numel() or not rows:
        raise ValueError("predictions and branch results must be aligned")
    returns = predictions.new_tensor([
        float(row["discounted_return"]) for row in rows
    ])
    keys = predictions.new_tensor([float(row["ranking_key"]) for row in rows])
    losses = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            sign = torch.sign(keys[left] - keys[right])
            if sign != 0:
                losses.append(torch.relu(
                    predictions.new_tensor(float(rank_margin))
                    - sign * (predictions[left] - predictions[right])
                ))
    ranking = (torch.stack(losses).mean() if losses else
               predictions.sum() * 0.0)
    regression = F.smooth_l1_loss(predictions, returns)
    return ranking + float(regression_weight) * regression


def evaluate_candidate_branches(env, policy, decision, candidates,
                                horizon_steps: int, max_steps: int,
                                gamma: float = 1.0):
    """Evaluate several assignments from exactly one online decision state."""
    rows = np.asarray(candidates, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != env.num_pursuers:
        raise ValueError("candidates must have shape [K, num_pursuers]")
    return [rollout_forced_assignment(
        env, policy.modules["encoder"], policy.modules["actor"],
        decision["encoded"], row, horizon_steps, max_steps, gamma,
    ) for row in rows]
