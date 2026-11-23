"""Unified candidate value scoring and stable sparse-demand allocation."""

from __future__ import annotations

import numpy as np
import torch

from .v3_candidate_generator import generate_bounded_candidates
from .v3_demand_distillation import (
    ScaleAwareCoalitionValueHead,
    build_demand_geometry,
    build_scale_aware_candidate_features,
)
from .v3_sparse_demand import (
    SparseDemandHead,
    capacities_from_sparse_demand,
    neighbor_capacity_modes,
)


def score_candidates(values, candidates, capacity, previous_assignment,
                     demand_weight, switch_weight) -> torch.Tensor:
    """Score each candidate after soft capacity and assignment-change costs."""
    values = torch.as_tensor(values)
    capacity = torch.as_tensor(capacity, dtype=values.dtype, device=values.device)
    raw_candidates = np.asarray(candidates)
    if (values.ndim != 1 or capacity.ndim != 1 or raw_candidates.ndim != 2
            or raw_candidates.shape[0] != values.numel()
            or raw_candidates.shape[1] == 0 or capacity.numel() == 0
            or not np.issubdtype(raw_candidates.dtype, np.integer)):
        raise ValueError("values, candidates, or capacity have invalid shape or dtype")
    if not torch.isfinite(values).all() or not torch.isfinite(capacity).all():
        raise ValueError("values and capacity must be finite")
    if (not np.isfinite(demand_weight) or not np.isfinite(switch_weight)):
        raise ValueError("score weights must be finite")
    if np.any(raw_candidates < 0) or np.any(raw_candidates >= capacity.numel()):
        raise ValueError("candidate task IDs are out of range")

    assignment = torch.as_tensor(raw_candidates, dtype=torch.long, device=values.device)
    counts = torch.nn.functional.one_hot(
        assignment, num_classes=capacity.numel()
    ).sum(dim=1).to(values.dtype)
    capacity_error = (counts - capacity).square().sum(dim=1)

    changed_fraction = torch.zeros_like(values)
    if previous_assignment is not None:
        previous = np.asarray(previous_assignment)
        if (previous.shape != (assignment.shape[1],)
                or not np.issubdtype(previous.dtype, np.integer)):
            raise ValueError("previous_assignment must contain one integer per agent")
        previous_tensor = torch.as_tensor(previous, device=values.device)
        changed_fraction = (assignment != previous_tensor).to(values.dtype).mean(dim=1)

    return values - float(demand_weight) * capacity_error - float(switch_weight) * changed_fraction


def capture_feasibility_penalty(candidates, alive, capture_size: int,
                                weight: float, remaining_fraction: float,
                                warmup_fraction: float,
                                urgency: bool) -> torch.Tensor:
    """Return a state-dependent penalty for allocations unable to start capture.

    The prior is deliberately outside the learned value so raw network outputs
    remain inspectable.  It activates only while multiple tasks are alive and
    either the configured warmup has elapsed or the geometry is urgent.
    """
    rows = np.asarray(candidates)
    live = np.asarray(alive, dtype=bool)
    if (rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0
            or not np.issubdtype(rows.dtype, np.integer)
            or live.ndim != 1 or live.size == 0
            or int(capture_size) != capture_size or capture_size < 1
            or not np.isfinite(weight) or weight < 0
            or not np.isfinite(remaining_fraction)
            or not 0 <= remaining_fraction <= 1
            or not np.isfinite(warmup_fraction)
            or not 0 <= warmup_fraction <= 1):
        raise ValueError("invalid capture-feasibility inputs")
    result = torch.zeros(rows.shape[0], dtype=torch.float32)
    if weight == 0 or live.sum() < 2:
        return result
    active = bool(urgency) or remaining_fraction <= warmup_fraction
    if not active:
        return result
    if np.any(rows < 0) or np.any(rows >= live.size):
        raise ValueError("candidate task IDs are out of range")
    counts = np.stack([
        np.bincount(row, minlength=live.size) for row in rows
    ])
    feasible = np.max(counts[:, live], axis=1) >= int(capture_size)
    result[torch.as_tensor(~feasible)] = float(weight)
    return result


class CaptureFreezeGuard:
    """Keep capture-critical coalitions intact with distance hysteresis."""

    def __init__(self, freeze_distance: float, release_distance: float,
                 capture_size: int = 3):
        if (not np.isfinite(freeze_distance) or freeze_distance <= 0
                or not np.isfinite(release_distance)
                or release_distance < freeze_distance
                or int(capture_size) != capture_size or capture_size < 1):
            raise ValueError("invalid capture-freeze configuration")
        self.freeze_distance = float(freeze_distance)
        self.release_distance = float(release_distance)
        self.capture_size = int(capture_size)
        self.reset()

    def reset(self):
        self.frozen = None

    def update(self, assignment, pursuer_positions, task_positions,
               alive_mask, hold_counters):
        assignment = np.asarray(assignment, dtype=np.int64)
        pursuers = np.asarray(pursuer_positions, dtype=np.float64)
        tasks = np.asarray(task_positions, dtype=np.float64)
        alive = np.asarray(alive_mask, dtype=bool)
        holds = np.asarray(hold_counters, dtype=np.int64)
        if (assignment.shape != (len(pursuers),) or pursuers.ndim != 2
                or tasks.ndim != 2 or pursuers.shape[1] < 2 or tasks.shape[1] < 2
                or alive.shape != (len(tasks),) or holds.shape != (len(tasks),)):
            raise ValueError("invalid capture-freeze state")
        if self.frozen is None or self.frozen.shape != alive.shape:
            self.frozen = np.zeros_like(alive)
        result = self.frozen.copy()
        for task_id in range(len(tasks)):
            if not alive[task_id]:
                result[task_id] = False
                continue
            members = np.flatnonzero(assignment == task_id)
            max_distance = (float("inf") if members.size == 0 else float(np.max(
                np.linalg.norm(
                    pursuers[members, :2] - tasks[task_id, :2], axis=1
                )
            )))
            holding = holds[task_id] > 0
            if result[task_id]:
                result[task_id] = bool(holding or max_distance <= self.release_distance)
            else:
                result[task_id] = bool(
                    holding or (
                        members.size >= self.capture_size
                        and max_distance <= self.freeze_distance
                    )
                )
        self.frozen = result
        return result.copy()


class StableAllianceGate:
    """Hold the accepted assignment until dwell and improvement allow a change."""

    def __init__(self, min_dwell_steps: int, improvement_margin: float):
        if int(min_dwell_steps) != min_dwell_steps or min_dwell_steps < 0:
            raise ValueError("min_dwell_steps must be a nonnegative integer")
        if not np.isfinite(improvement_margin) or improvement_margin < 0:
            raise ValueError("improvement_margin must be finite and nonnegative")
        self.min_dwell_steps = int(min_dwell_steps)
        self.improvement_margin = float(improvement_margin)
        self.reset()

    def reset(self):
        self.assignment = None
        self.accepted_score = None
        self.accepted_step = None

    def select(self, candidates, scores, step, force=False, incumbent_score=None,
               frozen_tasks=None):
        rows = np.asarray(candidates)
        score_tensor = torch.as_tensor(scores)
        if (rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0
                or score_tensor.ndim != 1 or score_tensor.numel() != rows.shape[0]
                or not np.issubdtype(rows.dtype, np.integer)
                or not torch.isfinite(score_tensor).all()):
            raise ValueError("candidates and scores must be nonempty and aligned")
        if incumbent_score is not None and not np.isfinite(incumbent_score):
            raise ValueError("incumbent_score must be finite")
        freeze_blocked = False
        if frozen_tasks is not None and self.assignment is not None:
            frozen = np.asarray(frozen_tasks, dtype=bool)
            if frozen.ndim != 1:
                raise ValueError("frozen_tasks must be a one-dimensional mask")
            locked_agents = np.flatnonzero(frozen[self.assignment])
            permitted = np.ones(rows.shape[0], dtype=bool)
            if locked_agents.size:
                permitted = np.all(
                    rows[:, locked_agents] == self.assignment[locked_agents], axis=1
                )
                if not permitted.any():
                    raise ValueError("capture freeze removed every candidate")
                unconstrained_best = int(torch.argmax(score_tensor).item())
                freeze_blocked = not bool(permitted[unconstrained_best])
                score_tensor = score_tensor.masked_fill(
                    ~torch.as_tensor(permitted, device=score_tensor.device), -torch.inf
                )
        best = int(torch.argmax(score_tensor).item())
        best_assignment = rows[best].astype(np.int64, copy=True)
        best_score = float(score_tensor[best].item())

        if self.assignment is None:
            reason = "initial"
        elif force:
            reason = "forced"
        elif freeze_blocked and np.array_equal(best_assignment, self.assignment):
            reason = "capture_freeze"
        elif step - self.accepted_step < self.min_dwell_steps:
            reason = "dwell"
        else:
            matching = np.flatnonzero(np.all(rows == self.assignment, axis=1))
            if matching.size:
                current_score = float(score_tensor[matching[0]].item())
            elif incumbent_score is not None:
                current_score = float(incumbent_score)
            else:
                current_score = self.accepted_score
            reason = ("improved" if best_score > current_score + self.improvement_margin
                      else "margin")

        if reason in ("initial", "forced", "improved"):
            switched = self.assignment is None or not np.array_equal(
                best_assignment, self.assignment
            )
            self.assignment = best_assignment
            self.accepted_score = best_score
            self.accepted_step = int(step)
        else:
            switched = False
        return self.assignment.copy(), {"switched": switched, "reason": reason}


class V3HybridAllocator:
    """Generate sparse-demand modes and select one value-ranked assignment."""

    def __init__(self, demand_head: SparseDemandHead,
                 value_head: ScaleAwareCoalitionValueHead, map_size: float,
                 capture_size: int = 3, mode_limit: int = 8,
                 max_candidates: int = 32, demand_weight: float = 1.0,
                 switch_weight: float = 0.5, min_dwell_steps: int = 0,
                 improvement_margin: float = 0.0,
                 freeze_distance=None,
                 freeze_release_ratio: float = 1.15,
                 mode_prior_weight: float = 0.0,
                 mode_prior_clip: float = 3.0,
                 infeasible_mode_penalty: float = 0.0,
                 infeasible_mode_warmup_fraction: float = 0.5):
        if mode_limit < 1 or max_candidates < mode_limit:
            raise ValueError("candidate limit must cover at least one of every mode")
        self.demand_head = demand_head
        self.value_head = value_head
        self.map_size = float(map_size)
        self.capture_size = int(capture_size)
        self.mode_limit = int(mode_limit)
        self.max_candidates = int(max_candidates)
        self.demand_weight = float(demand_weight)
        self.switch_weight = float(switch_weight)
        self.mode_prior_weight = float(mode_prior_weight)
        self.mode_prior_clip = float(mode_prior_clip)
        if (not np.isfinite(infeasible_mode_penalty)
                or infeasible_mode_penalty < 0
                or not np.isfinite(infeasible_mode_warmup_fraction)
                or not 0 <= infeasible_mode_warmup_fraction <= 1):
            raise ValueError("invalid capture-feasibility configuration")
        self.infeasible_mode_penalty = float(infeasible_mode_penalty)
        self.infeasible_mode_warmup_fraction = float(
            infeasible_mode_warmup_fraction
        )
        self.gate = StableAllianceGate(min_dwell_steps, improvement_margin)
        self.freeze_guard = (None if freeze_distance is None else CaptureFreezeGuard(
            freeze_distance=float(freeze_distance),
            release_distance=float(freeze_distance) * float(freeze_release_ratio),
            capture_size=self.capture_size,
        ))

    def reset(self):
        self.gate.reset()
        if self.freeze_guard is not None:
            self.freeze_guard.reset()

    def allocate(self, encoded: dict, snapshot: dict, alive_mask,
                 step: int, force: bool = False, hold_counters=None,
                 remaining_fraction: float = 1.0,
                 capture_urgent: bool = False) -> dict:
        pursuer_embeddings = encoded["pursuer_embeddings"]
        task_embeddings = encoded["task_embeddings"]
        relation_costs = encoded["relation_costs"]
        num_agents = int(encoded.get("num_pursuers", pursuer_embeddings.shape[0]))
        num_tasks = int(encoded.get("num_evaders", len(alive_mask)))
        if num_agents < 1 or num_tasks < 1:
            raise ValueError("allocation requires at least one agent and task")
        if (num_agents > pursuer_embeddings.shape[0]
                or num_tasks > task_embeddings.shape[0]
                or relation_costs.shape[0] < num_agents
                or relation_costs.shape[1] < num_tasks):
            raise ValueError("active counts exceed encoded tensor shapes")
        alive = torch.as_tensor(alive_mask, dtype=torch.bool,
                                device=task_embeddings.device)[:num_tasks]
        if alive.shape != (num_tasks,) or not alive.any():
            raise ValueError("alive_mask must contain at least one live task")

        positions = np.asarray(snapshot["positions"])
        if positions.ndim != 2 or positions.shape[0] < num_agents + num_tasks:
            raise ValueError("snapshot positions do not contain the active agents and tasks")
        pursuer_positions = positions[:num_agents, :2]
        task_positions = positions[num_agents:num_agents + num_tasks, :2]
        geometry = build_demand_geometry(
            pursuer_positions, task_positions, self.map_size
        ).to(task_embeddings)
        demand = self.demand_head(
            task_embeddings[:num_tasks], alive, num_agents,
            self.capture_size, geometry,
        )
        capacity = capacities_from_sparse_demand(demand, alive, num_agents)
        previous = self.gate.assignment
        invalid_previous = (previous is not None and (
            previous.shape != (num_agents,) or
            np.any(previous < 0) or np.any(previous >= num_tasks) or
            any(not bool(alive[task_id]) for task_id in previous)
        ))
        usable_previous = None if invalid_previous else previous
        current_mode = (None if usable_previous is None else tuple(
            np.bincount(usable_previous, minlength=num_tasks).tolist()
        ))
        modes = neighbor_capacity_modes(
            capacity, alive, num_agents, self.mode_limit,
            current_mode=current_mode,
        )
        if hasattr(self.demand_head, "score_modes"):
            mode_logits, mode_probabilities = self.demand_head.score_modes(
                task_embeddings[:num_tasks], alive, num_agents, modes,
                self.capture_size, geometry,
            )
        else:
            mode_logits = task_embeddings.new_zeros(len(modes))
            mode_probabilities = torch.softmax(mode_logits, dim=0)
        if self.freeze_guard is not None and usable_previous is not None:
            holds = (np.zeros(num_tasks, dtype=np.int64) if hold_counters is None
                     else np.asarray(hold_counters)[:num_tasks])
            frozen_tasks = self.freeze_guard.update(
                usable_previous, pursuer_positions, task_positions,
                alive.detach().cpu().numpy(), holds,
            )
        else:
            frozen_tasks = np.zeros(num_tasks, dtype=bool)
        candidates = generate_bounded_candidates(
            relation_costs[:num_agents, :num_tasks], modes, alive,
            current_assignment=usable_previous, max_candidates=self.max_candidates,
        )
        if (frozen_tasks.any() and usable_previous is not None
                and not np.any(np.all(candidates == usable_previous, axis=1))):
            if len(candidates) >= self.max_candidates:
                candidates = np.vstack([usable_previous, candidates[:-1]])
            else:
                candidates = np.vstack([usable_previous, candidates])
        features = torch.stack([
            build_scale_aware_candidate_features(
                pursuer_embeddings[:num_agents], task_embeddings[:num_tasks],
                pursuer_positions, task_positions, candidate, self.capture_size,
            ) for candidate in candidates
        ])
        raw_values = self.value_head(features, alive).reshape(-1)
        unified_scores = score_candidates(
            raw_values, candidates, capacity, usable_previous,
            self.demand_weight, self.switch_weight,
        )
        feasibility_penalty = capture_feasibility_penalty(
            candidates, alive.detach().cpu().numpy(), self.capture_size,
            self.infeasible_mode_penalty, remaining_fraction,
            self.infeasible_mode_warmup_fraction, capture_urgent,
        ).to(device=unified_scores.device, dtype=unified_scores.dtype)
        unified_scores = unified_scores - feasibility_penalty
        mode_to_probability = {
            tuple(mode): mode_probabilities[index]
            for index, mode in enumerate(modes)
        }
        candidate_mode_probabilities = torch.stack([
            mode_to_probability[tuple(np.bincount(
                candidate, minlength=num_tasks
            ).tolist())]
            for candidate in candidates
        ])
        if self.mode_prior_weight != 0.0:
            log_prior = candidate_mode_probabilities.clamp_min(1e-8).log()
            log_prior = log_prior.clamp(min=-self.mode_prior_clip, max=0.0)
            unified_scores = unified_scores + self.mode_prior_weight * log_prior
        incumbent_score = None
        if usable_previous is not None and not np.any(
                np.all(candidates == usable_previous, axis=1)):
            incumbent_features = build_scale_aware_candidate_features(
                pursuer_embeddings[:num_agents], task_embeddings[:num_tasks],
                pursuer_positions, task_positions, usable_previous, self.capture_size,
            )
            incumbent_value = self.value_head(
                incumbent_features.unsqueeze(0), alive
            ).reshape(-1)
            incumbent_unified = score_candidates(
                incumbent_value, usable_previous.reshape(1, -1), capacity,
                usable_previous, self.demand_weight, self.switch_weight,
            )
            incumbent_score = float(incumbent_unified[0].item())
        assignment, info = self.gate.select(
            candidates, unified_scores, step, force=force or invalid_previous,
            incumbent_score=incumbent_score, frozen_tasks=frozen_tasks,
        )
        return {
            "assignment": assignment,
            "demand": demand,
            "capacity": capacity,
            "modes": modes,
            "mode_logits": mode_logits,
            "mode_probabilities": mode_probabilities,
            "candidate_mode_probabilities": candidate_mode_probabilities,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "raw_values": raw_values,
            "unified_scores": unified_scores,
            "feasibility_penalty": feasibility_penalty,
            "switched": info["switched"],
            "reason": info["reason"],
            "frozen_tasks": frozen_tasks,
        }
