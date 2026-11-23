"""Real-flight V3 runtime with an oracle-style locked initial assignment.

The checkpoint loading, observation path, lifecycle handling and safety
projections are the same as :class:`V3HybridRuntime`.  The online allocator is
constructed only to preserve that loading/configuration contract; it is never
called.  Every decision applies ``lock_assignment`` and redirects members of a
captured task to the first remaining live task, matching ``v3_08_acceptance``.
"""

from typing import Sequence, Tuple

import numpy as np
import torch

from .runtime_v3_hybrid import V3HybridRuntime, limit_vector_acceleration
from .safety import filter_velocity
from .stage_env import MODE_EXIT, MODE_NAMES
from .train_v3_hybrid import (blend_post_capture_ring_actions,
                              should_use_post_capture_ring)
from .traditional_ring_baseline import RingGuidanceController
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment


class V3LockedRuntime(V3HybridRuntime):
    """V3 actor with a fixed 4v2 assignment and capture-triggered repair."""

    def __init__(self, checkpoint_path: str,
                 lock_assignment: Sequence[int] = (1, 0, 1, 1), **kwargs):
        super().__init__(checkpoint_path, **kwargs)
        values = list(lock_assignment)
        if len(values) != self.num_pursuers:
            raise ValueError(
                f"lock_assignment must contain {self.num_pursuers} task indices")
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise ValueError("lock_assignment entries must be integers")
        values = [int(value) for value in values]
        if any(value < 0 or value >= self.num_evaders for value in values):
            raise ValueError(
                f"lock_assignment entries must be in [0, {self.num_evaders - 1}]")
        self.lock_assignment = tuple(values)
        self.last_allocation = {"assignment": np.asarray(values, dtype=np.int64),
                                "reason": "locked", "switched": False}

    def _locked_assignment(self) -> np.ndarray:
        """Apply the oracle repair: dead-task members join the first live task."""
        active = np.flatnonzero(self.env.task_mask[:self.num_evaders])
        if not active.size:
            # act() never reaches actor inference after both tasks are captured;
            # retain a finite assignment for introspection if called directly.
            return np.zeros(self.num_pursuers, dtype=np.int64)
        assignment = np.asarray(self.lock_assignment, dtype=np.int64).copy()
        assignment[~np.isin(assignment, active)] = int(active[0])
        return assignment

    def act(self, positions_world: np.ndarray, velocities_world: np.ndarray,
            arena_origin_world: Sequence[float] = (0.0, 0.0)
            ) -> Tuple[np.ndarray, dict]:
        positions = np.asarray(positions_world, dtype=np.float32)
        velocities = np.asarray(velocities_world, dtype=np.float32)
        valid_shapes = {(self.num_agents, 2), (self.num_agents, 3)}
        if positions.shape not in valid_shapes or velocities.shape not in valid_shapes:
            raise ValueError(
                f"positions and velocities must both have shape "
                f"({self.num_agents}, 2) or ({self.num_agents}, 3)")
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("non-finite mocap state")

        origin = np.asarray(arena_origin_world, dtype=np.float32)
        local = np.zeros((self.num_agents, 3), dtype=np.float32)
        local[:, :2] = positions[:, :2] - origin[:2] + self.map_size / 2.0
        if positions.shape[1] == 3:
            local[:, 2] = positions[:, 2]
        else:
            local[:, 2] = self.env.flight_height
        self.env.positions[:self.num_agents] = local
        self.env.velocities[:self.num_agents, :2] = velocities[:, :2]
        if velocities.shape[1] == 3:
            self.env.velocities[:self.num_agents, 2] = velocities[:, 2]
        else:
            self.env.velocities[:self.num_agents, 2] = 0.0
        self.env.pursuer_mask[:self.num_pursuers] = True

        if self._step > 0:
            self.env.advance_lifecycle()
        self.env.episode_step_count = self._step

        # Oracle order: repair/apply membership before the single encode call.
        assignment = self._locked_assignment()
        previous_assignment = (None if self.last_allocation is None
                               else self.last_allocation.get("assignment"))
        switched = bool(previous_assignment is not None
                        and not np.array_equal(previous_assignment, assignment))
        apply_membership(self.env, assignment)
        self.last_allocation = {
            "assignment": assignment.copy(), "reason": "locked",
            "switched": switched,
        }
        snap = self.env.snapshot()
        with torch.no_grad():
            encoded = self.encoder.encode(snap, differentiable=False)
        membership = hard_K_from_assignment(
            assignment, self.env.task_mask, 6, 3,
            encoded["pursuer_embeddings"].device,
        )
        obs = build_v2_obs_torch(encoded, membership, snap, 6, 3)
        mask = torch.as_tensor(self.env.pursuer_mask, dtype=torch.float32,
                               device=obs.device)
        with torch.no_grad():
            actor_actions = self.actor(obs) * mask.unsqueeze(-1)

        live_count = int(np.count_nonzero(self.env.task_mask[:self.num_evaders]))
        if live_count == 1 and self.first_single_task_step is None:
            self.first_single_task_step = int(self._step)
        elif live_count != 1:
            self.first_single_task_step = None
            self.post_capture_ring = None
        ring_active = (self.post_capture_ring_delay is not None
                       and should_use_post_capture_ring(
                           self.env.task_mask[:self.num_evaders],
                           self.first_single_task_step, self._step,
                           self.env.time_step, self.post_capture_ring_delay,
                       ))
        if ring_active:
            if self.post_capture_ring is None:
                self.post_capture_ring = RingGuidanceController(
                    d_cap=self.d_cap, v_max=self.env.v_max,
                    n_p=self.num_pursuers,
                )
                self.post_capture_ring.reset(self.env)
            self.post_capture_ring.assign = assignment.copy()
            ring_actions = np.clip(self.post_capture_ring.act(self.env), -1.0, 1.0)
            elapsed = ((self._step - self.first_single_task_step)
                       * self.env.time_step)
            pursuer_actions = blend_post_capture_ring_actions(
                actor_actions.cpu().numpy()[:self.num_pursuers], ring_actions,
                elapsed_seconds=elapsed,
                delay_seconds=self.post_capture_ring_delay,
                transition_seconds=self.post_capture_ring_transition,
            )
        else:
            pursuer_actions = actor_actions.cpu().numpy()[:self.num_pursuers]

        v_max = float(getattr(self.env, "v_max", self.config.max_pursuer_speed))
        command = np.zeros((self.num_agents, 2), dtype=np.float32)
        command[:self.num_pursuers] = np.clip(pursuer_actions, -1.0, 1.0) * v_max
        dt = float(self.env.time_step)
        command[:self.num_pursuers] = self.transition_smoother.blend(
            command[:self.num_pursuers], assignment,
            self.command_velocity[:self.num_pursuers], dt,
        )
        for j in range(self.num_evaders):
            idx = self.num_pursuers + j
            if self.env.task_mask[j]:
                command[idx] = self._evader_command(j)
            elif self.env.stages[j] == MODE_EXIT:
                command[idx] = self.env._exit_velocity(j)
            else:
                command[idx] = 0.0

        command = filter_velocity(positions, command, origin, self.config)
        command[:self.num_pursuers] = limit_vector_acceleration(
            command[:self.num_pursuers], self.command_velocity[:self.num_pursuers],
            self.max_acceleration, dt,
        )
        self.command_velocity = command.copy()
        self._step += 1

        info = {
            "assignment": assignment.tolist(),
            "demand": None,
            "capacity": None,
            "candidate_count": None,
            "switched": switched,
            "reason": "locked",
            "frozen_tasks": [False] * self.num_evaders,
            "transition_active": bool(
                self.transition_smoother.elapsed is not None
                and np.any(self.transition_smoother.elapsed
                           < self.transition_smoother.duration)),
            "allocated": True,
            "lock_assignment": list(self.lock_assignment),
            "roles": ["core" if int(role) == 0 else "reserve"
                      for role in snap["roles"][:self.num_pursuers]],
            "stages": [MODE_NAMES.get(int(stage), "?")
                       for stage in snap["stages"][:self.num_evaders]],
            "task_masks": snap["task_mask"][:self.num_evaders].tolist(),
            "safe_masks": snap["safe_mask"][:self.num_evaders].tolist(),
            "captured": snap["captured"][:self.num_evaders].tolist(),
            "post_capture_ring_active": bool(ring_active),
            "episode_step": self._step - 1,
        }
        return command, info
