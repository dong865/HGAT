"""Real-flight runtime for the V3 hybrid sparse-demand allocation method.

Loads only the deployable inference stack of a
``v3_sparse_demand_value_hybrid_v1`` checkpoint (ST-HGAT encoder, relation
head, sparse demand head, scale-aware value head, shared actor) through the
strict :func:`train_v3_hybrid.load_v3_hybrid_checkpoint` loader.  Critics,
target networks and optimizers are never constructed.

The per-cycle contract mirrors :class:`runtime_stage.StageCoalitionRuntime`:

* world -> arena conversion ``local = world - origin + L/2``,
* mocap state written into the env (task/safe masks left alone so the
  capture -> exit -> landed lifecycle survives between cycles),
* ``env.advance_lifecycle()`` is the only capture arbiter (same criterion as
  training/evaluation),
* one ``encoder.encode`` per act, allocation throttled by ``assign_interval``
  plus immediate repair when the accepted assignment points at a dead task,
* shared actor output scaled by ``v_max`` for the pursuers, both evader slots
  driven exactly like the stage runtime (potential-field escape / exit
  velocity),
* assignment-change interpolation, ``filter_velocity`` and vector acceleration
  limiting before the command is returned.

Unlike training there is no ``env.step`` in real flight, so ``act`` mirrors
the training-side decision-step counter onto ``env.episode_step_count``:
cycle k corresponds to ``HybridPolicy.decision`` at ``episode_step_count == k``
(encode signature cache, allocation cadence and gate dwell all stay in lock
step with the training loop).
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import SixVTwoConfig
from .stage_env import StageCoalitionEnv, StageRewardConfig, MODE_EXIT, MODE_NAMES
from .stage_networks import StageFeatureEncoder, SharedActor
from .safety import filter_velocity
from .train_v3_hybrid import (load_v3_hybrid_checkpoint,
                              should_use_post_capture_ring,
                              blend_post_capture_ring_actions)
from .traditional_ring_baseline import RingGuidanceController
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler


_CHECKPOINT_DEFAULT = object()
from .v2_controller import v2_obs_dim
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment
from .v3_demand_distillation import ScaleAwareCoalitionValueHead
from .v3_hybrid_controller import V3HybridAllocator
from .v3_sparse_demand import SparseDemandHead


class AssignmentTransitionSmoother:
    """Blend commands only for pursuers whose accepted task changes."""

    def __init__(self, duration: float = 0.4):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be finite and non-negative")
        self.duration = float(duration)
        self.reset()

    def reset(self):
        self.assignment = None
        self.start_commands = None
        self.elapsed = None

    def blend(self, desired: np.ndarray, assignment: np.ndarray,
              previous_command: np.ndarray, dt: float) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float32)
        assignment = np.asarray(assignment, dtype=np.int64)
        previous_command = np.asarray(previous_command, dtype=np.float32)
        if (desired.ndim != 2 or desired.shape[1] != 2
                or assignment.shape != (desired.shape[0],)
                or previous_command.shape != desired.shape
                or not np.isfinite(desired).all()
                or not np.isfinite(previous_command).all()
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("invalid transition smoother input")
        if self.duration <= 0.0:
            return desired.copy()
        if self.assignment is None or self.assignment.shape != assignment.shape:
            self.assignment = assignment.copy()
            self.start_commands = previous_command.copy()
            self.elapsed = np.full(desired.shape[0], self.duration, dtype=np.float32)
            return desired.copy()

        changed = assignment != self.assignment
        if np.any(changed):
            self.start_commands[changed] = previous_command[changed]
            self.elapsed[changed] = 0.0
            self.assignment = assignment.copy()
        active = self.elapsed < self.duration
        self.elapsed[active] = np.minimum(self.duration, self.elapsed[active] + float(dt))
        alpha = np.clip(self.elapsed / self.duration, 0.0, 1.0)[:, None]
        blended = desired.copy()
        blended[active] = (
            (1.0 - alpha[active]) * self.start_commands[active]
            + alpha[active] * desired[active]
        )
        return blended


def limit_vector_acceleration(command: np.ndarray, previous: np.ndarray,
                              max_acceleration: float, dt: float) -> np.ndarray:
    """Limit the Euclidean velocity change per command row."""
    command = np.asarray(command, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    if (command.shape != previous.shape or command.ndim != 2 or command.shape[1] != 2
            or not np.isfinite(command).all() or not np.isfinite(previous).all()
            or not np.isfinite(max_acceleration) or max_acceleration <= 0
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("invalid acceleration-limit input")
    delta = command - previous
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    max_delta = float(max_acceleration) * float(dt)
    scale = np.minimum(1.0, max_delta / np.maximum(norms, 1e-12))
    return previous + delta * scale


class V3HybridRuntime:
    """Checkpoint -> planar velocity set-points for the V3 hybrid method."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[SixVTwoConfig] = None,
        device: str = "cpu",
        obstacles: Iterable[Tuple[float, float, float]] = (),
        scenario_seed: Optional[int] = None,
        demand_weight: float = 0.1,
        switch_weight: float = 0.5,
        evader_mode: str = "escape",
        transition_duration: float = 0.4,
        max_acceleration: float = 1.5,
        freeze_distance: Optional[float] = None,
        freeze_release_ratio: Optional[float] = None,
        mode_demand_checkpoint: Optional[str] = None,
        mode_prior_weight: float = 0.0,
        infeasible_mode_penalty: Optional[float] = None,
        infeasible_mode_warmup_fraction: Optional[float] = None,
        capture_urgency_distance: Optional[float] = None,
        post_capture_ring_delay=_CHECKPOINT_DEFAULT,
        post_capture_ring_transition=_CHECKPOINT_DEFAULT,
    ):
        self.config = config or SixVTwoConfig(num_pursuers=4, num_evaders=2)
        self.num_pursuers = self.config.num_pursuers
        self.num_evaders = self.config.num_evaders
        self.num_agents = self.num_pursuers + self.num_evaders
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.scenario_seed = None if scenario_seed is None else int(scenario_seed)
        self.evader_mode = evader_mode
        if not np.isfinite(max_acceleration) or max_acceleration <= 0:
            raise ValueError("max_acceleration must be finite and positive")
        self.max_acceleration = float(max_acceleration)
        self.transition_smoother = AssignmentTransitionSmoother(transition_duration)

        path = Path(checkpoint_path).expanduser().resolve()
        # Phase 1: strict schema validation only (also gives us the saved config).
        payload = load_v3_hybrid_checkpoint(path)
        ckpt = payload["config"]
        if post_capture_ring_delay is _CHECKPOINT_DEFAULT:
            post_capture_ring_delay = ckpt.get("post_capture_ring_delay")
        if post_capture_ring_transition is _CHECKPOINT_DEFAULT:
            post_capture_ring_transition = ckpt.get(
                "post_capture_ring_transition", 0.0)
        if (post_capture_ring_delay is not None
                and (not np.isfinite(post_capture_ring_delay)
                     or post_capture_ring_delay < 0)):
            raise ValueError("post_capture_ring_delay must be nonnegative or None")
        self.post_capture_ring_delay = (None if post_capture_ring_delay is None
                                        else float(post_capture_ring_delay))
        if (not np.isfinite(post_capture_ring_transition)
                or post_capture_ring_transition < 0):
            raise ValueError("post_capture_ring_transition must be nonnegative")
        self.post_capture_ring_transition = float(post_capture_ring_transition)
        self.map_size = float(ckpt.get("map_size", 8.0))
        self.d_cap = float(ckpt.get("d_cap", 2.0))
        self.hold_steps = int(ckpt.get("hold_steps", 5))
        hidden = int(ckpt["hidden_dim"])
        future_steps = int(ckpt["future_steps"])
        print(f"[runtime] V3 hybrid checkpoint episode={payload.get('episode')} "
              f"map={self.map_size} d_cap={self.d_cap} hold_steps={self.hold_steps}",
              flush=True)

        # The checkpoint trains at up to 6 pursuers / 3 evaders, so keep the
        # underlying buffers at that maximum scale and only reset to the active
        # team size (same contract as StageCoalitionRuntime).
        self.env = StageCoalitionEnv(
            length=self.map_size,
            num_obstacle=int(ckpt["num_obstacles"]),
            max_pursuers=6,
            max_evaders=3,
            d_cap=self.d_cap,
            hold_steps=self.hold_steps,
            seed=int(ckpt.get("seed", 0)),
            reward_config=StageRewardConfig(**ckpt["reward_config"]),
        )
        if ckpt.get("scenario_protocol", "stage") == "deploy5x5":
            self.env.scenario_sampler = Deploy5x5Sampler()
            self.env.v_max_e = 0.35
        self.env.reset(num_pursuers=self.num_pursuers, num_evaders=self.num_evaders,
                       scenario_seed=self.scenario_seed)
        self._set_obstacles(obstacles)

        self.encoder = StageFeatureEncoder(
            6, 3, int(ckpt["num_obstacles"]), self.map_size, hidden,
            future_steps, 10, 9, self.device,
            use_role_embedding=False, use_stage_embedding=False,
            use_graph_embedding=True, use_ring_slot_features=False,
        ).to(self.device)
        self.encoder._skip_legacy_plan = True
        self.demand_head = SparseDemandHead(hidden, width=int(ckpt["head_width"])).to(self.device)
        self.value_head = ScaleAwareCoalitionValueHead(hidden, width=int(ckpt["head_width"])).to(self.device)
        self.actor = SharedActor(
            v2_obs_dim(6, hidden, future_steps), hidden_dim=int(ckpt["actor_hidden"])
        ).to(self.device)
        # Phase 2: load the five deployable networks through the strict loader.
        load_v3_hybrid_checkpoint(
            path,
            modules={"encoder": self.encoder, "demand_head": self.demand_head,
                     "value_head": self.value_head, "actor": self.actor},
        )
        self.mode_demand_checkpoint = None
        if mode_demand_checkpoint is not None:
            mode_path = Path(mode_demand_checkpoint).expanduser().resolve()
            mode_payload = torch.load(
                mode_path, map_location=self.device, weights_only=False,
            )
            if mode_payload.get("architecture_version") != (
                    "v3_counterfactual_mode_demand_4v2_v1"):
                raise ValueError("external mode demand checkpoint has incompatible schema")
            if Path(mode_payload.get("base_checkpoint", "")).resolve() != path:
                raise ValueError("external mode demand was trained from a different base checkpoint")
            if (int(mode_payload.get("hidden_dim", -1)) != hidden
                    or int(mode_payload.get("head_width", -1)) != int(ckpt["head_width"])):
                raise ValueError("external mode demand dimensions do not match runtime")
            self.demand_head.load_state_dict(mode_payload["demand_head"])
            self.mode_demand_checkpoint = mode_path
        self.encoder.eval()
        self.demand_head.eval()
        self.value_head.eval()
        self.actor.eval()
        self.loaded_modules = ("st_hgat", "relation_head", "demand_head",
                               "value_head", "actor")

        allocation_config = ckpt["allocation_config"]
        self.max_steps = int(ckpt.get("max_steps", 2000))
        self.capture_urgency_distance = float(
            ckpt.get("capture_urgency_distance",
                     allocation_config.get("capture_urgency_distance", 1.5))
            if capture_urgency_distance is None else capture_urgency_distance
        )
        self.assign_interval = int(allocation_config["assign_interval"])
        if freeze_distance is None:
            freeze_distance = float(ckpt.get("freeze_distance_ratio", 0.8)) * self.d_cap
        if freeze_release_ratio is None:
            freeze_release_ratio = float(ckpt.get("freeze_release_ratio", 1.15))
        self.allocator = V3HybridAllocator(
            self.demand_head, self.value_head, self.map_size,
            capture_size=int(ckpt.get("capture_size", 3)),
            mode_limit=int(ckpt["max_capacity_modes"]),
            max_candidates=int(ckpt["max_alliance_candidates"]),
            demand_weight=float(demand_weight), switch_weight=float(switch_weight),
            freeze_distance=float(freeze_distance),
            freeze_release_ratio=float(freeze_release_ratio),
            mode_prior_weight=float(mode_prior_weight),
            infeasible_mode_penalty=float(
                allocation_config.get("infeasible_mode_penalty", 0.0)
                if infeasible_mode_penalty is None else infeasible_mode_penalty
            ),
            infeasible_mode_warmup_fraction=float(
                allocation_config.get("infeasible_mode_warmup_fraction", 0.5)
                if infeasible_mode_warmup_fraction is None
                else infeasible_mode_warmup_fraction
            ),
            **ckpt["gate_config"],
        )
        self.command_velocity = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_allocation = None
        self.first_single_task_step = None
        self.post_capture_ring = None
        self._step = 0
        self.reset()

    # ------------------------------------------------------------------ setup
    def _set_obstacles(self, obstacles: Iterable[Tuple[float, float, float]]) -> None:
        values = list(obstacles)
        if len(values) > self.config.num_obstacles:
            raise ValueError("too many obstacles for the trained graph")
        self._fixed_obstacles = values  # keep so reset() re-applies them
        self._apply_obstacles(values)

    def _apply_obstacles(self, values: List[Tuple[float, float, float]]) -> None:
        self.env.obstacles = []
        for x, y, radius in values:
            self.env.obstacles.append(
                type("O", (), {
                    "position": np.array([x, y], dtype=np.float64),
                    "radius": float(radius),
                    "height": 3.0,  # cylinder covers the flight band (obs_height)
                })()
            )

    def reset(self) -> None:
        self.encoder.reset(step=0)
        self.allocator.reset()
        self.last_allocation = None
        self.first_single_task_step = None
        self.post_capture_ring = None
        self.command_velocity.fill(0.0)
        self.transition_smoother.reset()
        # 固定场景种子时每次 reset 回到同一布局（逃逸机目标点/障碍），与 stage runtime 一致。
        self.env.reset(num_pursuers=self.num_pursuers, num_evaders=self.num_evaders,
                       scenario_seed=self.scenario_seed)
        # env.reset() regenerates random obstacles; re-apply the user/virtual set
        if getattr(self, "_fixed_obstacles", None):
            self._apply_obstacles(self._fixed_obstacles)
        self._step = 0

    # ------------------------------------------------------------------ act
    def act(
        self,
        positions_world: np.ndarray,
        velocities_world: np.ndarray,
        arena_origin_world: Sequence[float] = (0.0, 0.0),
    ) -> Tuple[np.ndarray, dict]:
        positions = np.asarray(positions_world, dtype=np.float32)
        velocities = np.asarray(velocities_world, dtype=np.float32)
        valid_shapes = {(self.num_agents, 2), (self.num_agents, 3)}
        if positions.shape not in valid_shapes or velocities.shape not in valid_shapes:
            raise ValueError(
                f"positions and velocities must both have shape "
                f"({self.num_agents}, 2) or ({self.num_agents}, 3)"
            )
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("non-finite mocap state")

        origin = np.asarray(arena_origin_world, dtype=np.float32)
        # 3-D support: xy in the arena frame, z from the mocap (or flight height)
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
        # task_mask/safe_mask must NOT be forced True here -- they encode the
        # capture->EXIT->LANDED lifecycle (same note as StageCoalitionRuntime).

        # The reset state is already the training decision state for step 0.
        # From the second mocap frame onward, advance once for the physical
        # transition that occurred since the preceding decision.
        if self._step > 0:
            self.env.advance_lifecycle()
        # 所有任务都已被捕获：不再分配/跑 actor，返回零速度让上层执行
        # 捕获停顿/原地下降/最后统一降落（否则 allocate 会因无 live task 报错）。
        if not bool(np.any(self.env.task_mask)):
            self.command_velocity[:] = 0.0
            return self.command_velocity.copy(), {
                "assignment": [-1] * self.num_pursuers,
                "stages": [int(stage) for stage in self.env.stages[: self.num_evaders]],
                "captured": [bool(value) for value in self.env.captured[: self.num_evaders]],
                "all_captured": True,
                "allocated": False,
            }
        # 2) mirror the training-side decision-step counter: cycle k matches
        #    HybridPolicy.decision at env.episode_step_count == k.
        self.env.episode_step_count = self._step
        snap = self.env.snapshot()

        # 3) encode once, then V3 sparse-demand allocation (throttled + repair)
        with torch.no_grad():
            encoded = self.encoder.encode(snap, differentiable=False)
        previous = self.allocator.gate.assignment
        invalid = previous is None or any(not self.env.task_mask[j] for j in previous)
        allocate = invalid or self._step % self.assign_interval == 0
        if allocate:
            remaining_fraction = max(0.0, 1.0 - self._step / self.max_steps)
            live_ids = np.flatnonzero(self.env.task_mask)
            capture_urgent = bool(live_ids.size and np.any(np.linalg.norm(
                self.env.positions[self.num_pursuers + live_ids, :2]
                - self.env.evader_goals[live_ids, :2], axis=1,
            ) <= self.capture_urgency_distance))
            self.last_allocation = self.allocator.allocate(
                encoded, snap, self.env.task_mask, self._step, force=invalid,
                hold_counters=self.env.hold_counters,
                remaining_fraction=remaining_fraction,
                capture_urgent=capture_urgent,
            )
            apply_membership(self.env, self.last_allocation["assignment"])

        snap = self.env.snapshot()
        assignment = self.last_allocation["assignment"].copy()
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
            ring_actions = np.clip(
                self.post_capture_ring.act(self.env), -1.0, 1.0,
            )
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
        command[:self.num_pursuers] = (
            np.clip(pursuer_actions, -1.0, 1.0) * v_max
        )
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

        # 4) safety + continuity projection (identical to the stage runtime)
        command = filter_velocity(positions, command, origin, self.config)
        command[:self.num_pursuers] = limit_vector_acceleration(
            command[:self.num_pursuers],
            self.command_velocity[:self.num_pursuers],
            self.max_acceleration, dt,
        )
        self.command_velocity = command.copy()
        self._step += 1

        allocation = self.last_allocation or {}
        demand = allocation.get("demand")
        capacity = allocation.get("capacity")
        info = {
            "assignment": assignment.tolist(),
            "demand": (demand.detach().cpu().numpy().tolist()
                       if torch.is_tensor(demand) else None),
            "capacity": (capacity.detach().cpu().numpy().tolist()
                         if torch.is_tensor(capacity) else None),
            "candidate_count": allocation.get("candidate_count"),
            "switched": bool(allocation.get("switched", False)),
            "reason": allocation.get("reason"),
            "frozen_tasks": np.asarray(
                allocation.get("frozen_tasks", np.zeros(self.num_evaders, dtype=bool))
            )[:self.num_evaders].tolist(),
            "transition_active": bool(
                self.transition_smoother.elapsed is not None
                and np.any(self.transition_smoother.elapsed < self.transition_smoother.duration)
            ),
            "allocated": bool(allocate),
            "roles": ["core" if int(r) == 0 else "reserve" for r in snap["roles"][:self.num_pursuers]],
            "stages": [MODE_NAMES.get(int(s), "?") for s in snap["stages"][:self.num_evaders]],
            "task_masks": snap["task_mask"][:self.num_evaders].tolist(),
            "safe_masks": snap["safe_mask"][:self.num_evaders].tolist(),
            "captured": snap["captured"][:self.num_evaders].tolist(),
            "post_capture_ring_active": bool(ring_active),
            "episode_step": self._step - 1,
        }
        return command, info

    def _evader_command(self, j: int) -> np.ndarray:
        if self.evader_mode == "stationary":
            return np.zeros(2, dtype=np.float32)
        # env._evader_velocity is already in m/s (bounded by v_max_e); pass it
        # through unmodified (same rule as StageCoalitionRuntime).
        return self.env._evader_velocity(j).astype(np.float32)
