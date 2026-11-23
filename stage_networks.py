"""Scale-adaptive Actor-Critic and ST-HGAT feature encoder for the paper method.

Implements the learning side of the revised HGAT route:

* all pursuers share one Actor:   a_i = pi_theta(h_i, h_j, h_G, e(m_j))
                                   -> normalized planar velocity [vx, vy]
* set-invariant graph Critic:     Q(G,A) = rho( sum_i mask_i psi(h_i, a_i, e(m)) )
* deterministic constant-velocity lookahead for geometric allocation features
* fixed-dim padded per-pursuer observations so that Np / Ne can change
  without changing network parameter dimensions
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .st_hgat import ST_HGAT, GraphBuilder
from .stage_coalitions import MaskedSinkhorn
from .stage_env import MODE_TRK, ROLE_CORE, ROLE_RES, ROLE_SUP

MODE_DIM = 5  # trk / blk / cap / exit / landed
ROLE_DIM = 3  # core / support / reserve


def unique_ring_slot_features(
    pursuer_positions: np.ndarray,
    evader_positions: np.ndarray,
    predictions: np.ndarray,
    assignments: np.ndarray,
    roles: np.ndarray,
    task_mask: np.ndarray,
    d_cap: float,
    map_size: float,
    evader_velocities: Optional[np.ndarray] = None,
    lookahead: float = 0.4,
    slot_state: Optional[Dict[int, Dict[int, int]]] = None,
    phase_hysteresis: float = 0.25,
    normalizer: float = 4.0,
) -> np.ndarray:
    """Return per-pursuer ``[dx/L, dy/L, valid]`` ring-slot features.

    This is a geometric interface to the learned actor, not a controller.
    Active core/support members receive distinct predicted-target ring slots.
    Bearing-order matching is equivariant to pursuer indexing and avoids slot
    crossings when a coalition is rebuilt.
    """
    pursuer_positions = np.asarray(pursuer_positions)
    evader_positions = np.asarray(evader_positions)
    predictions = np.asarray(predictions)
    assignments = np.asarray(assignments)
    roles = np.asarray(roles)
    task_mask = np.asarray(task_mask, dtype=bool)
    features = np.zeros((len(pursuer_positions), 3), dtype=np.float32)
    del map_size  # retained in the public signature for checkpoint/API stability
    radius = min(0.8 * float(d_cap), float(d_cap) - 0.08)
    for j in np.flatnonzero(task_mask):
        members = np.flatnonzero(
            (assignments == j) & np.isin(roles, (ROLE_CORE, ROLE_SUP))
        )
        if members.size == 0:
            continue
        current_center = evader_positions[j, :2]
        predicted_center = current_center
        if evader_velocities is not None and j < len(evader_velocities):
            predicted_center = (
                current_center
                + float(lookahead) * np.asarray(evader_velocities)[j, :2]
            )
        elif predictions.ndim == 3 and j < predictions.shape[0]:
            candidate = predictions[j, -1, :2]
            if np.all(np.isfinite(candidate)) and not np.allclose(candidate, 0.0):
                predicted_center = candidate
        heading = predicted_center - current_center
        heading_angle = (
            0.0 if float(np.linalg.norm(heading)) < 1e-6
            else float(np.arctan2(heading[1], heading[0]))
        )
        bearings = np.arctan2(
            pursuer_positions[members, 1] - predicted_center[1],
            pursuer_positions[members, 0] - predicted_center[0],
        )
        ordered_members = members[np.argsort(bearings, kind="stable")]
        ideal = heading_angle + np.pi + np.arange(members.size) * (
            2.0 * np.pi / members.size
        )
        slots = predicted_center + radius * np.stack(
            [np.cos(ideal), np.sin(ideal)], axis=1,
        )
        # Angular sorting fixes the cyclic order but not its phase.  Select the
        # cyclic phase with the least total travel so the -pi/pi branch cut does
        # not make every coalition member exchange slots at once.
        slot_ids = np.arange(members.size, dtype=np.int64)
        rotations = [np.roll(slot_ids, shift) for shift in range(members.size)]
        costs = [
            float(np.sum((slots[candidate] - pursuer_positions[ordered_members, :2]) ** 2))
            for candidate in rotations
        ]
        chosen_ids = rotations[int(np.argmin(costs))]
        if slot_state is not None:
            previous = slot_state.get(int(j), {})
            previous_valid = (
                set(previous) == set(map(int, members))
                and len(set(previous.values())) == members.size
                and all(0 <= int(v) < members.size for v in previous.values())
            )
            if previous_valid:
                previous_ids = np.asarray(
                    [previous[int(member)] for member in ordered_members], dtype=np.int64,
                )
                previous_cost = float(np.sum(
                    (slots[previous_ids] - pursuer_positions[ordered_members, :2]) ** 2
                ))
                best_cost = min(costs)
                required_improvement = float(phase_hysteresis) * members.size
                if previous_cost - best_cost <= required_improvement:
                    chosen_ids = previous_ids
            slot_state[int(j)] = {
                int(member): int(slot_id)
                for member, slot_id in zip(ordered_members, chosen_ids)
            }
        assigned_slots = slots[chosen_ids]
        for member, slot in zip(ordered_members, assigned_slots):
            features[member, :2] = (
                slot - pursuer_positions[member, :2]
            ) / float(normalizer)
            features[member, 2] = 1.0
    return features


class RelationCostHead(nn.Module):
    """Small graph-relation term f_omega(h_i, h_j) added to the edge cost."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        h_i, h_j = torch.broadcast_tensors(h_i, h_j)
        return 0.3 * torch.tanh(self.mlp(torch.cat([h_i, h_j, (h_i - h_j).abs()], dim=-1)).squeeze(-1))


class DemandHead(nn.Module):
    """Predict a differentiable expected core demand in the closed interval [1, 3]."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer("levels", torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, task_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(task_embeddings)
        demand = torch.softmax(logits, dim=-1) @ self.levels.to(logits)
        return logits, demand


def soft_coalition_context(
    pursuer_embeddings: torch.Tensor,
    task_embeddings: torch.Tensor,
    task_plan: torch.Tensor,
    pursuer_mask: torch.Tensor,
    task_mask: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable task and same-coalition member context per pursuer."""
    p_mask = pursuer_mask.to(task_plan).reshape(-1, 1)
    t_mask = task_mask.to(task_plan).reshape(1, -1)
    plan = task_plan * p_mask * t_mask
    task_context = plan @ task_embeddings
    affinity = plan @ plan.transpose(0, 1)
    affinity = affinity * p_mask * p_mask.transpose(0, 1)
    member_context = (affinity @ pursuer_embeddings) / affinity.sum(
        dim=-1, keepdim=True,
    ).clamp_min(1e-6)
    return torch.cat([task_context, member_context], dim=-1) * p_mask


def differentiable_task_plan(
    relation_costs: torch.Tensor,
    demand_hat: torch.Tensor,
    pursuer_mask: torch.Tensor,
    task_mask: torch.Tensor,
    sinkhorn: MaskedSinkhorn,
) -> torch.Tensor:
    """Build a soft member-task plan while routing unused mass to reserve.

    ``relation_costs`` are real costs: smaller values mean better matches.
    Sinkhorn minimizes total cost, so the plan puts more mass on low-cost
    edges. This matches the auxiliary supervision that trains the relation
    head to predict future-interception distance.
    """
    p_active = pursuer_mask.to(relation_costs).reshape(-1) > 0
    t_active = task_mask.to(relation_costs).reshape(-1) > 0
    result = relation_costs.new_zeros(relation_costs.shape)
    if not bool(p_active.any()) or not bool(t_active.any()):
        return result
    costs = relation_costs[p_active][:, t_active]
    demand = demand_hat[t_active].clamp(1.0, 3.0)
    supply_total = p_active.sum().to(relation_costs)
    scale = torch.clamp(supply_total / demand.sum().clamp_min(1e-6), max=1.0)
    task_demand = demand * scale
    reserve_demand = (supply_total - task_demand.sum()).clamp_min(0.0)
    if bool((reserve_demand > 1e-6).detach()):
        reserve_cost = costs.new_zeros(costs.shape[0], 1)
        extended_costs = torch.cat([costs, reserve_cost], dim=1)
        col_demand = torch.cat([task_demand, reserve_demand.reshape(1)])
    else:
        # Exact mass balance is required by Sinkhorn; distribute numerical
        # residue over task columns without detaching predicted proportions.
        col_demand = task_demand * (supply_total / task_demand.sum().clamp_min(1e-6))
        extended_costs = costs
    plan = sinkhorn(
        extended_costs,
        torch.ones(costs.shape[0], dtype=costs.dtype, device=costs.device),
        col_demand,
        torch.ones_like(extended_costs, dtype=torch.bool),
    )[:, : costs.shape[1]]
    result[p_active.unsqueeze(1) & t_active.unsqueeze(0)] = plan.reshape(-1)
    return result


class SharedTemporalNodeEncoder(nn.Module):
    """Per-node temporal encoder with no neighbour/message dependence."""

    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.obs_embed = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.recurrent = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.device = torch.device("cpu")

    def step(
        self,
        obs: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        adj_dict: Dict[str, torch.Tensor],
        dist_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del adj_dict, dist_dict
        obs = obs.to(self.device)
        if h_prev is None:
            h_prev = torch.zeros(
                obs.size(0), self.hidden_dim, dtype=obs.dtype, device=self.device,
            )
        else:
            h_prev = h_prev.to(self.device)
        return self.recurrent(self.obs_embed(obs), h_prev)


class StageFeatureEncoder(nn.Module):
    """Build padded per-pursuer observations and deterministic lookaheads.

    Observation layout (obs_dim):
        self 8 + teammates 2*(max_pursuers-1) + target 2 + stage one-hot 5
        + coalition size 1 + active mask 1 + nearest obstacles 4
        + graph 3*hidden (h_i, h_target, coalition pool) + CV lookahead 2*H
    """

    def __init__(
        self,
        max_pursuers: int = 6,
        max_evaders: int = 3,
        num_obstacles: int = 10,
        map_size: float = 8.0,
        hidden_dim: int = 32,
        future_steps: int = 10,
        seq_len: int = 10,
        node_dim: int = 9,
        device: Optional[torch.device] = None,
        use_role_embedding: bool = True,
        use_stage_embedding: bool = True,
        use_graph_embedding: bool = True,
        prediction_mode: str = "constant_velocity",
        perception_radius: Optional[float] = None,
        node_encoder: str = "st_hgat",
        coalition_pool: str = "attention",
        use_ring_slot_features: bool = False,
        ring_slot_normalizer: float = 4.0,
        ring_slot_phase_hysteresis: float = 0.25,
        output_layer_norm: bool = False,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.max_pursuers = max_pursuers
        self.max_evaders = max_evaders
        self.num_obstacles = num_obstacles
        self.map_size = float(map_size)
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.seq_len = seq_len
        self.device = device or torch.device("cpu")
        self.use_role_embedding = bool(use_role_embedding)
        self.use_stage_embedding = bool(use_stage_embedding)
        self.use_graph_embedding = bool(use_graph_embedding)
        # Kept only to load legacy checkpoints/commands.  SDGCR no longer has
        # a learned trajectory-prediction module; all lookaheads are analytic.
        del prediction_mode
        if node_encoder not in {"st_hgat", "shared_mlp"}:
            raise ValueError(f"invalid node_encoder {node_encoder!r}")
        if coalition_pool not in {"attention", "mean", "fixed_concat", "none"}:
            raise ValueError(f"invalid coalition_pool {coalition_pool!r}")
        self.node_encoder_name = node_encoder
        self.coalition_pool_mode = coalition_pool
        self.use_ring_slot_features = bool(use_ring_slot_features)
        self.ring_slot_normalizer = float(ring_slot_normalizer)
        self.ring_slot_phase_hysteresis = float(ring_slot_phase_hysteresis)
        if self.ring_slot_normalizer <= 0.0:
            raise ValueError("ring_slot_normalizer must be positive")
        # None = full observability (default); finite value gates each pursuer's
        # graph neighbours and observation features to a perception disk.
        self.perception_radius = None if perception_radius is None else float(perception_radius)

        if node_encoder == "st_hgat":
            self.st_hgat = ST_HGAT(
                node_dim=node_dim,
                hidden_dim=hidden_dim,
                edge_types=["adversarial", "cooperative", "obstacle"],
                num_layers=1,
                num_heads=1,
                dropout=0.1,
                map_size=self.map_size,
            )
        else:
            self.st_hgat = SharedTemporalNodeEncoder(node_dim, hidden_dim)
        self.st_hgat.device = self.device
        self.relation_head = RelationCostHead(hidden_dim)
        self.demand_head = DemandHead(hidden_dim)
        self.soft_sinkhorn = MaskedSinkhorn(epsilon=0.5, max_iters=80)
        # V2 路径不消费 encoder 内部的旧三分类需求/软计划（V2 用独立
        # ContinuousDemandHead + continuous_membership.build_soft_plan），
        # 跳过可避免 legacy Sinkhorn 的无谓开销与崩溃面。relation_costs 仍保留。
        self._skip_legacy_plan = False
        self.fixed_concat_projector = None
        if coalition_pool == "fixed_concat":
            # Deterministic ID-sensitive projection. Replay stores encoded
            # observations and the encoder is frozen by default, so a newly
            # trainable projector would otherwise remain random and make A3 an
            # invalid ablation. Each identity slot receives a distinct weight.
            projector = nn.Linear(
                max_pursuers * (hidden_dim + 1), hidden_dim, bias=False,
            )
            with torch.no_grad():
                projector.weight.zero_()
                for slot in range(max_pursuers):
                    coefficient = float(slot + 1) / float(max_pursuers)
                    offset = slot * (hidden_dim + 1)
                    for feature in range(hidden_dim):
                        projector.weight[feature, offset + feature] = coefficient
            projector.requires_grad_(False)
            self.fixed_concat_projector = projector

        self._h_prev: Optional[torch.Tensor] = None
        self._graph_signature: Optional[Tuple[int, int, int]] = None
        self._last_graph_step: Optional[int] = None
        self._last_h_active: Optional[torch.Tensor] = None
        self._pos_histories: List[deque] = [deque(maxlen=seq_len + future_steps + 1) for _ in range(max_evaders)]
        self._snapshots: List[deque] = [deque(maxlen=4096) for _ in range(max_evaders)]
        self._step = 0
        self._ring_slot_state: Dict[int, Dict[int, int]] = {}
        # Optional output normalisation. Unregularised, the jointly trained
        # encoder's output scale drifts upward (+40% over 9.5k episodes), which
        # pushes the actor into tanh saturation and degrades capture success.
        # Normalising the encoding pins its scale. Off by default so existing
        # checkpoints and behaviour are unchanged.
        self.output_layer_norm = bool(output_layer_norm)
        # LayerNorm alone only PINS the scale at unit variance (mean |x| ~ 0.8),
        # which is HIGHER than the drifted value it was meant to fix (~0.77). The
        # explicit gain is what actually lowers the scale: norm -> unit variance,
        # then * gain, so gain=0.3 lands the encoding near 0.25.
        self.output_gain = float(output_gain)
        if self.output_layer_norm:
            self.output_norm = nn.LayerNorm(self.obs_dim)

        self.to(self.device)

    # ---------------------------------------------------------------- helpers
    @property
    def obs_dim(self) -> int:
        return (
            10  # self: pos(2) + vel(2) + boundary(4) + z + vz
            + 3 * (self.max_pursuers - 1)  # teammate: pos(2) + z
            + 3  # target: pos(2) + z
            + MODE_DIM
            + ROLE_DIM
            + 1
            + 1
            + 4
            + 3 * self.hidden_dim
            + 2 * self.future_steps
            + 1
            + (3 if self.use_ring_slot_features else 0)
        )

    def reset(self, step: int = 0) -> None:
        self._h_prev = None
        self._graph_signature = None
        self._last_graph_step = None
        self._last_h_active = None
        self._ring_slot_state.clear()
        self._step = step
        for j in range(self.max_evaders):
            self._pos_histories[j].clear()
            self._snapshots[j].clear()

    # ---------------------------------------------------------------- encoding
    def encode(self, snap: Dict, differentiable: bool = False) -> Dict:
        """Encode one physical snapshot.

        ``differentiable=False`` preserves the production replay contract: the
        returned observation is detached.  The experimental end-to-end trainer
        opts into ``differentiable=True`` so graph features remain connected to
        ST-HGAT while static geometry stays constant.
        """
        n_p = int(snap["num_pursuers"])
        n_ev = int(snap["num_evaders"])
        positions = np.asarray(snap["positions"])
        velocities = np.asarray(snap["velocities"])
        pursuer_mask = np.asarray(snap["pursuer_mask"])[:n_p]
        task_mask = np.asarray(snap["task_mask"])[:n_ev]
        stages = np.asarray(snap["stages"])[:n_ev]
        captured = np.asarray(snap["captured"])[:n_ev]
        assignment = np.asarray(snap["assignment"])[:n_p]
        roles = np.asarray(snap.get("roles", np.full(n_p, ROLE_CORE)))[:n_p]

        # ---- build active graph
        active_p = [i for i in range(n_p) if pursuer_mask[i]]
        active_e = [j for j in range(n_ev) if task_mask[j]]
        n_pa, n_ea = len(active_p), len(active_e)
        obstacles = snap["obstacles"]
        no = len(obstacles)
        pos_active = np.concatenate(
            [
                positions[active_p],
                positions[[n_p + j for j in active_e]] if active_e else np.zeros((0, 3)),
                np.stack([np.r_[o[0], 0.0] for o in obstacles]) if no else np.zeros((0, 3)),
            ],
            axis=0,
        ).astype(np.float32)
        vel_active = np.concatenate(
            [
                velocities[active_p],
                velocities[[n_p + j for j in active_e]] if active_e else np.zeros((0, 3)),
                np.zeros((no, 3), dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        builder = GraphBuilder(
            num_pursuers=n_pa,
            num_evaders=n_ea,
            num_obstacles=no,
            map_size=self.map_size,
        )
        adj_dict, dist_dict, node_features = builder.build_graph(pos_active, vel_active)

        # ---- partial observability: restrict each pursuer's graph neighbours
        # ---- to its perception disk (obs content changes with visibility).
        # graph node layout: [0:n_pa) = active pursuers (active_p order),
        # [n_pa:n_pa+n_ea) = active evaders (active_e order), rest = obstacles.
        if self.perception_radius is not None:
            R = self.perception_radius
            coop = adj_dict["cooperative"].numpy()
            adv = adj_dict["adversarial"].numpy()
            for a, gi in enumerate(active_p):
                p_pos = positions[gi]
                for c in range(n_pa):
                    if c == a:
                        continue
                    gc = active_p[c]
                    if np.linalg.norm(p_pos - positions[gc]) > R:
                        coop[a, c] = 0.0
                for b, j in enumerate(active_e):
                    ge = n_p + j
                    if np.linalg.norm(p_pos - positions[ge]) > R:
                        adv[a, n_pa + b] = 0.0
                        adv[n_pa + b, a] = 0.0  # pursuer is the sensor for both directions
            adj_dict["cooperative"] = torch.FloatTensor(coop)
            adj_dict["adversarial"] = torch.FloatTensor(adv)
        node_features = node_features.to(self.device)

        # graph signature must encode the *identity* of the active entities, not
        # just their count: if one pursuer drops offline while another joins and
        # the counts stay equal, the recurrent hidden state would otherwise be
        # assigned to the wrong entities (P1: dynamic online/offline support)
        signature = (tuple(active_p), tuple(active_e), no)
        snap_step = int(snap.get("episode_step", self._step))
        same_physics_step = (
            snap_step == self._last_graph_step
            and signature == self._graph_signature
            and self._last_h_active is not None
        )
        if same_physics_step:
            h_active = self._last_h_active.to(self.device)
            h_prev_input = self._h_prev.detach().clone()
        else:
            if signature != self._graph_signature:
                self._h_prev = None
                self._graph_signature = signature
            if self._h_prev is None or self._h_prev.size(0) != node_features.size(0):
                self._h_prev = torch.zeros(node_features.size(0), self.hidden_dim, device=self.device)
            h_prev_input = self._h_prev.detach().clone()
            with torch.set_grad_enabled(self.training):
                h_active = self.st_hgat.step(node_features, self._h_prev, adj_dict, dist_dict)
            self._h_prev = h_active.detach()
            self._last_h_active = h_active.detach()
            self._last_graph_step = snap_step
        is_new_step = not same_physics_step

        h_pursuer = h_active[:n_pa]
        h_evader = h_active[n_pa : n_pa + n_ea] if n_ea else torch.zeros(0, self.hidden_dim, device=self.device)

        # ---- deterministic lookahead (no learned trajectory predictor)
        predictions = np.zeros((self.max_evaders, self.future_steps, 2), dtype=np.float32)
        uncertainty = np.zeros(self.max_evaders, dtype=np.float32)
        target_h = torch.zeros(self.max_evaders, self.hidden_dim, device=self.device)
        for a, j in enumerate(active_e):
            emb = h_evader[a] if differentiable else h_evader[a].detach()
            pos = positions[n_p + j, :2]
            vel = velocities[n_p + j, :2]
            for k in range(self.future_steps):
                predictions[j, k] = np.clip(
                    pos + (k + 1) * (1.0 / 50.0) * vel, 0.0, self.map_size,
                ).astype(np.float32)
            if is_new_step:
                self._snapshots[j].append(
                    {
                        "step": snap_step,
                        "j": j,
                        "node_features": node_features.detach().cpu().clone(),
                        "h_prev": h_prev_input.detach().cpu().clone(),
                        "adj_dict": {k: v.detach().cpu().clone() for k, v in adj_dict.items()},
                        "dist_dict": {k: v.detach().cpu().clone() for k, v in dist_dict.items()},
                        "n_pa": n_pa,
                        "evader_offset": a,
                        "pursuer_xy": positions[active_p, :2].astype(np.float32) / self.map_size,
                    }
                )
            target_h[j] = emb
        # Evader positions advance once per physical step, even if roles are re-encoded.
        if is_new_step:
            for j in range(n_ev):
                self._pos_histories[j].append((snap_step, positions[n_p + j].copy()))
                while len(self._pos_histories[j]) > self.seq_len + self.future_steps + 1:
                    self._pos_histories[j].popleft()
            self._step = snap_step + 1

        # Differentiable member-task allocation. Fixed-size padded tensors keep
        # the public encoder contract stable across 4v2/5v2/6v2/6v3 scales.
        pursuer_h_full = h_active.new_zeros(self.max_pursuers, self.hidden_dim)
        if active_p:
            pursuer_h_full[torch.as_tensor(active_p, device=self.device)] = h_pursuer
        if not getattr(self, "_skip_legacy_plan", False):
            demand_logits, demand_hat = self.demand_head(target_h)
        else:
            demand_logits = target_h.new_zeros(self.max_evaders, 3)
            demand_hat = target_h.new_zeros(self.max_evaders)
        relation_costs = self.relation_head(
            pursuer_h_full[:, None, :], target_h[None, :, :],
        )
        pursuer_mask_tensor = torch.as_tensor(
            np.pad(pursuer_mask.astype(np.float32), (0, self.max_pursuers - n_p)),
            device=self.device,
        )
        task_mask_tensor = torch.as_tensor(
            np.pad(task_mask.astype(np.float32), (0, self.max_evaders - n_ev)),
            device=self.device,
        )
        if not getattr(self, "_skip_legacy_plan", False):
            soft_assignment = differentiable_task_plan(
                relation_costs, demand_hat, pursuer_mask_tensor, task_mask_tensor,
                self.soft_sinkhorn,
            )
            soft_context = soft_coalition_context(
                pursuer_h_full, target_h, soft_assignment,
                pursuer_mask_tensor, task_mask_tensor,
            )
        else:
            soft_assignment = relation_costs.new_zeros(
                relation_costs.shape[0], relation_costs.shape[1])
            soft_context = relation_costs.new_zeros(
                relation_costs.shape[0], 2 * self.hidden_dim)

        ring_slots = None
        if self.use_ring_slot_features:
            ring_slots = unique_ring_slot_features(
                positions[:n_p], positions[n_p:n_p + n_ev], predictions,
                assignment, roles, task_mask,
                d_cap=float(snap.get("d_cap", 2.0)), map_size=self.map_size,
                evader_velocities=velocities[n_p:n_p + n_ev], lookahead=0.4,
                slot_state=self._ring_slot_state,
                phase_hysteresis=self.ring_slot_phase_hysteresis,
                normalizer=self.ring_slot_normalizer,
            )

        # ---- per-pursuer padded observations
        obs = np.zeros((self.max_pursuers, self.obs_dim), dtype=np.float32)
        differentiable_rows = [
            torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
            for _ in range(self.max_pursuers)
        ]
        masks = np.zeros(self.max_pursuers, dtype=np.float32)
        h_i_map = {idx: h_pursuer[a] for a, idx in enumerate(active_p)}
        h_j_map = {j: target_h[j] for j in active_e}
        active_p_set = set(active_p)

        for i in range(n_p):
            if not pursuer_mask[i]:
                continue
            masks[i] = 1.0
            p_pos = positions[i]
            p_vel = velocities[i]
            L = self.map_size
            # strict partial observability: this pursuer only senses entities
            # within its perception disk; anything else must not leak into obs
            if self.perception_radius is None:
                def _sees(idx: int) -> bool:
                    return True
            else:
                def _sees(idx: int) -> bool:
                    return bool(
                        np.linalg.norm(p_pos - positions[idx]) <= self.perception_radius
                    )

            base = [
                p_pos[0] / L,
                p_pos[1] / L,
                p_vel[0] / 0.35,
                p_vel[1] / 0.35,
                p_pos[0] / L,
                (L - p_pos[0]) / L,
                p_pos[1] / L,
                (L - p_pos[1]) / L,
                p_pos[2] / 2.5,
                p_vel[2] / 0.35,
            ]

            team = []
            others = sorted(
                [(float(np.linalg.norm(positions[k] - p_pos)), k) for k in range(n_p) if k != i and k in active_p_set]
            )
            # partial observability: invisible teammates contribute zero features
            others = [
                (d, k) for d, k in others
                if self.perception_radius is None or d <= self.perception_radius
            ]
            for _, k in others[: self.max_pursuers - 1]:
                team += [positions[k][0] / L, positions[k][1] / L, positions[k][2] / 2.5]
            while len(team) < 3 * (self.max_pursuers - 1):
                team += [0.0, 0.0, 0.0]

            role = int(roles[i]) if i < roles.shape[0] else ROLE_CORE
            target = -1  # V2 基线修复：失活任务/离线成员的 legacy 状态不触发 UnboundLocalError
            if role == ROLE_RES or not (0 <= int(assignment[i]) < n_ev and task_mask[int(assignment[i])]):
                # mobile reserve: risk-weighted aggregate over VISIBLE active
                # targets only (invisible targets must not leak their state)
                weights = []
                for j in range(n_ev):
                    if not task_mask[j] or not _sees(n_p + j):
                        weights.append(0.0)
                        continue
                    d = float(np.linalg.norm(positions[n_p + j] - p_pos))
                    weights.append(1.0 / (1.0 + d))
                wsum = float(np.sum(weights)) or 1.0
                weights = [w / wsum for w in weights]
                t_pos = [0.0, 0.0, 0.0]
                for j in range(n_ev):
                    if weights[j] <= 0.0:
                        continue
                    t_pos[0] += weights[j] * positions[n_p + j][0] / L
                    t_pos[1] += weights[j] * positions[n_p + j][1] / L
                    t_pos[2] += weights[j] * positions[n_p + j][2] / 2.5
                stage_code = MODE_TRK
                coalition_size = 0.0
                h_target = torch.zeros(self.hidden_dim, device=self.device)
                for j in range(n_ev):
                    if task_mask[j] and weights[j] > 0.0:
                        h_target = h_target + weights[j] * h_j_map.get(j, torch.zeros(self.hidden_dim, device=self.device))
                pred_feat = np.zeros(self.future_steps * 2, dtype=np.float32)
                for j in range(n_ev):
                    if task_mask[j] and weights[j] > 0.0:
                        # predictions are in meters; obs features must be normalized
                        pred_feat += weights[j] * predictions[j].reshape(-1)
                pred_feat /= self.map_size
                unc = np.array([sum(weights[j] * uncertainty[j] for j in range(n_ev))], dtype=np.float32)
            else:
                target = int(assignment[i])
                # assigned target state is only observable within perception;
                # when the target is out of sight its position/embedding/prediction
                # are hidden (zeroed), so moving it cannot change this obs
                target_visible = _sees(n_p + target)
                if target_visible:
                    t_pos = [
                        positions[n_p + target][0] / L,
                        positions[n_p + target][1] / L,
                        positions[n_p + target][2] / 2.5,
                    ]
                    h_target = h_j_map.get(target, torch.zeros(self.hidden_dim, device=self.device))
                    pred_feat = predictions[target].reshape(-1) / self.map_size
                    unc = np.array([uncertainty[target]], dtype=np.float32)
                else:
                    t_pos = [0.0, 0.0, 0.0]
                    h_target = torch.zeros(self.hidden_dim, device=self.device)
                    pred_feat = np.zeros(self.future_steps * 2, dtype=np.float32)
                    unc = np.array([0.0], dtype=np.float32)
                stage_code = int(stages[target])
                coalition_size = float(len([k for k in range(n_p) if roles[k] == ROLE_CORE and assignment[k] == target])) / 3.0

            stage_oh = [0.0] * MODE_DIM
            if self.use_stage_embedding:
                stage_oh[min(stage_code, MODE_DIM - 1)] = 1.0
            role_oh = [0.0] * ROLE_DIM
            if self.use_role_embedding:
                role_oh[min(role, ROLE_DIM - 1)] = 1.0  # ablation: zero embedding

            obst = self._nearest_obstacles(positions[i], obstacles, k=2, L=L)

            h_i = h_i_map.get(i)
            if h_i is None:
                h_i = torch.zeros(self.hidden_dim, device=self.device)
            coalition_pool = self._coalition_pool(h_pursuer, active_p, roles, assignment, role, target if role != ROLE_RES else -1, device=self.device, query=h_i)
            if differentiable:
                # The policy sees soft task/member context. Discrete roles and
                # assignments remain in the non-graph features and hard path.
                h_target = soft_context[i, : self.hidden_dim]
                coalition_pool = soft_context[i, self.hidden_dim :]
            if not self.use_graph_embedding:
                # ablation: remove all learned graph features from the observation
                # (node embeddings + coalition pool) to test raw geometry only.
                h_i = torch.zeros(self.hidden_dim, device=self.device)
                h_target = torch.zeros(self.hidden_dim, device=self.device)
                coalition_pool = torch.zeros(self.hidden_dim, device=self.device)
            graph_tensor = torch.cat([h_i, h_target, coalition_pool])
            graph = graph_tensor.detach().cpu().numpy()

            slot_feat = np.zeros(0, dtype=np.float32)
            if ring_slots is not None:
                slot_feat = ring_slots[i].copy()
                assigned = int(assignment[i])
                if assigned < 0 or assigned >= n_ev or not _sees(n_p + assigned):
                    slot_feat[:] = 0.0

            vec = np.concatenate(
                [np.asarray(base, dtype=np.float32),
                 np.asarray(team, dtype=np.float32),
                 np.asarray(t_pos, dtype=np.float32),
                 np.asarray(stage_oh, dtype=np.float32),
                 np.asarray(role_oh, dtype=np.float32),
                 np.asarray([coalition_size, 1.0], dtype=np.float32),
                 np.asarray(obst, dtype=np.float32),
                 graph.astype(np.float32),
                 pred_feat.astype(np.float32),
                 unc.astype(np.float32),
                 slot_feat]
            )
            obs[i] = vec
            if differentiable:
                prefix = np.concatenate(
                    [np.asarray(base, dtype=np.float32),
                     np.asarray(team, dtype=np.float32),
                     np.asarray(t_pos, dtype=np.float32),
                     np.asarray(stage_oh, dtype=np.float32),
                     np.asarray(role_oh, dtype=np.float32),
                     np.asarray([coalition_size, 1.0], dtype=np.float32),
                     np.asarray(obst, dtype=np.float32)]
                )
                suffix = np.concatenate(
                    [pred_feat.astype(np.float32), unc.astype(np.float32), slot_feat]
                )
                differentiable_rows[i] = torch.cat(
                    [torch.as_tensor(prefix, device=self.device),
                     graph_tensor,
                     torch.as_tensor(suffix, device=self.device)]
                )

        # ---- relation cost (n_p x n_ev) for the allocator
        relation_cost = relation_costs[:n_p, :n_ev].detach().cpu().numpy().astype(np.float32)

        encoded_obs = (
            torch.stack(differentiable_rows)
            if differentiable
            else torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        if getattr(self, "output_norm", None) is not None:
            encoded_obs = self.output_norm(encoded_obs) * self.output_gain
        return {
            "obs": encoded_obs,
            "masks": torch.as_tensor(masks, dtype=torch.float32, device=self.device),
            "predictions": predictions,
            "uncertainty": uncertainty,
            "relation_cost": relation_cost,
            "relation_costs": relation_costs,
            "demand_logits": demand_logits,
            "demand_hat": demand_hat,
            "soft_assignment": soft_assignment,
            "pursuer_embeddings": pursuer_h_full,
            "task_embeddings": target_h,
            "stages": stages.astype(np.int64),
            "task_mask": task_mask.astype(np.int64),
            "captured": captured.astype(np.int64),
            "assignment": assignment.astype(np.int64),
            "num_pursuers": n_p,
            "num_evaders": n_ev,
        }

    # ---------------------------------------------------------------- helpers
    def _nearest_task(self, positions, n_p, n_ev, task_mask, i) -> int:
        best, best_d = -1, float("inf")
        for j in range(n_ev):
            if not task_mask[j]:
                continue
            d = float(np.linalg.norm(positions[i] - positions[n_p + j]))
            if d < best_d:
                best_d, best = d, j
        return best

    def _nearest_obstacles(self, pos: np.ndarray, obstacles: Sequence, k: int, L: float) -> List[float]:
        # partial observability: only obstacles within the perception disk are sensed
        if self.perception_radius is None:
            ranked = sorted(obstacles, key=lambda o: float(np.linalg.norm(np.asarray(pos)[:2] - o[0])))
        else:
            sensed = [
                o for o in obstacles
                if float(np.linalg.norm(np.asarray(pos)[:2] - o[0])) <= self.perception_radius
            ]
            ranked = sorted(sensed, key=lambda o: float(np.linalg.norm(np.asarray(pos)[:2] - o[0])))
        out = []
        for o in ranked[:k]:
            out += [o[0][0] / L, o[0][1] / L]
        while len(out) < 2 * k:
            out += [0.0, 0.0]
        return out

    def _coalition_pool(self, h_pursuer: torch.Tensor, active_p: Sequence[int],
                        roles: np.ndarray, assignment: np.ndarray, role: int,
                        target: int, device: torch.device,
                        query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Task-role-conditioned attended coalition pooling (core + support).

        Pool members = every pursuer assigned to ``target`` with role CORE or
        SUPPORT (the paper coalition).  When a query (this pursuer's embedding)
        is available the member contributions are re-weighted by a softmax over
        the learned relation head -- an actual attended pool, not a plain mean.
        The result is normalized by the total attention mass, so the pooled
        feature stays scale-invariant to the coalition size.
        """
        if role == ROLE_RES or self.coalition_pool_mode == "none":
            return torch.zeros(self.hidden_dim, device=device)
        members = [
            a for a, idx in enumerate(active_p)
            if roles[idx] in (ROLE_CORE, ROLE_SUP) and assignment[idx] == target
        ]
        if not members:
            return torch.zeros(self.hidden_dim, device=device)
        mem_h = torch.stack([h_pursuer[m] for m in members])  # (K, H)
        if self.coalition_pool_mode == "mean":
            return torch.mean(mem_h, dim=0)
        if self.coalition_pool_mode == "fixed_concat":
            slots = torch.zeros(
                self.max_pursuers,
                self.hidden_dim + 1,
                dtype=h_pursuer.dtype,
                device=device,
            )
            for active_offset in members:
                pursuer_id = int(active_p[active_offset])
                slots[pursuer_id, : self.hidden_dim] = h_pursuer[active_offset]
                slots[pursuer_id, self.hidden_dim] = 1.0
            return self.fixed_concat_projector(slots.reshape(-1))
        if query is None or members.__len__() == 1:
            return torch.mean(mem_h, dim=0)
        with torch.no_grad():
            costs = torch.stack(
                [self.relation_head(query, h_pursuer[m]).squeeze(-1) for m in members]
            )
            # relation_head is trained as a COST (larger = worse match), so the
            # attention weight must use the NEGATED cost. Reusing the member-task
            # head for member-member pairs is an inference-time similarity prior
            # only; it receives no gradient here.
            weights = torch.softmax(-costs, dim=0)
        return torch.sum(weights.unsqueeze(-1) * mem_h, dim=0)

    def auxiliary_parameters(self) -> List[nn.Parameter]:
        """Parameters optimized by the delayed relation objective."""
        return (
            list(self.st_hgat.parameters())
            + list(self.relation_head.parameters())
            + list(self.demand_head.parameters())
        )

    def auxiliary_loss(
        self,
        current_step: int,
        batch_size: int = 32,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Delayed future-interception supervision for the relation encoder."""
        samples = []
        for j in range(self.max_evaders):
            pos_by_step = {s: p for s, p in self._pos_histories[j]}
            for item in self._snapshots[j]:
                t = int(item["step"])
                if t + self.future_steps > current_step:
                    continue
                future = [pos_by_step.get(t + h) for h in range(1, self.future_steps + 1)]
                if any(p is None for p in future):
                    continue
                targets = np.stack([p[:2] / self.map_size for p in future]).astype(np.float32)
                samples.append((item, targets))
        if len(samples) < 2:
            return None

        rng = np.random.default_rng(current_step)
        picked = [samples[i] for i in rng.permutation(len(samples))[:batch_size]]
        relation_losses = []
        for item, target_np in picked:
            h_active = self.st_hgat.step(
                item["node_features"].to(self.device),
                item["h_prev"].to(self.device),
                {k: v.to(self.device) for k, v in item["adj_dict"].items()},
                {k: v.to(self.device) for k, v in item["dist_dict"].items()},
            )
            n_pa = int(item["n_pa"])
            h_evader = h_active[n_pa + int(item["evader_offset"])]
            targets = torch.as_tensor(target_np, dtype=torch.float32, device=self.device)
            h_pursuer = h_active[:n_pa]
            h_target = h_evader.unsqueeze(0).expand(n_pa, -1)
            relation_pred = self.relation_head(h_pursuer, h_target)
            pursuer_xy = torch.as_tensor(item["pursuer_xy"], dtype=torch.float32, device=self.device)
            future_distance = torch.linalg.vector_norm(
                targets.unsqueeze(0) - pursuer_xy.unsqueeze(1), dim=-1,
            ).min(dim=1).values
            relation_target = 0.3 * torch.tanh(4.0 * (future_distance - 0.15))
            relation_losses.append(torch.nn.functional.smooth_l1_loss(relation_pred, relation_target))

        relation_loss = torch.stack(relation_losses).mean()
        return {
            "total": relation_loss,
            "relation_loss": relation_loss,
        }


class SharedActor(nn.Module):
    """Per-node shared policy: obs -> normalized planar velocity in [-1, 1]^2."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Pre-tanh policy logits.

        The action-norm penalty must be applied HERE, not on the tanh output:
        d(tanh)/d(logit) = 1 - tanh^2, so at |a|=0.98 the penalty's gradient is
        attenuated ~25x exactly where it is needed most, which is why the action
        magnitude kept growing (0.975 -> 1.108) despite the penalty.
        """
        return self.net(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class SetInvariantCritic(nn.Module):
    """Centralized set-invariant critic: Q(G,A) = rho(sum_i mask_i psi(h_i,a_i,e))."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(obs_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """obs: (B, Np_max, obs_dim), actions: (B, Np_max, 2), masks: (B, Np_max)."""
        per_node = self.psi(torch.cat([obs, actions], dim=-1))  # (B, Np, H)
        summed = torch.sum(per_node * masks.unsqueeze(-1), dim=1)  # (B, H)
        return self.rho(summed).squeeze(-1)  # (B,)
