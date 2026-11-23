"""Core-support-reserve coalition allocation (paper core method, revised route).

Implements the algorithm side of main_core_support.tex:

* stage capacity kappa(m) constrains only the minimum number of *core* members
  (trk=1, blk=2, cap=3, exit/landed=0); support and mobile-reserve members are
  not counted against it;
* every active pursuer receives exactly one operational role: core of target j,
  support of target j, or global mobile reserve (roles are mutually exclusive
  and exhaustive, eq. role_exhaustive);
* support slots are generated from the predicted escape direction (eq.
  support_slots) and the reserve cost is a *risk-weighted travel time* to the
  candidate support slots -- never a "stop moving" idle cost (eq. reserve_cost);
* masked Sinkhorn over the extended matrix [C^c | C^s | C^r] (eq. masked_ot),
  followed by a discrete projection maximizing <P*, Z> with separate penalties
  for target switches and within-target role switches (eq. discrete_projection);
* hysteresis + dwell time suppress reconfiguration churn (eq. switch_t).
"""

from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .stage_env import (
    MODE_TRK,
    MODE_BLK,
    MODE_CAP,
    MODE_EXIT,
    ROLE_CORE,
    ROLE_SUP,
    ROLE_RES,
    escape_gap,
    gap_containing,
    mode_capacity,
)
from .training_protocol import DEFAULT_D_CAP


class MaskedSinkhorn(nn.Module):
    """Entropy-regularized optimal transport with invalid-edge masking."""

    def __init__(self, epsilon: float = 0.5, max_iters: int = 200, threshold: float = 1e-9):
        super().__init__()
        self.epsilon = float(epsilon)
        self.max_iters = int(max_iters)
        self.threshold = float(threshold)

    def forward(
        self,
        cost: torch.Tensor,
        row_supply: torch.Tensor,
        col_demand: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log-domain Sinkhorn.

        Args:
            cost:      (M, N) cost matrix (masked edges -> +inf)
            row_supply: (M,) row marginals (1.0 for every active pursuer)
            col_demand: (N,) column marginals (core/support/reserve demands)
            mask:      (M, N) bool, True = valid edge
        Returns:
            P: (M, N) soft transport plan.
        """
        device = cost.device
        M, N = cost.shape
        eps = self.epsilon
        orig_dtype = cost.dtype
        cost = cost.to(torch.float64)
        row_supply = row_supply.to(torch.float64)
        col_demand = col_demand.to(torch.float64)

        masked_cost = torch.where(mask, cost, torch.full_like(cost, float("inf")))
        log_K = -masked_cost / eps
        log_K = torch.where(mask, log_K, torch.full_like(log_K, -float("inf")))

        u = torch.zeros(M, device=device, dtype=torch.float64)
        v = torch.zeros(N, device=device, dtype=torch.float64)
        log_r = torch.log(row_supply + 1e-15)
        log_c = torch.log(col_demand + 1e-15)

        for _ in range(self.max_iters):
            u_prev = u.clone()
            u = log_r - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_c - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
            if torch.max(torch.abs(u - u_prev)) < self.threshold:
                break

        log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
        P = torch.exp(log_P)
        P = torch.where(torch.isfinite(P), P, torch.zeros_like(P))
        return P.to(orig_dtype)


def hard_plan_numpy(soft_plan: torch.Tensor) -> np.ndarray:
    """Cross the sole autograd boundary used by discrete environment execution."""
    return soft_plan.detach().to(device="cpu", dtype=torch.float64).numpy()


def extend_task_plan_for_hard_projection(
    task_plan: torch.Tensor,
    active_tasks: torch.Tensor,
    support_scale: float = 0.5,
) -> torch.Tensor:
    """Split one learned task plan into core/support/reserve execution columns."""
    selected = task_plan.index_select(1, active_tasks.to(task_plan.device).long())
    task_mass = selected.sum(dim=-1, keepdim=True)
    selected = selected / task_mass.clamp_min(1.0)
    reserve = (1.0 - selected.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    support_scale = max(0.0, float(support_scale))
    core = selected / (1.0 + support_scale)
    support = selected - core
    return torch.cat([core, support, reserve], dim=-1)


def coalition_laplacian(assignment: Sequence[int], n_pursuers: int) -> np.ndarray:
    """Laplacian of the coalition graph: edge (i,k) iff pursuers share a target."""
    L = np.zeros((n_pursuers, n_pursuers))
    for i in range(n_pursuers):
        for k in range(i + 1, n_pursuers):
            if assignment[i] >= 0 and assignment[i] == assignment[k]:
                L[i, k] = -1.0
                L[k, i] = -1.0
                L[i, i] += 1.0
                L[k, k] += 1.0
    return L


def support_slots(
    evader_pos: np.ndarray,
    pred_head: np.ndarray,
    d_f: float = 0.6,
    d_l: float = 0.5,
) -> List[np.ndarray]:
    """Two support slots along the predicted escape direction (eq. support_slots).

    Operates in the horizontal plane: only the (x, y) components of the evader
    position / predicted heading are used, so the returned slots are 2-D.
    """
    evader_xy = np.asarray(evader_pos, dtype=np.float64)[:2]
    head_xy = np.asarray(pred_head, dtype=np.float64)[:2]
    delta = head_xy - evader_xy
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        d_hat = np.array([1.0, 0.0])
    else:
        d_hat = delta / norm
    perp = np.array([-d_hat[1], d_hat[0]])
    base = evader_xy + d_f * d_hat
    return [base + d_l * perp, base - d_l * perp]


class StageCoalitionAllocator:
    """Joint stage-core-support-reserve optimization with switch suppression."""

    def __init__(
        self,
        d_cap: float = DEFAULT_D_CAP,
        psi_max_blk: float = 150.0 * np.pi / 180.0,
        switch_penalty: float = 0.15,
        hysteresis: float = 0.05,
        min_dwell_steps: int = 15,
        soft_infeasible_distance: float = 1.8,
        hard_infeasible_distance: float = 2.4,
        progress_patience_steps: int = 10,
        progress_epsilon: float = 0.05,
        topology_penalty: float = 0.05,
        epsilon: float = 0.5,
        device: str = "auto",
        support_cap: int = 2,
        role_set: str = "core_support_reserve",
        use_reserve: bool = True,
        demand_mode: str = "stage",
        d_f: float = 0.6,
        d_l: float = 0.5,
        merge_core_support: bool = False,
        use_stage_feasibility: bool = True,
        allow_promotion: bool = True,
        coalition_mode: str = "dynamic",
        solver_mode: str = "sinkhorn",
        # soft stage-feasibility costs (structure hard floor = 1 core per active
        # target; stage capacities beyond that are soft, never hard rejections)
        feasibility_capacity_weight: float = 1.0,
        feasibility_distance_weight: float = 1.0,
        feasibility_geometry_weight: float = 1.0,
        feasibility_distance_band: float = 1.0,
        # counterfactual / cross-layer dwell: when any active evader has >=3
        # pursuers within d_cap, lock that trio as its core (nearest 3) and
        # freeze the plan until capture / hard infeasibility / stalled progress
        lock_on_close3: bool = False,
        lock_radius: Optional[float] = None,
        lock_preassign_on_capture: bool = False,
        lock_support_slots: int = 0,
    ):
        if role_set not in {"core_support_reserve", "core_reserve"}:
            raise ValueError(
                "role_set must be 'core_support_reserve' or 'core_reserve'",
            )
        if demand_mode not in {"stage", "fixed", "direct"}:
            raise ValueError(
                "demand_mode must be 'stage', 'fixed', or 'direct'",
            )
        if solver_mode not in {"sinkhorn", "greedy", "hungarian"}:
            raise ValueError(
                "solver_mode must be 'sinkhorn', 'greedy', or 'hungarian'",
            )
        self.role_set = role_set
        self.use_reserve = bool(use_reserve)
        self.demand_mode = demand_mode
        self.solver_mode = solver_mode
        self.d_cap = float(d_cap)
        self.psi_max_blk = float(psi_max_blk)
        self.switch_penalty = float(switch_penalty)
        self.hysteresis = float(hysteresis)
        self.min_dwell_steps = int(min_dwell_steps)
        self.soft_infeasible_distance = float(soft_infeasible_distance)
        self.hard_infeasible_distance = float(hard_infeasible_distance)
        self.progress_patience_steps = int(progress_patience_steps)
        self.progress_epsilon = float(progress_epsilon)
        self.topology_penalty = float(topology_penalty)
        self.epsilon = float(epsilon)
        self.support_cap = 0 if role_set == "core_reserve" else int(support_cap)
        self.d_f = float(d_f)
        self.d_l = float(d_l)
        # ablation switches (paper section: ablation study)
        self.merge_core_support = bool(merge_core_support)
        self.use_stage_feasibility = bool(use_stage_feasibility)
        self.allow_promotion = bool(allow_promotion)
        if coalition_mode not in {"dynamic", "fixed_membership"}:
            raise ValueError(f"invalid coalition_mode {coalition_mode!r}")
        self.coalition_mode = coalition_mode
        for name, value in (
            ("feasibility_capacity_weight", feasibility_capacity_weight),
            ("feasibility_distance_weight", feasibility_distance_weight),
            ("feasibility_geometry_weight", feasibility_geometry_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            setattr(self, name, float(value))
        if feasibility_distance_band <= 0.0:
            raise ValueError(
                f"feasibility_distance_band must be > 0, got {feasibility_distance_band}"
            )
        self.feasibility_distance_band = float(feasibility_distance_band)
        self.lock_on_close3 = bool(lock_on_close3)
        self.lock_radius = None if lock_radius is None else float(lock_radius)
        self.lock_preassign_on_capture = bool(lock_preassign_on_capture)
        self.lock_support_slots = max(0, int(lock_support_slots))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.sinkhorn = MaskedSinkhorn(epsilon=epsilon)
        self.reset()

    def reset(self) -> None:
        self.prev_roles: Optional[np.ndarray] = None
        self.prev_core_target: Optional[np.ndarray] = None
        self.prev_sup_target: Optional[np.ndarray] = None
        self.prev_stages: Optional[np.ndarray] = None
        self.steps_since_switch = self.min_dwell_steps
        self._last_switch_step = -10**9
        # behaviour-neutral diagnostics for the stability-gate question
        self.diag_forced_switch = 0
        self.diag_force_reason = {}
        self.diag_blocked_by_dwell = 0
        self.diag_blocked_by_hysteresis = 0
        self.diag_blocked_by_no_change = 0
        self._best_core_distance: Dict[int, float] = {}
        self._last_progress_step: Dict[int, int] = {}
        self.total_target_switches = 0
        self.total_role_switches = 0
        self._lock_members = {}
        self.fixed_group_by_pursuer: Optional[np.ndarray] = None
        self._fixed_target_by_group: Dict[int, int] = {}
        # lock_on_close3 bookkeeping: target -> locked core member list
        self._lock_members: Dict[int, List[int]] = {}
        self._prev_free_mask: Optional[np.ndarray] = None

    def _reconfiguration_reason(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        task_masks: np.ndarray,
        step: int,
    ) -> str:
        """Classify whether the previous role plan must be held or released.

        A capture attempt is an active target with at least three previously
        assigned core members. Core and support assignments are held as one
        plan during minimum dwell and while every capture coalition is feasible
        or making measurable progress. A completed target bypasses dwell so its
        members can be redeployed immediately.
        """
        if self.prev_roles is None or self.prev_core_target is None:
            return "initial"

        previous_targets = np.where(
            self.prev_roles == ROLE_CORE,
            self.prev_core_target,
            np.where(self.prev_roles == ROLE_SUP, self.prev_sup_target, -1),
        )
        for target in np.unique(previous_targets[previous_targets >= 0]):
            if int(target) >= len(task_masks) or not bool(task_masks[int(target)]):
                return "capture_completed"

        capture_targets = []
        for target in np.flatnonzero(task_masks):
            members = np.flatnonzero(
                (self.prev_roles == ROLE_CORE) & (self.prev_core_target == int(target))
            )
            if len(members) >= 3:
                capture_targets.append((int(target), members))
        if not capture_targets:
            return "unlocked"

        if int(step) - self._last_switch_step < self.min_dwell_steps:
            return "minimum_dwell"

        progressing = False
        for target, members in capture_targets:
            max_distance = float(np.max(np.linalg.norm(
                pursuer_pos[members] - evader_pos[target], axis=1,
            )))
            if max_distance > self.hard_infeasible_distance:
                return "hard_infeasible"
            if max_distance <= self.soft_infeasible_distance:
                self._best_core_distance.pop(target, None)
                self._last_progress_step[target] = int(step)
                continue

            best_distance = self._best_core_distance.get(target)
            if best_distance is None:
                self._best_core_distance[target] = max_distance
                self._last_progress_step[target] = int(step)
                progressing = True
                continue
            if max_distance <= best_distance - self.progress_epsilon:
                self._best_core_distance[target] = max_distance
                self._last_progress_step[target] = int(step)
                progressing = True
                continue
            if int(step) - self._last_progress_step.get(
                target, int(step)
            ) >= self.progress_patience_steps:
                return "progress_stalled"
            progressing = True
        return "progressing" if progressing else "capture_feasible"

    @staticmethod
    def _reason_holds_plan(reason: str) -> bool:
        return reason in {"minimum_dwell", "progressing", "capture_feasible"}

    # ------------------------------------------------------------ feasibility
    def evaluate_feasibility(
        self,
        core: Sequence[int],
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
    ) -> Dict[str, object]:
        k = len(core)
        pts = pursuer_pos[list(core)] if k else np.zeros((0, 3))
        psi = escape_gap(pts[:, :2], evader_pos[:2])
        max_dist = float(np.max(np.linalg.norm(pts - evader_pos, axis=1))) if k else float("inf")
        f_trk = float(k >= 1)
        f_blk = float(k >= 2 and gap_containing(pts[:, :2], evader_pos[:2], evader_vel[:2]) <= self.psi_max_blk)
        f_cap = float(k >= 3 and max_dist <= self.d_cap and psi < np.pi)
        realized = MODE_CAP if f_cap > 0.5 else (MODE_BLK if f_blk > 0.5 else MODE_TRK)
        return {
            "k": k,
            "psi": psi,
            "max_dist": max_dist,
            "f_trk": f_trk,
            "f_blk": f_blk,
            "f_cap": f_cap,
            "realized_stage": realized,
        }

    def evaluate_soft_feasibility(
        self,
        core: Sequence[int],
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        desired_stage: int,
    ) -> Dict[str, float]:
        """Continuous stage-feasibility cost for a candidate core coalition.

        Unlike ``evaluate_feasibility`` (hard boolean gates), this returns
        bounded soft deficits that scale smoothly with how far the coalition is
        from the desired stage.  The environment's strict capture rule
        (>=3 within d_cap, convex-hull enclosure, hold_steps) is never changed
        by this cost; the allocator only *prefers* cheaper coalitions.
        """
        members = [int(i) for i in core]
        k = len(members)
        pts = pursuer_pos[members] if k else np.zeros((0, 3))
        evader = np.asarray(evader_pos, dtype=np.float64)
        required = max(1, int(mode_capacity(int(desired_stage))))
        capacity_deficit = float(max(0, required - k) / max(required, 1))

        max_dist = (
            float(np.max(np.linalg.norm(pts[:, :2] - evader[:2], axis=1)))
            if k else float("inf")
        )
        distance_deficit = float(
            np.clip(
                max(0.0, max_dist - self.d_cap) / max(self.feasibility_distance_band, 1e-9),
                0.0, 1.0,
            )
        )

        geometry_deficit = 0.0
        stage = int(desired_stage)
        if stage == MODE_CAP:
            if k < 3:
                geometry_deficit = 1.0
            else:
                psi = escape_gap(pts[:, :2], evader[:2])
                if not np.isfinite(psi):
                    geometry_deficit = 1.0
                else:
                    geometry_deficit = float(
                        np.clip(max(0.0, psi - np.pi) / np.pi, 0.0, 1.0)
                    )
        elif stage == MODE_BLK:
            if k < 2:
                geometry_deficit = 1.0
            else:
                gap = gap_containing(pts[:, :2], evader[:2], np.asarray(evader_vel, dtype=np.float64)[:2])
                if not np.isfinite(gap):
                    geometry_deficit = 1.0
                else:
                    geometry_deficit = float(
                        np.clip(
                            max(0.0, gap - self.psi_max_blk) / max(self.psi_max_blk, 1e-9),
                            0.0, 1.0,
                        )
                    )
        # TRK has no geometric closure requirement
        total = (
            self.feasibility_capacity_weight * capacity_deficit
            + self.feasibility_distance_weight * distance_deficit
            + self.feasibility_geometry_weight * geometry_deficit
        )
        return {
            "total": float(total),
            "capacity_deficit": float(capacity_deficit),
            "distance_deficit": float(distance_deficit),
            "geometry_deficit": float(geometry_deficit),
        }

    def _soft_allocation_metrics(
        self,
        core_target: np.ndarray,
        roles: np.ndarray,
        stages: np.ndarray,
        active: np.ndarray,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
    ) -> Dict[str, float]:
        """Mean soft-feasibility deficits over active targets + capture ratio.

        ``hard_capture_feasible_ratio`` reuses the allocator's strict f_cap as a
        *diagnostic only*; the environment's capture rule is the authority.
        """
        if active.size == 0:
            return {
                "feasibility_cost": 0.0,
                "capacity_deficit": 0.0,
                "distance_deficit": 0.0,
                "geometry_deficit": 0.0,
                "hard_capture_feasible_ratio": 0.0,
            }
        cost, cap, dist, geo, hard = 0.0, 0.0, 0.0, 0.0, 0.0
        for j in active:
            j = int(j)
            core = [
                i for i in range(len(roles))
                if roles[i] == ROLE_CORE and int(core_target[i]) == j
            ]
            d = self.evaluate_soft_feasibility(
                core, pursuer_pos, evader_pos[j], evader_vel[j], int(stages[j]),
            )
            cost += d["total"]
            cap += d["capacity_deficit"]
            dist += d["distance_deficit"]
            geo += d["geometry_deficit"]
            hard += float(
                self.evaluate_feasibility(
                    core, pursuer_pos, evader_pos[j], evader_vel[j],
                )["f_cap"]
            )
        m = max(1, active.size)
        return {
            "feasibility_cost": float(cost / m),
            "capacity_deficit": float(cap / m),
            "distance_deficit": float(dist / m),
            "geometry_deficit": float(geo / m),
            "hard_capture_feasible_ratio": float(hard / m),
        }

    def _close3_lock_plan(
        self,
        active: np.ndarray,
        task_masks: np.ndarray,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        prefs: np.ndarray,
        n_p: int,
        free_mask: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Counterfactual/cross-layer dwell: lock nearest-3 core while inside d_cap.

        Returns a frozen role plan (or None) when lock_on_close3 is active and
        some active evader currently has >=3 pursuers within ``d_cap`` (locked
        trios persist until the target is captured, members exceed the hard
        infeasible distance, or the lock registry is cleared on reset).
        """
        if not self.lock_on_close3:
            return None
        radius = self.d_cap if self.lock_radius is None else float(self.lock_radius)
        # prune expired locks (captured target leaves task mask; trio ran away)
        captured_release = False
        for j in list(self._lock_members):
            j = int(j)
            members = self._lock_members[j]
            if j >= len(task_masks) or not task_masks[j]:
                del self._lock_members[j]
                captured_release = True
                continue
            if members:
                md = float(np.max(
                    np.linalg.norm(pursuer_pos[members] - evader_pos[j], axis=1)
                ))
                if md > self.hard_infeasible_distance:
                    del self._lock_members[j]
        # acquire new locks on close3
        for j in active:
            j = int(j)
            if j in self._lock_members or not task_masks[j]:
                continue
            d = np.linalg.norm(pursuer_pos[:, :2] - evader_pos[j][:2], axis=1)
            cand = [i for i in range(n_p) if free_mask[i] and d[i] <= radius]
            if len(cand) >= 3:
                trio = [int(i) for i in sorted(cand, key=lambda i: float(d[i]))[:3]]
                self._lock_members[j] = trio
        # pre-lock after a capture: dedicate the nearest three to the remaining
        # target immediately (no radius gate) so a second coalition re-forms fast
        if captured_release and self.lock_preassign_on_capture:
            for j in active:
                j = int(j)
                if j in self._lock_members or not task_masks[j]:
                    continue
                d = np.linalg.norm(pursuer_pos[:, :2] - evader_pos[j][:2], axis=1)
                cand = [i for i in range(n_p) if free_mask[i]]
                if len(cand) >= 3:
                    trio = [int(i) for i in sorted(cand, key=lambda i: float(d[i]))[:3]]
                    self._lock_members[j] = trio
        if not self._lock_members:
            return None
        core = (
            self.prev_core_target.copy()
            if self.prev_core_target is not None and len(self.prev_core_target) == n_p
            else -np.ones(n_p, dtype=np.int64)
        )
        sup = (
            self.prev_sup_target.copy()
            if self.prev_sup_target is not None and len(self.prev_sup_target) == n_p
            else -np.ones(n_p, dtype=np.int64)
        )
        roles = (
            self.prev_roles.copy()
            if self.prev_roles is not None and len(self.prev_roles) == n_p
            else np.full(n_p, ROLE_RES, dtype=np.int64)
        )
        for j, members in self._lock_members.items():
            for i in members:
                core[i] = j
                roles[i] = ROLE_CORE
                if int(sup[i]) == int(j):
                    sup[i] = -1
        # optional support slots: dedicated extra members to the locked target
        # (they count toward the coalition geometry but not the core floor)
        if self.lock_support_slots > 0:
            for j, members in self._lock_members.items():
                d = np.linalg.norm(pursuer_pos[:, :2] - evader_pos[int(j)][:2], axis=1)
                extras = [
                    int(i) for i in sorted(range(n_p), key=lambda i: float(d[i]))
                    if free_mask[i] and int(core[i]) == -1 and int(sup[i]) != int(j)
                ][: self.lock_support_slots]
                for i in extras:
                    core[i] = -1
                    sup[i] = int(j)
                    roles[i] = ROLE_SUP
        stages = (
            self.prev_stages.copy()
            if self.prev_stages is not None and len(self.prev_stages) == len(prefs)
            else prefs.copy()
        )
        return {
            "core_target": core,
            "sup_target": sup,
            "roles": roles,
            "stages": stages,
        }

    # ------------------------------------------------------------ demand
    def _compute_demand(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        task_masks: np.ndarray,
        free_mask: np.ndarray,
    ) -> np.ndarray:
        """Target-level integer demand kappa in {1,2,3} mapped to TRK/BLK/CAP."""
        if self.demand_mode == "fixed":
            return self._fixed_demand(task_masks)
        if self.demand_mode == "direct":
            return self._direct_demand(
                pursuer_pos, evader_pos, evader_vel, task_masks, free_mask,
            )
        return self._stage_demand(
            pursuer_pos, evader_pos, evader_vel, task_masks, free_mask,
        )

    def _kappa_to_stage(self, kappa: int) -> int:
        return {1: MODE_TRK, 2: MODE_BLK, 3: MODE_CAP}.get(int(kappa), MODE_TRK)

    def _fixed_demand(self, task_masks: np.ndarray) -> np.ndarray:
        """Every active target always requests the configured fixed stage.

        Default is MODE_CAP (kappa=3), preserving the historical `demand=fixed`
        behaviour. `self.fixed_demand_stage` allows a kappa dose-response scan
        (MODE_TRK=1 / MODE_BLK=2 / MODE_CAP=3) so the under-allocation
        hypothesis can be tested instead of assumed.
        """
        stage = int(getattr(self, "fixed_demand_stage", MODE_CAP))
        n_ev = len(task_masks)
        stages = np.full(n_ev, MODE_EXIT, dtype=np.int64)
        for j in range(n_ev):
            if task_masks[j]:
                stages[j] = stage
        return stages

    def _stage_demand(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        task_masks: np.ndarray,
        free_mask: np.ndarray,
    ) -> np.ndarray:
        """Original staged heuristic (cap > blk > trk) from urgency."""
        n_ev = len(task_masks)
        prefs = np.full(n_ev, MODE_TRK, dtype=np.int64)
        free_idx = np.where(free_mask)[0]
        for j in range(n_ev):
            if not task_masks[j]:
                prefs[j] = MODE_EXIT
                continue
            if free_idx.size < 3:
                prefs[j] = MODE_BLK if free_idx.size >= 2 else MODE_TRK
                continue
            dists = np.linalg.norm(pursuer_pos[free_idx] - evader_pos[j], axis=1)
            nearest_order = np.argsort(dists)[:3]
            nearest = free_idx[nearest_order]
            pts = pursuer_pos[nearest]
            psi = escape_gap(pts[:, :2], evader_pos[j][:2])
            if float(np.max(dists[nearest_order])) <= self.d_cap * 1.35 and psi < np.pi:
                prefs[j] = MODE_CAP
            elif free_idx.size >= 2:
                prefs[j] = MODE_BLK
        return prefs

    def _direct_demand(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        task_masks: np.ndarray,
        free_mask: np.ndarray,
    ) -> np.ndarray:
        """Direct geometric demand kappa = f(d_min, psi, closure).

        Maps the integer demand to TRK/BLK/CAP only for backward compatibility
        with the existing stage/capacity interface; the actor may receive a
        zeroed stage one-hot when use_stage_embedding is disabled.
        """
        n_ev = len(task_masks)
        prefs = np.full(n_ev, MODE_TRK, dtype=np.int64)
        free_idx = np.where(free_mask)[0]
        for j in range(n_ev):
            if not task_masks[j]:
                prefs[j] = MODE_EXIT
                continue
            if free_idx.size < 1:
                continue
            dists = np.linalg.norm(pursuer_pos[free_idx] - evader_pos[j], axis=1)
            nearest_order = np.argsort(dists)
            kappa = 1
            if free_idx.size >= 3:
                top3 = free_idx[nearest_order[:3]]
                pts = pursuer_pos[top3]
                psi = escape_gap(pts[:, :2], evader_pos[j][:2])
                if float(np.max(dists[nearest_order[:3]])) <= self.d_cap and psi < np.pi:
                    kappa = 3
            if kappa == 1 and free_idx.size >= 2:
                top2 = free_idx[nearest_order[:2]]
                pts = pursuer_pos[top2]
                gap = gap_containing(
                    pts[:, :2], evader_pos[j][:2], evader_vel[j][:2],
                )
                if np.isfinite(gap) and gap <= self.psi_max_blk:
                    kappa = 2
            prefs[j] = self._kappa_to_stage(kappa)
        return prefs

    def stage_preferences(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        task_masks: np.ndarray,
        free_mask: np.ndarray,
    ) -> np.ndarray:
        """Target-level preferred stage (cap > blk > trk) from urgency heuristics."""
        return self._compute_demand(
            pursuer_pos, evader_pos, evader_vel, predictions, uncertainty,
            task_masks, free_mask,
        )

    # ------------------------------------------------------------ edge costs
    def edge_costs(
        self,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        obstacles: Sequence[Tuple[np.ndarray, float]],
        prev_core_target: Optional[np.ndarray],
        prev_sup_target: Optional[np.ndarray],
        relation_cost: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[np.ndarray]]]:
        """Core cost C^c, support cost C^s, reserve cost C^r and support slots."""
        n_p, n_ev = pursuer_pos.shape[0], evader_pos.shape[0]
        C_c = np.zeros((n_p, n_ev), dtype=np.float64)
        C_s = np.zeros((n_p, n_ev), dtype=np.float64)
        # per-pursuer obstacle risk (permutation equivariant; each row uses its own)
        r_obs_arr = np.zeros(n_p, dtype=np.float64)
        for i in range(n_p):
            r_obs = 0.0
            for o_pos, o_r in obstacles:
                if np.linalg.norm(pursuer_pos[i, :2] - o_pos) < o_r + 0.6:
                    r_obs += (o_r + 0.6 - np.linalg.norm(pursuer_pos[i, :2] - o_pos)) / 0.6
            r_obs_arr[i] = r_obs
        slots: List[List[np.ndarray]] = []
        for j in range(n_ev):
            pred = predictions[j] if predictions is not None else evader_pos[j][None, :2]
            head = pred[-1] if len(pred) else evader_pos[j][:2]
            slots.append(support_slots(evader_pos[j], head, self.d_f, self.d_l))
        for i in range(n_p):
            r_obs = r_obs_arr[i]
            for j in range(n_ev):
                pred = predictions[j] if predictions is not None else evader_pos[j][None, :2]
                d_int = float(np.min(np.linalg.norm(pred - pursuer_pos[i, :2], axis=1)))
                unc = float(uncertainty[j]) if uncertainty is not None else 0.0
                sw_c = 0.0
                if prev_core_target is not None and int(prev_core_target[i]) != j:
                    sw_c = 1.0
                rel = float(relation_cost[i, j]) if relation_cost is not None else 0.0
                C_c[i, j] = (
                    1.0 * d_int + 1.5 * r_obs + 0.5 * unc + self.switch_penalty * sw_c + rel
                )
                # support cost: time to the nearest support slot - promotion gain
                t_slot = min(
                    float(np.linalg.norm(pursuer_pos[i, :2] - q) / 0.35) for q in slots[j]
                )
                gain = max(0.0, 1.0 - d_int / (2.0 * self.d_cap))
                sw_s = 0.0
                if prev_sup_target is not None and int(prev_sup_target[i]) != j:
                    sw_s = 1.0
                C_s[i, j] = 1.0 * t_slot - 0.5 * gain + self.switch_penalty * sw_s + 0.5 * r_obs
        # reserve cost: risk-weighted travel time to the support slots of all targets
        C_r = np.zeros(n_p, dtype=np.float64)
        if n_ev > 0:
            urgency = np.zeros(n_ev)
            for j in range(n_ev):
                pred = predictions[j] if predictions is not None else evader_pos[j][None, :]
                dist = float(np.linalg.norm(pursuer_pos - evader_pos[j], axis=1).min())
                urgency[j] = 1.0 / (1.0 + dist) + (
                    float(uncertainty[j]) if uncertainty is not None else 0.0
                )
            rho = urgency / max(float(np.sum(urgency)), 1e-9)
            for i in range(n_p):
                cost_res = 0.0
                for j in range(n_ev):
                    t_min = min(
                        float(np.linalg.norm(pursuer_pos[i, :2] - q) / 0.35) for q in slots[j]
                    )
                    cost_res += rho[j] * t_min
                C_r[i] = cost_res + 1.5 * r_obs_arr[i]
        return C_c, C_s, C_r, slots

    # ------------------------------------------------------------ allocation
    def allocate(
        self,
        pursuer_pos: np.ndarray,
        pursuer_vel: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        predictions: Optional[np.ndarray],
        uncertainty: Optional[np.ndarray],
        obstacles: Sequence[Tuple[np.ndarray, float]],
        task_masks: np.ndarray,
        free_mask: np.ndarray,
        relation_cost: Optional[np.ndarray] = None,
        step: int = 0,
        soft_task_plan: Optional[torch.Tensor] = None,
    ) -> Dict:
        n_p = len(pursuer_pos)
        n_ev = len(task_masks)
        active = np.where(task_masks)[0]

        if active.size == 0:
            core = -np.ones(n_p, dtype=np.int64)
            sup = -np.ones(n_p, dtype=np.int64)
            roles = np.full(n_p, ROLE_RES, dtype=np.int64)
            stages = np.full(n_ev, MODE_EXIT, dtype=np.int64)
            packed = self._pack(core, sup, roles, stages, None, step, 0, 0)
            packed.update(self._soft_allocation_metrics(
                core, roles, stages, active, pursuer_pos, evader_pos, evader_vel,
            ))
            self._prev_free_mask = free_mask.copy()
            return packed

        membership_changed = (
            self._prev_free_mask is not None
            and not np.array_equal(self._prev_free_mask, free_mask)
        )

        # ---- preferred stages and resource downgrade
        prefs = self.stage_preferences(
            pursuer_pos, evader_pos, evader_vel, predictions, uncertainty, task_masks, free_mask
        )
        free_n = int(np.count_nonzero(free_mask))
        if self.demand_mode != "fixed":
            # fixed demand intentionally keeps CAP demand and reports deficit;
            # other modes degrade gracefully when resources are scarce.
            prefs = self._downgrade_for_resources(prefs, active, free_n)

        # ---- costs
        C_c, C_s, C_r, slots = self.edge_costs(
            pursuer_pos, evader_pos, evader_vel, predictions, uncertainty,
            obstacles, self.prev_core_target, self.prev_sup_target, relation_cost,
        )

        # ---- masked Sinkhorn over [C^c | C^s | C^r]
        m = active.shape[0]
        caps = np.array([mode_capacity(int(prefs[j])) for j in active], dtype=np.float64)
        rem = float(free_n) - float(np.sum(caps))
        if rem > 0 and not self.use_reserve:
            # No-Reserve ablation: preserve the CSR encoding/network shape,
            # but allocate every unit left after core demand to Support.
            sup_cap = np.zeros(m, dtype=np.float64)
            for unit in range(int(round(rem))):
                sup_cap[unit % m] += 1.0
        elif rem > 0:
            sup_cap = np.full(m, min(self.support_cap, rem))
            # trim so that the reserve column keeps at least one unit of mass
            while float(np.sum(sup_cap)) > rem - 1.0 and float(np.sum(sup_cap)) > 0.0:
                sup_cap[np.argmax(sup_cap)] -= 1.0
            sup_cap = np.maximum(sup_cap, 0.0)
        else:
            sup_cap = np.zeros(m)
        res_cap = max(0.0, float(free_n) - float(np.sum(caps)) - float(np.sum(sup_cap)))
        demands = np.concatenate([caps, sup_cap, [res_cap]])
        n_cols = 2 * m + 1
        cost_ext = np.zeros((n_p, n_cols))
        cost_ext[:, :m] = C_c[:, active]
        cost_ext[:, m : 2 * m] = C_s[:, active]
        cost_ext[:, 2 * m] = C_r
        mask = np.zeros((n_p, n_cols), dtype=bool)
        for i in range(n_p):
            for a in range(m):
                mask[i, a] = bool(free_mask[i] and task_masks[active[a]])          # core
                mask[i, m + a] = bool(free_mask[i] and task_masks[active[a]])      # support
            mask[i, 2 * m] = bool(free_mask[i] and self.use_reserve)               # reserve
        row_supply = free_mask.astype(np.float64)

        soft = None
        if soft_task_plan is not None and self.solver_mode == "sinkhorn":
            plan_tensor = torch.as_tensor(
                soft_task_plan, dtype=torch.float32, device=self.device,
            )
            if plan_tensor.shape != (n_p, n_ev):
                raise ValueError(
                    f"soft_task_plan shape {tuple(plan_tensor.shape)} != {(n_p, n_ev)}"
                )
            learned_extended = extend_task_plan_for_hard_projection(
                plan_tensor, torch.as_tensor(active, device=self.device),
            )
            learned_extended = learned_extended * torch.as_tensor(
                free_mask, dtype=learned_extended.dtype, device=self.device,
            ).unsqueeze(-1)
            soft = hard_plan_numpy(learned_extended)
        elif self.coalition_mode == "fixed_membership" or self.solver_mode == "sinkhorn":
            # Compatibility path for legacy callers that do not provide the
            # policy's learned P_soft. The end-to-end SDGCR path never enters it.
            soft = hard_plan_numpy(self.sinkhorn(
                torch.as_tensor(cost_ext, dtype=torch.float32, device=self.device),
                torch.as_tensor(row_supply, dtype=torch.float32, device=self.device),
                torch.as_tensor(demands, dtype=torch.float32, device=self.device),
                torch.as_tensor(mask, dtype=torch.bool, device=self.device),
            ))

        if self.coalition_mode == "fixed_membership":
            return self._allocate_fixed_membership(
                active, prefs, C_c, soft, pursuer_pos, evader_pos, evader_vel,
                free_mask, step,
            )

        # ---- discrete assignment
        if self.solver_mode == "sinkhorn":
            if not self.use_stage_feasibility:
                best = self._project_unconstrained(
                    soft, active, prefs, free_mask, n_ev,
                )
            else:
                forbidden = set()
                if not self.allow_promotion and self.prev_roles is not None and self.prev_sup_target is not None:
                    for i in range(n_p):
                        if self.prev_roles[i] == ROLE_SUP and 0 <= int(self.prev_sup_target[i]) < n_ev:
                            forbidden.add((i, int(self.prev_sup_target[i])))
                best = self._project(
                    soft, active, prefs, pursuer_pos, evader_pos, evader_vel,
                    free_mask, n_p, n_ev, forbidden=forbidden,
                )
                if best["core_target"] is None:
                    best = self._project(
                        soft, active, prefs, pursuer_pos, evader_pos, evader_vel,
                        free_mask, n_p, n_ev, strict_capacity=False, forbidden=forbidden,
                    )
                if best["core_target"] is None:
                    # graceful degradation when no stage-feasible plan exists.
                    trk_prefs = prefs.copy()
                    for j in range(len(trk_prefs)):
                        trk_prefs[j] = MODE_TRK
                    best = self._project(
                        soft, active, trk_prefs, pursuer_pos, evader_pos, evader_vel,
                        free_mask, n_p, n_ev, strict_capacity=False, forbidden=forbidden,
                    )
        elif self.solver_mode == "greedy":
            best = self._solve_greedy(
                cost_ext, mask, demands, active, prefs, n_p, n_ev,
            )
        else:
            best = self._solve_hungarian(
                cost_ext, mask, demands, active, prefs, n_p, n_ev,
            )
        assert best["core_target"] is not None, "no feasible role assignment found"

        # ---- adaptive feasibility gate + score hysteresis
        switched_t, switched_r = 0, 0
        has_previous = (
            self.prev_roles is not None
            and self.prev_stages is not None
            and len(self.prev_roles) == n_p
            and len(self.prev_stages) == n_ev
        )
        reason = self._reconfiguration_reason(
            pursuer_pos, evader_pos, task_masks, step,
        )
        if membership_changed:
            reason = "membership_changed"
        lock_plan = self._close3_lock_plan(
            active, task_masks, pursuer_pos, evader_pos, prefs, n_p, free_mask,
        )
        if membership_changed:
            lock_plan = None
        changed = False
        if lock_plan is not None:
            # close3 lock overrides candidate/hysteresis: freeze the plan while
            # a target has >=3 pursuers inside d_cap (cross-layer dwell probe)
            core = lock_plan["core_target"]
            sup = lock_plan["sup_target"]
            roles = lock_plan["roles"]
            stages = lock_plan["stages"]
            reason = "close3_locked"
            switched_t, switched_r = self._count_switches(core, sup, roles)
            changed = switched_t + switched_r > 0
        elif has_previous and self._reason_holds_plan(reason):
            # Lock core and support as one plan.  This is the support-role
            # hysteresis required by the physical deployment protocol.
            core = self.prev_core_target.copy()
            sup = self.prev_sup_target.copy()
            roles = self.prev_roles.copy()
            stages = self.prev_stages.copy()
        elif has_previous:
            force_reconfigure = reason in {
                "capture_completed", "hard_infeasible", "progress_stalled",
                "membership_changed",
            }
            # Diagnostics only (behaviour-neutral counters). These exist to answer a
            # concrete experimental question: the hysteresis/dwell gate can ONLY
            # suppress *marginal* reconfigurations, while `force_reconfigure` bypasses
            # it entirely. If forced reasons dominate, turning the gate off cannot
            # change the switch count -- which is exactly what was observed.
            if force_reconfigure:
                self.diag_forced_switch += 1
                self.diag_force_reason[reason] = self.diag_force_reason.get(reason, 0) + 1
            accept_best = force_reconfigure
            if not force_reconfigure:
                may_switch = int(step) - self._last_switch_step >= self.min_dwell_steps
                if not may_switch:
                    self.diag_blocked_by_dwell += 1
                if self.solver_mode == "sinkhorn":
                    old_score = self._score(
                        soft, self.prev_core_target, self.prev_sup_target, self.prev_roles,
                        self.prev_stages, active, pursuer_pos, evader_pos, evader_vel,
                    )
                    new_score = best["score"]
                    accept_best = may_switch and new_score - old_score >= self.hysteresis
                    if may_switch and not accept_best:
                        self.diag_blocked_by_hysteresis += 1
                else:
                    # Greedy/Hungarian produce a discrete optimum; switch if the new
                    # plan differs and dwell has elapsed.
                    switched_t, switched_r = self._count_switches(
                        best["core_target"], best["sup_target"], best["roles"],
                    )
                    accept_best = may_switch and (switched_t + switched_r > 0)
                    if may_switch and not accept_best:
                        self.diag_blocked_by_no_change += 1
            if accept_best:
                core, sup, roles, stages = (
                    best["core_target"], best["sup_target"], best["roles"], best["stages"],
                )
                switched_t, switched_r = self._count_switches(core, sup, roles)
                changed = switched_t + switched_r > 0
            else:
                core = self.prev_core_target.copy()
                sup = self.prev_sup_target.copy()
                roles = self.prev_roles.copy()
                stages = self.prev_stages.copy()
        else:
            core, sup, roles, stages = (
                best["core_target"], best["sup_target"], best["roles"], best["stages"],
            )
            self._last_switch_step = int(step)

        repaired = self._enforce_no_reserve(
            {"core_target": core, "sup_target": sup, "roles": roles},
            active, free_mask, soft, pursuer_pos, evader_pos,
        )
        core, sup, roles = (
            repaired["core_target"], repaired["sup_target"], repaired["roles"],
        )

        if changed:
            self._last_switch_step = int(step)
            self._best_core_distance.clear()
            self._last_progress_step.clear()
        self.prev_core_target = core.copy()
        self.prev_sup_target = sup.copy()
        self.prev_roles = roles.copy()
        self.prev_stages = stages.copy()
        self.steps_since_switch = max(0, int(step) - self._last_switch_step)
        self.total_target_switches += switched_t
        self.total_role_switches += switched_r
        self._prev_free_mask = free_mask.copy()

        packed = self._pack(core, sup, roles, stages, soft, step, switched_t, switched_r)
        packed.update(self._soft_allocation_metrics(
            core, roles, stages, active, pursuer_pos, evader_pos, evader_vel,
        ))
        packed["reconfiguration_reason"] = reason
        return packed

    # ------------------------------------------------------------ internals
    def _project_unconstrained(
        self,
        soft: np.ndarray,
        active: np.ndarray,
        prefs: np.ndarray,
        free_mask: np.ndarray,
        n_ev: int,
    ) -> Dict:
        """Convert Sinkhorn mass to roles without capacity/geometry repair."""
        n_p = soft.shape[0]
        m = len(active)
        core = -np.ones(n_p, dtype=np.int64)
        sup = -np.ones(n_p, dtype=np.int64)
        roles = np.full(n_p, ROLE_RES, dtype=np.int64)
        score = 0.0
        for i in range(n_p):
            if not free_mask[i]:
                continue
            column = int(np.argmax(soft[i]))
            score += float(soft[i, column])
            if column < m:
                core[i] = int(active[column])
                roles[i] = ROLE_CORE
            elif column < 2 * m:
                sup[i] = int(active[column - m])
                roles[i] = ROLE_SUP
        assert np.all((core < n_ev) & (sup < n_ev))
        return {
            "core_target": core,
            "sup_target": sup,
            "roles": roles,
            "stages": prefs.copy(),
            "score": score,
        }

    def _assignment_from_columns(
        self,
        assignment: Sequence[int],
        active: np.ndarray,
        prefs: np.ndarray,
        n_p: int,
        n_ev: int,
    ) -> Dict:
        """Map extended column indices to core_target/sup_target/roles/stages."""
        m = len(active)
        core = -np.ones(n_p, dtype=np.int64)
        sup = -np.ones(n_p, dtype=np.int64)
        roles = np.full(n_p, ROLE_RES, dtype=np.int64)
        for i, col in enumerate(assignment):
            if col < 0:
                continue
            if col < m:
                core[i] = int(active[col])
                roles[i] = ROLE_CORE
            elif col < 2 * m:
                sup[i] = int(active[col - m])
                roles[i] = ROLE_SUP
        return {
            "core_target": core,
            "sup_target": sup,
            "roles": roles,
            "stages": prefs.copy(),
            "score": 0.0,
        }

    def _solve_greedy(
        self,
        cost_ext: np.ndarray,
        mask: np.ndarray,
        demands: np.ndarray,
        active: np.ndarray,
        prefs: np.ndarray,
        n_p: int,
        n_ev: int,
    ) -> Dict:
        """Capacity-aware greedy assignment to the cheapest valid column."""
        free_idx = np.where(mask.any(axis=1))[0]
        col_demand = demands.copy()
        assigned_col = -np.ones(n_p, dtype=np.int64)
        entries = []
        for i in free_idx:
            for col in np.where(mask[i])[0]:
                entries.append((float(cost_ext[i, col]), int(i), int(col)))
        entries.sort(key=lambda x: x[0])
        for _, i, col in entries:
            if assigned_col[i] >= 0 or col_demand[col] <= 0.0:
                continue
            assigned_col[i] = col
            col_demand[col] -= 1.0
        # any unassigned pursuer becomes reserve (last column if valid)
        reserve_col = cost_ext.shape[1] - 1
        for i in free_idx:
            if assigned_col[i] < 0 and mask[i, reserve_col]:
                assigned_col[i] = reserve_col
        return self._assignment_from_columns(
            assigned_col, active, prefs, n_p, n_ev,
        )

    def _solve_hungarian(
        self,
        cost_ext: np.ndarray,
        mask: np.ndarray,
        demands: np.ndarray,
        active: np.ndarray,
        prefs: np.ndarray,
        n_p: int,
        n_ev: int,
    ) -> Dict:
        """Linear-sum assignment over duplicated capacity slots."""
        free_idx = np.where(mask.any(axis=1))[0]
        n_free = len(free_idx)
        if n_free == 0:
            return self._assignment_from_columns(
                -np.ones(n_p, dtype=np.int64), active, prefs, n_p, n_ev,
            )
        # build slot list: (column, target_role) for each unit of demand
        slots = []  # slot index -> (column, role)
        m = len(active)
        for a in range(m):
            for _ in range(int(round(demands[a]))):
                slots.append((a, ROLE_CORE))
            for _ in range(int(round(demands[m + a]))):
                slots.append((m + a, ROLE_SUP))
        # reserve slots fill the rest so we have a square n_free x n_free problem
        reserve_col = cost_ext.shape[1] - 1
        n_reserve = max(0, n_free - len(slots))
        for _ in range(n_reserve):
            slots.append((reserve_col, ROLE_RES))
        # if not enough slots, pad with dummy reserve slots (zero cost, ignored)
        while len(slots) < n_free:
            slots.append((reserve_col, ROLE_RES))
        slots = slots[:n_free]

        # cost matrix: free pursuer x slot
        cost_mat = np.full((n_free, n_free), 1e6, dtype=np.float64)
        for r, i in enumerate(free_idx):
            for c, (col, _) in enumerate(slots):
                if mask[i, col]:
                    cost_mat[r, c] = cost_ext[i, col]
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        assigned_col = -np.ones(n_p, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            i = free_idx[r]
            col, _ = slots[c]
            assigned_col[i] = col
        return self._assignment_from_columns(
            assigned_col, active, prefs, n_p, n_ev,
        )

    def _allocate_fixed_membership(
        self,
        active: np.ndarray,
        prefs: np.ndarray,
        core_cost: np.ndarray,
        soft: np.ndarray,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        free_mask: np.ndarray,
        step: int,
    ) -> Dict:
        """Allocate immutable pursuer groups whose target may change as a unit."""
        n_p = len(pursuer_pos)
        n_ev = len(prefs)
        if self.fixed_group_by_pursuer is None or len(self.fixed_group_by_pursuer) != n_p:
            groups = -np.ones(n_p, dtype=np.int64)
            online = list(np.flatnonzero(free_mask))
            base, extra = divmod(len(online), len(active))
            remaining = set(int(i) for i in online)
            for group, target in enumerate(active):
                size = base + int(group < extra)
                ranked = sorted(remaining, key=lambda i: (float(core_cost[i, target]), i))
                chosen = ranked[:size] if group < len(active) - 1 else sorted(remaining)
                for i in chosen:
                    groups[i] = group
                    remaining.discard(i)
                self._fixed_target_by_group[group] = int(target)
            self.fixed_group_by_pursuer = groups

        core = -np.ones(n_p, dtype=np.int64)
        sup = -np.ones(n_p, dtype=np.int64)
        roles = np.full(n_p, ROLE_RES, dtype=np.int64)
        stages = prefs.copy()
        for group in np.unique(self.fixed_group_by_pursuer[self.fixed_group_by_pursuer >= 0]):
            members = np.flatnonzero(
                (self.fixed_group_by_pursuer == group) & free_mask,
            )
            if len(members) == 0:
                continue
            target = self._fixed_target_by_group.get(int(group), -1)
            if target not in active:
                target = min(
                    (int(j) for j in active),
                    key=lambda j: (float(np.mean(core_cost[members, j])), j),
                )
                self._fixed_target_by_group[int(group)] = target
            ranked = sorted(
                (int(i) for i in members),
                key=lambda i: (float(core_cost[i, target]), i),
            )
            core_members = ranked[: min(3, len(ranked))]
            for i in core_members:
                core[i] = target
                roles[i] = ROLE_CORE
            for i in ranked[len(core_members):]:
                sup[i] = target
                roles[i] = ROLE_SUP
            if self.use_stage_feasibility:
                feasibility = self.evaluate_feasibility(
                    core_members, pursuer_pos, evader_pos[target], evader_vel[target],
                )
                stages[target] = int(feasibility["realized_stage"])

        target_switch, role_switch = self._count_switches(core, sup, roles)
        self.prev_core_target = core.copy()
        self.prev_sup_target = sup.copy()
        self.prev_roles = roles.copy()
        self.prev_stages = stages.copy()
        self.total_target_switches += target_switch
        self.total_role_switches += role_switch
        packed = self._pack(
            core, sup, roles, stages, soft, step, target_switch, role_switch,
        )
        packed.update(self._soft_allocation_metrics(
            core, roles, stages, active, pursuer_pos, evader_pos, evader_vel,
        ))
        packed["reconfiguration_reason"] = "fixed_membership"
        return packed

    def _downgrade_for_resources(self, prefs: np.ndarray, active: np.ndarray, free_n: int) -> np.ndarray:
        while True:
            total = sum(mode_capacity(int(prefs[j])) for j in active)
            if total <= free_n:
                return prefs
            order = sorted(active, key=lambda j: -int(prefs[j]))
            candidate = None
            for j in order:
                if int(prefs[j]) == MODE_CAP:
                    candidate = j
                    break
            if candidate is None:
                for j in order:
                    if int(prefs[j]) == MODE_BLK:
                        candidate = j
                        break
            if candidate is None:
                return prefs
            prefs[candidate] = int(prefs[candidate]) - 1

    def _project(
        self,
        soft: np.ndarray,
        active: np.ndarray,
        prefs: np.ndarray,
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        free_mask: np.ndarray,
        n_p: int,
        n_ev: int,
        strict_capacity: bool = True,
        forbidden: Optional[set] = None,
    ) -> Dict:
        """Enumerate core coalitions (>= kappa) and support/reserve of the rest."""
        m = active.shape[0]
        free_idx = list(np.where(free_mask)[0])
        best = {
            "core_target": None, "sup_target": None, "roles": None,
            "stages": None, "score": -float("inf"),
        }
        if len(free_idx) == 0:
            core = -np.ones(n_p, dtype=np.int64)
            sup = -np.ones(n_p, dtype=np.int64)
            roles = np.full(n_p, ROLE_RES, dtype=np.int64)
            return {**best, "core_target": core, "sup_target": sup, "roles": roles,
                    "stages": prefs.copy(), "score": 0.0}

        # ---- enumerate core membership: target j gets k_j in [1, desired_j]
        # (hard floor is exactly one core per active target; stage capacities
        # beyond that are soft preferences, never hard rejections)
        desired = [mode_capacity(int(prefs[j])) for j in active]
        # prune core candidates per target by the soft plan (keeps enumeration small)
        core_candidates = []
        for a in range(m):
            col = soft[:, a]
            top = np.argsort(col)[::-1][: int(desired[a]) + 2]
            core_candidates.append([int(t) for t in top if int(t) in set(free_idx)])
        core_plans = self._enumerate_core(
            soft, free_idx, active, desired, m, strict_capacity, forbidden, core_candidates
        )
        if not core_plans:
            return best

        for core_plan in core_plans:
            core_target = -np.ones(n_p, dtype=np.int64)
            for i, slot in zip(free_idx, core_plan):
                if slot >= 0:
                    core_target[i] = int(active[slot])
            used = {i for i, s in zip(free_idx, core_plan) if s >= 0}
            rest = [i for i in free_idx if i not in used]

            # desired stages are kept as-is (soft feasibility); the env applies
            # the strict capture rule independently, never via this field
            stages = prefs.copy()

            # ---- support/reserve of the remaining pursuers (pruned enumeration)
            sup_plans = self._enumerate_support(rest, active, soft, m)
            for sup_plan in sup_plans:
                sup_target = -np.ones(n_p, dtype=np.int64)
                for i, slot in zip(rest, sup_plan):
                    if slot >= 0:
                        sup_target[i] = int(active[slot])
                roles = np.full(n_p, ROLE_RES, dtype=np.int64)
                for i in range(n_p):
                    if core_target[i] >= 0:
                        roles[i] = ROLE_CORE
                    elif sup_target[i] >= 0:
                        roles[i] = ROLE_SUP
                score = self._score(
                    soft, core_target, sup_target, roles, stages, active,
                    pursuer_pos, evader_pos, evader_vel,
                )
                if score > best["score"]:
                    core_final = core_target.copy()
                    sup_final = sup_target.copy()
                    roles_final = roles.copy()
                    if self.merge_core_support:
                        # ablation: core and support collapse into one member role
                        for i in range(n_p):
                            if roles_final[i] == ROLE_SUP and sup_final[i] >= 0:
                                core_final[i] = sup_final[i]
                                sup_final[i] = -1
                                roles_final[i] = ROLE_CORE
                    best = {
                        "core_target": core_final, "sup_target": sup_final,
                        "roles": roles_final, "stages": stages.copy(), "score": score,
                    }
        return best

    def _enumerate_core(
        self,
        soft: np.ndarray,
        free_idx: Sequence[int],
        active: np.ndarray,
        desired: Sequence[int],
        m: int,
        strict: bool,
        forbidden: Optional[set] = None,
        candidates: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[Tuple[int, ...]]:
        """Yield tuples (slot per free pursuer, -1 = not core).

        Soft stage feasibility: every active target keeps a hard floor of ONE
        core member; the per-target upper bound is its desired stage capacity
        (plus one extra when ``strict`` is False).  Desired CAP/BLK capacities
        are no longer hard lower bounds, so a CAP target may receive 1-2 core
        members while the coalition is still assembling.

        To stay tractable (full ``combinations`` over 1..desired members per
        target explodes combinatorially), each size ``k`` contributes only its
        TOP-k members by soft core value — the natural greedy completion of
        that size — which keeps the search linear in the enumeration ranges.
        """
        forbidden = forbidden or set()
        if candidates is None:
            candidates = [list(free_idx) for _ in range(m)]
        # rank each target's candidate pool once by soft core value (desc)
        ranked_pool = []
        for a in range(m):
            col = soft[:, int(active[a])]
            pool = sorted(
                {int(i) for i in candidates[a] if int(i) in free_idx},
                key=lambda i: (-float(col[i]), int(i)),
            )
            ranked_pool.append(pool)
        results: List[Tuple[int, ...]] = []
        state = [-1] * len(free_idx)
        idx_of = {i: a for a, i in enumerate(free_idx)}

        def rec(target_pos: int, remaining: List[int], state: List[int]) -> None:
            if target_pos == m:
                results.append(tuple(state))
                return
            k_low = 1
            k_high = int(desired[target_pos])
            if not strict and len(remaining) > k_high:
                k_high += 1  # fallback may over-allocate one core member
            k_high = min(k_high, len(remaining))
            available = [i for i in ranked_pool[target_pos] if i in remaining]
            if len(available) < k_low:
                return
            target_col = soft[:, int(active[target_pos])]
            # top-k greedy per size (already ranked), skipping forbidden members
            chosen_prefix = []
            for i in available:
                if len(chosen_prefix) >= k_high:
                    break
                if (int(i), int(active[target_pos])) in forbidden:
                    continue
                chosen_prefix.append(i)
            for k in range(k_low, min(k_high, len(available)) + 1):
                chosen = chosen_prefix[:k]
                if len(chosen) < k:
                    continue
                new_state = state.copy()
                for i in chosen:
                    new_state[idx_of[i]] = target_pos
                new_rem = [i for i in remaining if i not in chosen]
                rec(target_pos + 1, new_rem, new_state)

        rec(0, list(free_idx), state)
        return results

    def _enumerate_support(
        self,
        rest: Sequence[int],
        active: np.ndarray,
        soft: np.ndarray,
        m: int,
    ) -> List[Tuple[int, ...]]:
        """Support/reserve choices for the non-core pursuers (pruned by soft plan)."""
        if not rest:
            return [()]
        choices = []
        for i in rest:
            row = soft[i, m : 2 * m]  # support columns
            top = np.argsort(row)[::-1][:2]
            allowed = [int(t) for t in top]
            if self.use_reserve:
                allowed = [-1] + allowed
            choices.append(allowed)
        return list(product(*choices))

    def _enforce_no_reserve(
        self,
        plan: Dict,
        active: np.ndarray,
        free_mask: np.ndarray,
        soft: Optional[np.ndarray],
        pursuer_pos: np.ndarray,
        evader_pos: np.ndarray,
    ) -> Dict:
        """Project residual active Reserve members onto Support targets.

        This final projection covers all planner branches, including close-3
        lock and held previous plans, without changing role IDs or actor input.
        """
        if self.use_reserve or len(active) == 0:
            return plan
        core = plan["core_target"].copy()
        sup = plan["sup_target"].copy()
        roles = plan["roles"].copy()
        m = len(active)
        for i in np.where(free_mask & (roles == ROLE_RES))[0]:
            if soft is not None:
                local = int(np.argmax(soft[i, m : 2 * m]))
            else:
                local = int(np.argmin(
                    np.linalg.norm(evader_pos[active] - pursuer_pos[i], axis=1),
                ))
            roles[i] = ROLE_SUP
            sup[i] = int(active[local])
            core[i] = -1
        return {**plan, "core_target": core, "sup_target": sup, "roles": roles}

    def _score(
        self,
        soft: np.ndarray,
        core_target: np.ndarray,
        sup_target: np.ndarray,
        roles: np.ndarray,
        stages: np.ndarray,
        active: np.ndarray,
        pursuer_pos: Optional[np.ndarray] = None,
        evader_pos: Optional[np.ndarray] = None,
        evader_vel: Optional[np.ndarray] = None,
    ) -> float:
        n_p = core_target.shape[0]
        m = active.shape[0]
        score = 0.0
        for i in range(n_p):
            if roles[i] == ROLE_CORE and core_target[i] >= 0:
                hit = np.where(active == core_target[i])[0]
                if hit.size == 0:
                    score += float(soft[i, 2 * m])  # target left the task graph
                else:
                    score += float(soft[i, int(hit[0])])
            elif roles[i] == ROLE_SUP and sup_target[i] >= 0:
                hit = np.where(active == sup_target[i])[0]
                if hit.size == 0:
                    score += float(soft[i, 2 * m])
                else:
                    score += float(soft[i, m + int(hit[0])])
            else:
                score += float(soft[i, 2 * m])
        for a, j in enumerate(active):
            score += 0.25 * int(stages[j])
        # soft stage-feasibility cost: prefer coalitions closer to the desired
        # stage, but never hard-reject (hard floor of 1 core is enforced by the
        # enumeration ranges, not by this penalty)
        if (
            pursuer_pos is not None
            and evader_pos is not None
            and evader_vel is not None
        ):
            for j in active:
                core = [
                    i for i in range(n_p)
                    if roles[i] == ROLE_CORE and int(core_target[i]) == int(j)
                ]
                soft_cost = self.evaluate_soft_feasibility(
                    core, pursuer_pos, evader_pos[int(j)], evader_vel[int(j)],
                    int(stages[int(j)]),
                )
                score -= float(soft_cost["total"])
        if self.prev_roles is not None:
            t_sw, r_sw = self._count_switches(core_target, sup_target, roles)
            score -= self.switch_penalty * (t_sw + 0.5 * r_sw)
            new_target = np.where(
                roles == ROLE_CORE,
                core_target,
                np.where(roles == ROLE_SUP, sup_target, -1),
            )
            old_target = np.where(
                self.prev_roles == ROLE_CORE,
                self.prev_core_target,
                np.where(self.prev_roles == ROLE_SUP, self.prev_sup_target, -1),
            )
            delta_l = coalition_laplacian(new_target, n_p) - coalition_laplacian(old_target, n_p)
            score -= self.topology_penalty * float(np.sum(delta_l * delta_l))
        return float(score)

    def _count_switches(
        self,
        core_target: np.ndarray,
        sup_target: np.ndarray,
        roles: np.ndarray,
    ) -> Tuple[int, int]:
        if self.prev_roles is None:
            return 0, 0
        target_sw, role_sw = 0, 0
        for i in range(core_target.shape[0]):
            if roles[i] == ROLE_RES:
                new_t = -1
            elif roles[i] == ROLE_SUP:
                new_t = int(sup_target[i])
            else:
                new_t = int(core_target[i])
            if self.prev_roles[i] == ROLE_RES:
                old_t = -1
            elif self.prev_roles[i] == ROLE_SUP:
                old_t = int(self.prev_sup_target[i])
            else:
                old_t = int(self.prev_core_target[i])
            if new_t != old_t:
                target_sw += 1
            elif int(self.prev_roles[i]) != int(roles[i]):
                role_sw += 1
        return target_sw, role_sw

    def _pack(
        self,
        core_target: np.ndarray,
        sup_target: np.ndarray,
        roles: np.ndarray,
        stages: np.ndarray,
        soft: Optional[np.ndarray],
        step: int,
        target_switch: int,
        role_switch: int,
    ) -> Dict:
        invalid = 0
        capacity_deficit = 0
        for j in range(len(stages)):
            core = [i for i in range(len(roles)) if roles[i] == ROLE_CORE and core_target[i] == j]
            required = mode_capacity(int(stages[j]))
            if len(core) < required:
                invalid += 1
                capacity_deficit += required - len(core)
        idle = int(np.count_nonzero(roles == ROLE_RES))
        return {
            "core_assignment": core_target.copy(),
            "sup_assignment": sup_target.copy(),
            "roles": roles.copy(),
            "stages": stages.copy(),
            "soft": soft,
            "target_switch": int(target_switch),
            "role_switch": int(role_switch),
            "invalid": int(invalid),
            "capacity_deficit": int(capacity_deficit),
            "idle": idle,
            "total_target_switches": self.total_target_switches,
            "total_role_switches": self.total_role_switches,
            "step": step,
        }
