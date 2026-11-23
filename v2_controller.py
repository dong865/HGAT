"""continuous_membership_v2 统一控制器（阶段 C 核心交付）。

一个 `V2Controller.controller_step(env)` 同时服务训练采集、评估与 runtime：
每个物理时刻只推进一次图编码器，计算连续需求与软计划，按周期/事件做
硬求解与门控，用同一个 H 与接受后的归属构造 V2 观测，Actor 输出动作。

契约（V2_SCHEMA.md）：
- 学习计划缺失时报错（demand_source=learned_continuous 时需求头必须存在）；
- 每步只调 encode 一次（同物理步的重编码命中缓存，不二次推进）；
- Actor 永远看"已接受"的硬配置；
- 日志携带 architecture_version 与 allocation_source=learned_full_plan。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .continuous_membership import (
    ASSIGNMENT_UNASSIGNED,
    DEFAULT_COUNT_PENALTY,
    DEFAULT_HYSTERESIS,
    DEFAULT_MIN_DWELL_STEPS,
    DEFAULT_SINKHORN_ITERS,
    DEFAULT_SWITCH_PENALTY,
    DEFAULT_EPSILON_OT,
    ContinuousDemandHead,
    GateEvent,
    build_membership_context,
    build_soft_plan,
    hard_plan_to_assignment,
    membership_gate,
    solve_membership,
)
from .stage_env import StageCoalitionEnv
from .stage_networks import StageFeatureEncoder
from .v2_env_adapter import (
    apply_membership,
    capture_hold_active,
    membership_validity,
)

ARCHITECTURE_VERSION = "continuous_membership_v2"
ALLOCATION_SOURCE = "learned_full_plan"
OBSERVATION_SCHEMA_VERSION = "v2_no_stage_role"


def v2_obs_dim(max_pursuers: int, hidden_dim: int, future_steps: int) -> int:
    """[base10 | teammates 3*(Np-1) | target3 | membership2 | obst4
        | h_i H | t_i H | c_i H | pred 2F | unc1]"""
    return (10 + 3 * (max_pursuers - 1) + 3 + 2 + 4
            + 3 * hidden_dim + 2 * future_steps + 1)


class V2Controller:
    def __init__(
        self,
        encoder: StageFeatureEncoder,
        demand_head: ContinuousDemandHead,
        actor,
        device: torch.device,
        max_pursuers: int = 6,
        max_evaders: int = 3,
        epsilon_ot: float = DEFAULT_EPSILON_OT,
        sinkhorn_iters: int = DEFAULT_SINKHORN_ITERS,
        count_penalty: float = DEFAULT_COUNT_PENALTY,
        switch_penalty: float = DEFAULT_SWITCH_PENALTY,
        hysteresis: float = DEFAULT_HYSTERESIS,
        min_dwell_steps: int = DEFAULT_MIN_DWELL_STEPS,
        assign_interval: int = 5,
        demand_source: str = "learned_continuous",   # learned_continuous | fixed
        fixed_demand: float = 3.0,
        stop_plan_gradient: bool = False,             # 消融：上下文入口 detach P
        target_context: str = "hard",                 # 交互观测恒 hard；TD 目标分支见阶段 D
    ):
        assert demand_source in ("learned_continuous", "fixed")
        self.encoder = encoder
        self.demand_head = demand_head
        self.actor = actor
        self.device = device
        self.max_pursuers = int(max_pursuers)
        self.max_evaders = int(max_evaders)
        self.epsilon_ot = float(epsilon_ot)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.count_penalty = float(count_penalty)
        self.switch_penalty = float(switch_penalty)
        self.hysteresis = float(hysteresis)
        self.min_dwell_steps = int(min_dwell_steps)
        self.assign_interval = int(assign_interval)
        self.demand_source = demand_source
        self.fixed_demand = float(fixed_demand)
        self.stop_plan_gradient = bool(stop_plan_gradient)
        self.target_context = str(target_context)
        self.reset()

    # ------------------------------------------------------------------ 状态
    def reset(self):
        self.encoder.reset(step=0)
        self._step = 0
        self._assignment = np.full(self.max_pursuers, ASSIGNMENT_UNASSIGNED, dtype=np.int64)
        self._dwell = 10 ** 9          # 首步视为已可重构
        self._prev_energy = float('inf')   # 保证首次候选必被接受（无合法前配置可比）
        self._event: Optional[GateEvent] = None
        self.last_info: Dict = {}

    def request_force_reconfigure(self, reason: str = "stagnation"):
        self._event = GateEvent(force=True, reason=reason)

    # ------------------------------------------------------------------ 观测
    def _build_v2_obs(self, enc: Dict, assignment: np.ndarray,
                      env: StageCoalitionEnv) -> Tuple[torch.Tensor, torch.Tensor]:
        """软/硬共用同一聚合：K 为 [Np_max, Nt_max] 的 P（训练上下文）
        或 X one-hot（交互观测）。本函数构造 Actor 实际观测，K=X。"""
        n_p, n_ev = env.num_pursuers, env.num_evaders
        H = self.encoder.hidden_dim
        L = self.encoder.map_size
        F_ = self.encoder.future_steps
        np_max, nt_max = self.max_pursuers, self.max_evaders

        obs = np.zeros((np_max, v2_obs_dim(np_max, H, F_)), dtype=np.float32)
        masks = np.zeros(np_max, dtype=np.float32)

        # 硬 one-hot K（真实任务列，不含空列）
        X = torch.zeros((np_max, nt_max), dtype=torch.float32, device=self.device)
        for i in range(n_p):
            a = int(assignment[i])
            if 0 <= a < nt_max and bool(env.task_mask[a]):
                X[i, a] = 1.0

        pursuer_emb = enc["pursuer_embeddings"]            # [Np_max, H]
        task_emb = enc["task_embeddings"]                  # [Nt_max, H]
        with torch.no_grad():
            t_ctx, c_ctx = build_membership_context(X, pursuer_emb, task_emb)
        t_ctx = t_ctx.cpu().numpy()
        c_ctx = c_ctx.cpu().numpy()
        h_np = pursuer_emb.detach().cpu().numpy()
        pred = enc["predictions"]
        unc = enc["uncertainty"]

        # 未分配成员的目标几何：对可见活跃目标按风险加权（保留移动能力，非悬停）
        positions = env.positions
        for i in range(n_p):
            if not bool(env.pursuer_mask[i]):
                continue
            masks[i] = 1.0
            p_pos = positions[i]
            base = [
                p_pos[0] / L, p_pos[1] / L,
                env.velocities[i][0] / 0.35, env.velocities[i][1] / 0.35,
                p_pos[0] / L, (L - p_pos[0]) / L,
                p_pos[1] / L, (L - p_pos[1]) / L,
                p_pos[2] / 2.5, env.velocities[i][2] / 0.35,
            ]
            team = []
            others = sorted(
                [(float(np.linalg.norm(positions[k] - p_pos)), k)
                 for k in range(n_p) if k != i and env.pursuer_mask[k]]
            )
            for _, k in others[: np_max - 1]:
                team += [positions[k][0] / L, positions[k][1] / L, positions[k][2] / 2.5]
            while len(team) < 3 * (np_max - 1):
                team += [0.0, 0.0, 0.0]

            a = int(assignment[i])
            if 0 <= a < n_ev and bool(env.task_mask[a]):
                t_pos = [positions[n_p + a][0] / L, positions[n_p + a][1] / L,
                         positions[n_p + a][2] / 2.5]
                pred_feat = pred[a].reshape(-1) / L
                u = [unc[a]]
            else:
                weights = []
                for j in range(n_ev):
                    if not env.task_mask[j]:
                        weights.append(0.0)
                        continue
                    d = float(np.linalg.norm(positions[n_p + j] - p_pos))
                    weights.append(1.0 / (1.0 + d))
                wsum = float(np.sum(weights)) or 1.0
                weights = [w / wsum for w in weights]
                t_pos = [0.0, 0.0, 0.0]
                for j in range(n_ev):
                    if weights[j] <= 0:
                        continue
                    t_pos[0] += weights[j] * positions[n_p + j][0] / L
                    t_pos[1] += weights[j] * positions[n_p + j][1] / L
                    t_pos[2] += weights[j] * positions[n_p + j][2] / 2.5
                pred_feat = np.zeros(2 * F_, dtype=np.float32)
                u = [0.0]

            same = int(X[i].sum().item() > 0.5)
            same_count = int(X[i].sum().item()) and int(X.sum(dim=0)[int(np.argmax(X[i].cpu().numpy()))].item()) or 0
            n_avail = max(1, int(env.pursuer_mask[:n_p].sum()))
            membership_feat = [float(same), same_count / n_avail]

            obst = []
            obs_list = sorted(env.obstacles, key=lambda o: float(np.linalg.norm(o.position[:2] - p_pos[:2])))
            for o in obs_list[:2]:
                obst += [o.position[0] / L, o.position[1] / L]
            while len(obst) < 4:
                obst += [0.0, 0.0]

            vec = np.concatenate([
                np.asarray(base, dtype=np.float32),
                np.asarray(team, dtype=np.float32),
                np.asarray(t_pos, dtype=np.float32),
                np.asarray(membership_feat, dtype=np.float32),
                np.asarray(obst, dtype=np.float32),
                h_np[i].astype(np.float32),
                t_ctx[i].astype(np.float32),
                c_ctx[i].astype(np.float32),
                pred_feat.astype(np.float32),
                np.asarray(u, dtype=np.float32),
            ])
            obs[i] = vec
        return (torch.as_tensor(obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(masks, dtype=torch.float32, device=self.device))

    # ------------------------------------------------------------------ 主步
    def _demand(self, task_emb: torch.Tensor, task_mask: torch.Tensor) -> torch.Tensor:
        if self.demand_source == "learned_continuous":
            if self.demand_head is None:
                raise RuntimeError("demand_source=learned_continuous 但 demand_head 缺失")
            return self.demand_head(task_emb, task_mask)
        return torch.full((task_mask.shape[0],), self.fixed_demand,
                          device=task_mask.device) * task_mask.to(torch.float32)

    def controller_step(self, env: StageCoalitionEnv) -> Tuple[np.ndarray, Dict]:
        """第 10 节 policy_step。返回 (pursuer_actions[Np,2], info)。"""
        env.advance_lifecycle()
        snap = env.snapshot()
        n_p, n_ev = env.num_pursuers, env.num_evaders

        # 1) 每物理时刻只推进一次图编码器（后续同快照重编码命中缓存）
        enc = self.encoder.encode(snap, differentiable=False)

        # 2) 连续需求 → 有效容量 → 软计划（同一份 P̄ 用于训练上下文与硬求解）
        task_mask_t = torch.as_tensor(
            np.pad(env.task_mask[:n_ev].astype(np.float32),
                   (0, self.max_evaders - n_ev)), device=self.device)
        pursuer_mask_t = torch.as_tensor(
            np.pad(env.pursuer_mask[:n_p].astype(np.float32),
                   (0, self.max_pursuers - n_p)), device=self.device)
        demand = self._demand(enc["task_embeddings"], task_mask_t)
        nt_max = self.max_evaders
        costs = enc["relation_costs"]
        if costs.shape[1] < nt_max:
            costs = torch.nn.functional.pad(costs, (0, nt_max - costs.shape[1]))
        plan = build_soft_plan(costs, demand, pursuer_mask_t, task_mask_t,
                               epsilon=self.epsilon_ot, iters=self.sinkhorn_iters)

        # 3) 重构周期 / 强制事件 / 归属失效应立即处理（不等普通周期）
        reconfigured = False
        gate_reason = "period_pending"
        prev_valid = membership_validity(env, self._assignment)
        needs_reconfig = (
            self._step % self.assign_interval == 0
            or (self._event and self._event.force)
            or (not bool(prev_valid.all()))
        )
        if needs_reconfig:
            hard, E_c, terms = solve_membership(
                plan, self._assignment,
                switch_penalty=self.switch_penalty,
                count_penalty=self.count_penalty,
            )
            result = membership_gate(
                plan, self._assignment, hard, E_c, self._prev_energy, terms,
                prev_valid=prev_valid,
                dwell_steps_since_change=self._dwell,
                capture_hold_active=capture_hold_active(env),
                event=self._event,
                hysteresis=self.hysteresis,
                min_dwell_steps=self.min_dwell_steps,
            )
            if result.accepted or result.reason == "invalid_fixed":
                apply_membership(env, result.assignment)
                self._assignment = result.assignment.copy()
                self._dwell = 0
                self._prev_energy = result.candidate_energy
                reconfigured = True
            else:
                self._dwell += 1
            gate_reason = result.reason
            self._event = None
        else:
            self._dwell += 1

        # 4) 同一 H + 接受后归属 → V2 观测 → Actor
        enc2 = self.encoder.encode(env.snapshot(), differentiable=False)   # 缓存命中
        obs, mask = self._build_v2_obs(enc2, self._assignment, env)
        with torch.no_grad():
            acts = self.actor(obs)
        actions = acts[:n_p].cpu().numpy()

        info = {
            "architecture_version": ARCHITECTURE_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "allocation_source": ALLOCATION_SOURCE,
            "demand_raw": demand[:n_ev].detach().cpu().numpy().tolist(),
            "capacity_task": plan.capacity_task[:n_ev].detach().cpu().numpy().tolist(),
            "capacity_idle": float(plan.capacity_idle),
            "demand_scale_alpha": float(plan.demand_scale_alpha),
            "row_residual_max": float(plan.row_residual.abs().max()),
            "col_residual_max": float(plan.col_residual.abs().max()),
            "assignment": self._assignment[:n_p].tolist(),
            "reconfigured": reconfigured,
            "gate_reason": gate_reason,
            "dwell": self._dwell,
        }
        self.last_info = info
        self._step += 1
        return actions, info
