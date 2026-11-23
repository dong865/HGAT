"""continuous_membership_v2: 纯算法组件（阶段 B 交付物）。

包含第 3–8 节规定的：连续需求头、统一容量构造、正代价 Sinkhorn 软计划、
显式任务 ID 契约、统一软/硬上下文聚合、凸人数偏差代价的线性指派硬配置、
以及不依赖角色/阶段的稳定门控。

设计约束：
- 连续值只用于需求与软优化；硬配置始终是整数一人一归属。
- 所有模式（含固定需求消融）共用同一容量函数与同一求解器。
- Sinkhorn 内部 float64 计算但保留 autograd；不做 numpy 转换。
- 空目的地是数学上的未分配目的地，不是功能角色类别。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------

ASSIGNMENT_UNASSIGNED = -1  # env.assignment[i] == -1 表示未分配
EMPTY_DESTINATION = "empty"  # 数学符号；padded full_plan 最后一列承载
DEFAULT_EPSILON_OT = 0.5
DEFAULT_SINKHORN_ITERS = 80
DEFAULT_COUNT_PENALTY = 1.0
DEFAULT_SWITCH_PENALTY = 0.5
DEFAULT_HYSTERESIS = 0.05
DEFAULT_MIN_DWELL_STEPS = 5
PLAN_EPS = 1e-8  # log 计划的数值地板（只用于硬评分，不修改软 OT 边际）


# ---------------------------------------------------------------------------
# 数据结构（第 4 节）
# ---------------------------------------------------------------------------


@dataclass
class TaskPlan:
    """软运输计划与容量。所有 padded 张量按全局 ID 还原。"""

    task_plan: torch.Tensor            # [Np_max, Nt_max] 只含真实任务列
    full_plan: torch.Tensor            # [Np_max, Nt_max + 1]，最后一列=未分配
    demand_raw: torch.Tensor           # [Nt_max]，非活跃任务 = 0
    capacity_task: torch.Tensor        # [Nt_max]，非活跃任务 = 0
    capacity_idle: torch.Tensor        # 标量
    available_agent_mask: torch.Tensor
    active_task_mask: torch.Tensor
    active_agent_ids: torch.Tensor
    active_task_ids: torch.Tensor
    row_residual: torch.Tensor         # 行边际残差（应≈0）
    col_residual: torch.Tensor         # 列边际残差（应≈0）
    demand_scale_alpha: torch.Tensor   # 供给缩放系数

    @property
    def num_agents(self) -> int:
        return int(self.available_agent_mask.sum().item())

    @property
    def num_tasks(self) -> int:
        return int(self.active_task_mask.sum().item())


@dataclass
class MembershipPlan:
    """硬配置执行结果。"""

    assignment: np.ndarray             # [Np_max]，全局任务 ID 或 -1
    hard_full_plan: np.ndarray         # [Np_max, Nt_max+1]，非可用行=0
    task_counts: np.ndarray            # [Nt_max]
    accepted: bool
    reason: str
    candidate_energy: float
    previous_energy: float
    energy_terms: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 连续需求头（第 3 节）
# ---------------------------------------------------------------------------


def softplus_inverse(x: float) -> float:
    return float(math.log(math.expm1(x))) if x > 20 else float(np.log(np.expm1(x)))


class ContinuousDemandHead(nn.Module):
    """每个活跃任务一个非负连续标量需求。

    语义为状态相关的期望资源量/分配偏好；不预设任何人数档位。
    bias 以 inverse-softplus(demand_init) 初始化，使初始需求为统一正值。
    """

    def __init__(self, hidden_dim: int, demand_init: float = 3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        # 初始化契约：末层权重清零 + bias=softplus^{-1}(demand_init)，
        # 使任意输入下初始需求严格等于 demand_init（权重可随后学习）。
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.fill_(softplus_inverse(float(demand_init)))
        self.demand_init = float(demand_init)

    def forward(self, task_embeddings: torch.Tensor,
                task_mask: torch.Tensor) -> torch.Tensor:
        raw = self.net(task_embeddings).squeeze(-1)
        demand = F.softplus(raw)
        return demand * task_mask.to(demand)


# ---------------------------------------------------------------------------
# 统一容量构造（第 5.1 节）
# ---------------------------------------------------------------------------


def build_capacity(demand: torch.Tensor,
                   available_agent_mask: torch.Tensor,
                   active_task_mask: torch.Tensor,
                   dtype: torch.dtype = torch.float64
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """alpha = min(1, N / sum d_j)；b_j = alpha d_j；b_empty = N - sum b_j。

    所有模式（含固定需求消融）必须调用此同一函数。
    返回 (alpha, capacity_task[padded], capacity_idle 标量, 活跃任务需求)。
    """
    d = demand.to(dtype)
    task_mask = active_task_mask.to(dtype)
    agent_mask = available_agent_mask.to(dtype)

    n_avail = agent_mask.sum()
    demand_active = d * task_mask
    total_demand = demand_active.sum().clamp_min(1e-12)

    alpha = torch.clamp(n_avail / total_demand, max=1.0)
    capacity_task = alpha * demand_active
    capacity_idle = n_avail - capacity_task.sum()
    # 数值保护：不改变总质量恒等式
    capacity_idle = torch.clamp(capacity_idle, min=0.0)
    return alpha, capacity_task, capacity_idle, demand_active


# ---------------------------------------------------------------------------
# 正代价 Sinkhorn 软计划（第 5.2–5.4 节）
# ---------------------------------------------------------------------------


def build_soft_plan(costs: torch.Tensor,
                    demand: torch.Tensor,
                    available_agent_mask: torch.Tensor,
                    active_task_mask: torch.Tensor,
                    epsilon: float = DEFAULT_EPSILON_OT,
                    iters: int = DEFAULT_SINKHORN_ITERS
                    ) -> TaskPlan:
    """求解第 5.3 节运输问题。

    costs: [Np_max, Nt_max] 正向代价（越大越差），未压缩 padded。
    内部以活跃子集压缩、float64 计算，返回前按全局 ID 还原。
    关系代价语义：未来距离越大代价越高；此处收正代价，不再取负。
    """
    device = costs.device
    orig_dtype = costs.dtype
    a_mask = available_agent_mask.to(torch.bool)
    t_mask = active_task_mask.to(torch.bool)
    n_max, m_max = costs.shape

    task_plan = costs.new_zeros((n_max, m_max))
    full_plan = costs.new_zeros((n_max, m_max + 1))
    zero_cap = torch.zeros((), dtype=torch.float64, device=device)
    row_residual = torch.zeros(n_max, dtype=torch.float64, device=device)
    col_residual = torch.zeros(m_max + 1, dtype=torch.float64, device=device)

    active_i = torch.nonzero(a_mask, as_tuple=False).squeeze(-1)
    active_j = torch.nonzero(t_mask, as_tuple=False).squeeze(-1)
    alpha, cap_task, cap_idle, demand_active = build_capacity(
        demand, available_agent_mask, active_task_mask
    )

    n_t = int(active_i.numel())
    m_t = int(active_j.numel())

    if n_t == 0:
        return TaskPlan(task_plan, full_plan, demand_active.to(orig_dtype),
                        cap_task.to(orig_dtype), cap_idle.to(orig_dtype),
                        available_agent_mask, active_task_mask,
                        active_i, active_j, row_residual, col_residual,
                        alpha)

    # ---- 压缩到活跃子集 ----
    C = costs[active_i][:, active_j].to(torch.float64)          # [n_t, m_t]
    d_active = demand_active[active_j]                          # [m_t]
    b_task = cap_task[active_j]                                 # [m_t]
    b_empty = cap_idle.to(torch.float64)

    if m_t == 0:
        # 无活跃任务：每个可用成员的空目的地质量为 1
        full = task_plan.new_zeros((n_t, 1), dtype=torch.float64)
        full[:, 0] = 1.0
        full_plan[active_i, m_max] = full[:, 0].to(orig_dtype)
        row_residual[active_i] = 0.0
        col_residual[m_max] = float(n_t) - 1.0 * n_t
        return TaskPlan(task_plan, full_plan, demand_active.to(orig_dtype),
                        cap_task.to(orig_dtype), cap_idle.to(orig_dtype),
                        available_agent_mask, active_task_mask,
                        active_i, active_j, row_residual, col_residual, alpha)

    # 第 5.2 节：C_bar = [C | 0]（空目的地零代价）
    C_bar = torch.cat([C, torch.zeros((n_t, 1), dtype=torch.float64, device=device)], dim=1)
    col_dem = torch.cat([b_task, b_empty.reshape(1)])           # [m_t+1]

    # ---- log-domain Sinkhorn ----
    # mask：所有同质可用成员可流向每个活跃任务与空目的地（首版无禁边）
    valid = torch.ones_like(C_bar, dtype=torch.bool)
    log_K = torch.where(valid, -C_bar / float(epsilon),
                        torch.full_like(C_bar, -float("inf")))
    log_dem = torch.log(col_dem.clamp_min(1e-15))
    log_supply = torch.log(torch.ones(n_t, dtype=torch.float64, device=device))

    u = torch.zeros(n_t, dtype=torch.float64, device=device)
    v = torch.zeros(m_t + 1, dtype=torch.float64, device=device)
    # 自适应预算：先按 iters 跑（含 1e-9 早停）；若残差超差且未达硬上限则续跑，
    # 保证守恒精度与"直接跑满上限"逐位一致，常见情形白赚速度。
    HARD_CAP = max(int(iters), 300)
    for _ in range(HARD_CAP):
        u_prev = u.clone()
        u = log_supply - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_dem - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
        if torch.max(torch.abs(u - u_prev)) < 1e-9:
            break

    log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    P_bar = torch.exp(log_P)
    P_bar = torch.where(torch.isfinite(P_bar), P_bar, torch.zeros_like(P_bar))
    # 零容量列保持为零列
    P_bar = torch.where((col_dem > 0).unsqueeze(0), P_bar, torch.zeros_like(P_bar))

    # ---- 残差记录（不静默掩盖失败）----
    r_row = P_bar.sum(dim=1) - 1.0
    r_col = P_bar.sum(dim=0) - col_dem
    if (not torch.all(torch.isfinite(P_bar))) or \
       float(r_row.abs().max()) > 1e-3 or float(r_col.abs().max()) > 1e-3:
        raise FloatingPointError(
            f"soft plan conservation violated: row_max={float(r_row.abs().max()):.3g} "
            f"col_max={float(r_col.abs().max()):.3g}; refusing to mask a failed plan"
        )

    # ---- 还原全局 ID（显式行列索引，避免广播歧义）----
    P_real = P_bar[:, :m_t].to(orig_dtype)
    p_empty = P_bar[:, m_t].to(orig_dtype)
    task_plan[active_i[:, None], active_j[None, :]] = P_real
    full_plan[:, :m_max] = task_plan
    full_plan[active_i, m_max] = p_empty

    row_residual[active_i] = r_row
    col_residual[active_j] = r_col[:m_t]
    col_residual[m_max] = r_col[m_t]

    return TaskPlan(task_plan, full_plan, demand_active.to(orig_dtype),
                    cap_task.to(orig_dtype), cap_idle.to(orig_dtype),
                    available_agent_mask, active_task_mask,
                    active_i, active_j, row_residual, col_residual, alpha)


# ---------------------------------------------------------------------------
# 统一上下文聚合（第 6.2 节）
# ---------------------------------------------------------------------------


def build_membership_context(K: torch.Tensor,
                             pursuer_emb: torch.Tensor,
                             task_emb: torch.Tensor,
                             epsilon_c: float = 1e-6
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """同一聚合函数分别用于软计划 K=P 与硬配置 K=X。

    K: [Np_max, Nt_max] 真实任务列（不含空列）。
    返回 (t_context, c_context)，未分配成员（K 行为零）的两块上下文为零。
    """
    T = K @ task_emb                                   # [Np, H] 任务上下文
    W = K @ K.transpose(0, 1)                          # [Np, Np] 亲和
    denom = W.sum(dim=-1, keepdim=True).clamp_min(float(epsilon_c))
    C = (W @ pursuer_emb) / denom
    return T, C


# ---------------------------------------------------------------------------
# 凸人数偏差的线性指派硬求解（第 7 节）
# ---------------------------------------------------------------------------


def _assignment_energy(full_plan: torch.Tensor,
                       hard_full: np.ndarray,
                       prev_assignment: np.ndarray,
                       capacity_task: np.ndarray,
                       capacity_idle: float,
                       switch_penalty: float,
                       count_penalty: float,
                       plan_eps: float = PLAN_EPS) -> Tuple[float, dict]:
    """第 7.1 节能量 E(X) 及其分项。"""
    P = np.clip(full_plan.astype(np.float64), plan_eps, None)
    X = hard_full.astype(np.float64)
    log_term = float(-(X * np.log(P)).sum())
    switch = 0.0
    for i in range(X.shape[0]):
        chosen = int(np.argmax(X[i]))
        prev = int(prev_assignment[i]) if i < len(prev_assignment) else ASSIGNMENT_UNASSIGNED
        # 列约定：真实任务 j -> 列 j；未分配 -> 最后一列
        is_empty_col = (chosen == X.shape[1] - 1)
        dest = ASSIGNMENT_UNASSIGNED if is_empty_col else chosen
        if dest != prev:
            switch += float(switch_penalty)
    counts = X.sum(axis=0)
    caps = np.append(capacity_task, capacity_idle)
    count_dev = float(count_penalty) * float(((counts - caps) ** 2).sum())
    return log_term + switch + count_dev, {
        "log_plan": log_term, "switch": switch, "count_deviation": count_dev,
    }


def solve_membership(plan: TaskPlan,
                     prev_assignment: np.ndarray,
                     switch_penalty: float = DEFAULT_SWITCH_PENALTY,
                     count_penalty: float = DEFAULT_COUNT_PENALTY,
                     plan_eps: float = PLAN_EPS
                     ) -> Tuple[np.ndarray, float, dict]:
    """带凸人数偏差代价的线性指派（第 7.2 节）。

    对目的地 j 与单位计数 k=1..N 构造边际代价
        Delta_j(k) = lambda_n * (2k - 1 - 2 b_j)
    A[i, (j,k)] = -log(max(P_bar_ij, eps)) + switch_cost(i,j) + Delta_j(k)。
    k 是硬求解器内部单位计数，不是功能档位、不在可微图内。
    返回 (hard_full_plan[padded], energy, energy_terms)。
    """
    device = plan.full_plan.device
    n_max, m1 = plan.full_plan.shape
    m_max = m1 - 1
    active_i = plan.active_agent_ids.cpu().numpy()
    active_j = plan.active_task_ids.cpu().numpy()
    n_t, m_t = len(active_i), len(active_j)

    hard = np.zeros((n_max, m1), dtype=np.float64)
    if n_t == 0:
        return hard, 0.0, {"log_plan": 0.0, "switch": 0.0, "count_deviation": 0.0}

    P_bar = plan.full_plan[active_i].detach().cpu().numpy().astype(np.float64)
    P_bar = np.clip(P_bar, plan_eps, None)

    b_task = plan.capacity_task[active_j].detach().cpu().numpy().astype(np.float64)
    b_empty = float(plan.capacity_idle.detach().cpu().numpy())
    b_all = np.append(b_task, b_empty)

    prev = np.asarray(prev_assignment).astype(np.int64)

    # 目的地在压缩列空间中的索引：真实任务 -> 0..m_t-1，空 -> m_t
    n_cols = (m_t + 1) * n_t
    A = np.zeros((n_t, n_cols), dtype=np.float64)
    for i_local, i_global in enumerate(active_i):
        prev_dest = prev[i_global] if i_global < len(prev) else ASSIGNMENT_UNASSIGNED
        for j_local in range(m_t + 1):
            j_global = active_j[j_local] if j_local < m_t else -1
            switch_cost = 0.0 if j_global == prev_dest else float(switch_penalty)
            base = -math.log(P_bar[i_local, j_local]) + switch_cost
            for k in range(1, n_t + 1):
                delta = float(count_penalty) * (2 * k - 1 - 2 * b_all[j_local])
                A[i_local, j_local * n_t + (k - 1)] = base + delta

    row_ind, col_ind = linear_sum_assignment(A)
    energy_terms = {"log_plan": 0.0, "switch": 0.0, "count_deviation": 0.0}
    for i_local, col in zip(row_ind, col_ind):
        j_local = col // n_t
        hard[active_i[i_local], j_local if j_local < m_t else m_max] = 1.0
        energy_terms["log_plan"] += -math.log(P_bar[i_local, j_local])
        j_global = active_j[j_local] if j_local < m_t else ASSIGNMENT_UNASSIGNED
        i_global = active_i[i_local]
        prev_dest = prev[i_global] if i_global < len(prev) else ASSIGNMENT_UNASSIGNED
        if j_global != prev_dest:
            energy_terms["switch"] += float(switch_penalty)
    # 只在活跃任务列 + 空列上计算人数偏差（非活跃 padding 列不计入）
    comp_cols = list(active_j) + [m_max]
    counts = hard[active_i][:, comp_cols].sum(axis=0)
    caps = np.append(b_task, b_empty)
    energy_terms["count_deviation"] = float(count_penalty) * float(((counts - caps) ** 2).sum())
    energy = sum(energy_terms.values())
    return hard, energy, energy_terms


def hard_plan_to_assignment(hard_full_plan: np.ndarray) -> np.ndarray:
    """[Np, Nt+1] 二进制 full plan -> assignment 全局 ID（未分配 = -1）。"""
    n, m1 = hard_full_plan.shape
    assignment = np.full(n, ASSIGNMENT_UNASSIGNED, dtype=np.int64)
    for i in range(n):
        cols = np.nonzero(hard_full_plan[i] > 0.5)[0]
        if len(cols):
            assignment[i] = int(cols[0]) if cols[0] < m1 - 1 else ASSIGNMENT_UNASSIGNED
    return assignment


# ---------------------------------------------------------------------------
# 稳定门控（第 8 节）
# ---------------------------------------------------------------------------


@dataclass
class GateEvent:
    """强制重构事件（停滞等）。"""
    force: bool = False
    reason: str = ""


def membership_gate(plan: TaskPlan,
                    prev_assignment: np.ndarray,
                    candidate_hard: np.ndarray,
                    candidate_energy: float,
                    prev_energy: float,
                    energy_terms: dict,
                    prev_valid: np.ndarray,          # 旧配置各成员归属是否仍合法
                    dwell_steps_since_change: int,
                    capture_hold_active: bool,       # 当前成员集正满足捕获保持
                    event: Optional[GateEvent] = None,
                    hysteresis: float = DEFAULT_HYSTERESIS,
                    min_dwell_steps: int = DEFAULT_MIN_DWELL_STEPS
                    ) -> MembershipPlan:
    """第 8 节门控，优先级（高→低）：

    1. 失效归属修复：成员离线 / 指向已完成或失活任务的归属立即修复，
       不受驻留与捕获保持阻挡（失活的捕获保持任务走此分支，而不是被保持）。
    2. 强制重构事件（停滞等）：越过普通驻留/滞回，但不能无条件打破
       仍然合法的捕获保持之前的失效检查（上一条已处理失活）。
    3. 捕获保持：当前实际成员集满足物理保持条件时保留配置（无角色重写）。
    4. 普通接受：驻留期满且能量改善超过滞回。
    """
    prev_assignment = np.asarray(prev_assignment).astype(np.int64)
    cand_assign = hard_plan_to_assignment(candidate_hard)
    invalid = (~np.asarray(prev_valid, dtype=bool)) if prev_valid is not None \
        else np.zeros(len(prev_assignment), dtype=bool)
    invalid_idx = np.nonzero(invalid)[0]
    has_invalid = int(invalid_idx.size) > 0

    def _result(assign, hard, accepted, reason):
        return MembershipPlan(
            assignment=assign, hard_full_plan=hard,
            task_counts=_task_counts(assign, plan),
            accepted=accepted, reason=reason,
            candidate_energy=candidate_energy, previous_energy=prev_energy,
            energy_terms=energy_terms,
        )

    # 1) 失效归属立即修复：仅替换失效成员的归属为候选目的地，其余保持
    if has_invalid:
        fixed = prev_assignment.copy()
        for i in invalid_idx:
            fixed[i] = cand_assign[i]
        return _result(fixed, _assignment_to_hard_full(fixed, plan), True,
                       "invalid_fixed")

    # 2) 强制事件越过驻留/滞回（force 不遮蔽上一条的失效语义）
    if event is not None and event.force:
        return _result(cand_assign, candidate_hard, True,
                       f"force:{event.reason}")

    # 3) 捕获保持（仅当旧配置完全合法时可达此分支）
    if capture_hold_active:
        return _result(prev_assignment.copy(),
                       _assignment_to_hard_full(prev_assignment, plan),
                       False, "capture_hold")

    # 4) 普通接受
    improvement = prev_energy - candidate_energy
    if dwell_steps_since_change >= min_dwell_steps and improvement > hysteresis:
        return _result(cand_assign, candidate_hard, True, "improved")
    reason = "dwell" if dwell_steps_since_change < min_dwell_steps else "hysteresis"
    return _result(prev_assignment.copy(),
                   _assignment_to_hard_full(prev_assignment, plan),
                   False, reason)


def _assignment_to_hard_full(assignment: np.ndarray, plan: TaskPlan) -> np.ndarray:
    n_max = plan.full_plan.shape[0]
    m_max = plan.full_plan.shape[1] - 1
    hard = np.zeros((n_max, m_max + 1), dtype=np.float64)
    for i, a in enumerate(assignment):
        if a == ASSIGNMENT_UNASSIGNED:
            hard[i, m_max] = 1.0
        elif 0 <= a < m_max:
            hard[i, a] = 1.0
    return hard


def _task_counts(assignment: np.ndarray, plan: TaskPlan) -> np.ndarray:
    m_max = plan.full_plan.shape[1] - 1
    counts = np.zeros(m_max, dtype=np.int64)
    for a in assignment:
        if 0 <= a < m_max:
            counts[a] += 1
    return counts
