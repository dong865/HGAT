"""continuous_membership_v2 验收测试（交接文档第 15 节 A 组 + B 组有限差分）。

运行：python -m crazyflie_examples.six_v_two_hgat.test_continuous_membership
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if (_HERE / "continuous_membership.py").exists():
    sys.path.insert(0, str(_HERE.parent))

from crazyflie_examples.six_v_two_hgat.continuous_membership import (
    ASSIGNMENT_UNASSIGNED,
    ContinuousDemandHead,
    build_capacity,
    build_membership_context,
    build_soft_plan,
    hard_plan_to_assignment,
    membership_gate,
    solve_membership,
    GateEvent,
)

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------------------
# A1 代价方向：低 cost 匹配获得更大软质量（非退化 2x2）
# ---------------------------------------------------------------------------

def test_cost_direction():
    costs = torch.tensor([[0.1, 1.0], [1.0, 0.1]])       # 对角优
    demand = torch.tensor([2.0, 2.0])
    a_mask = torch.tensor([True, True])
    t_mask = torch.tensor([True, True])
    plan = build_soft_plan(costs, demand, a_mask, t_mask, epsilon=0.05, iters=200)
    P = plan.task_plan
    check("A1 cost direction", P[0, 0] > 0.6 and P[1, 1] > 0.6,
          f"P={P.numpy().round(3).tolist()}")


# ---------------------------------------------------------------------------
# A2 OT 守恒：供给充足/不足/空任务/零成员/非活跃 padding/空目的地零容量
# ---------------------------------------------------------------------------

def test_conservation():
    ok = True
    # 充足
    p = build_soft_plan(torch.rand(6, 3) * 0.3, torch.full((3,), 2.0),
                        torch.ones(6, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    ok &= float(p.row_residual.abs().max()) < 1e-3 and float(p.col_residual.abs().max()) < 1e-3
    # 不足（需求总和 > N）
    p = build_soft_plan(torch.rand(4, 2) * 0.3, torch.full((2,), 3.0),
                        torch.ones(4, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    ok &= abs(float(p.full_plan.sum()) - 4.0) < 1e-3
    ok &= float(p.row_residual.abs().max()) < 1e-3
    # 空任务（M=0, N>0）：质量全进空目的地
    p = build_soft_plan(torch.rand(4, 3), torch.zeros(3),
                        torch.ones(4, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    ok &= abs(float(p.full_plan[:, -1].sum()) - 4.0) < 1e-6
    # 零成员（N=0）
    p = build_soft_plan(torch.rand(6, 2), torch.ones(2),
                        torch.zeros(6, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    ok &= float(p.full_plan.abs().sum()) == 0.0
    # 非活跃 padding：非活跃列保持零
    costs = torch.rand(6, 3) * 0.3
    p = build_soft_plan(costs, torch.tensor([2.0, 2.0, 5.0]),
                        torch.ones(6, dtype=torch.bool),
                        torch.tensor([True, True, False]))
    ok &= float(p.task_plan[:, 2].abs().sum()) == 0.0
    ok &= float(p.capacity_task[2]) == 0.0
    # 空目的地零容量情形：b_empty=0 时空列保持为零
    p = build_soft_plan(costs[:, :2], torch.tensor([3.0, 3.0]),
                        torch.ones(6, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    ok &= float(p.full_plan[:, -1].abs().sum()) < 1e-6
    check("A2 OT conservation / edge cases", ok)


# ---------------------------------------------------------------------------
# A3 全局 ID：活动任务 [1] / [1,2] 等非连续编号不错列
# ---------------------------------------------------------------------------

def test_global_ids():
    ok = True
    costs = torch.rand(6, 3) * 0.3
    demand = torch.tensor([0.0, 2.0, 0.0])
    p = build_soft_plan(costs, demand, torch.ones(6, dtype=torch.bool),
                        torch.tensor([False, True, False]))
    ok &= p.active_task_ids.tolist() == [1]
    ok &= float(p.task_plan[:, 0].abs().sum()) == 0.0
    ok &= float(p.task_plan[:, 2].abs().sum()) == 0.0
    ok &= float(p.task_plan[:, 1].sum()) + float(p.full_plan[:, -1].sum()) > 5.9

    demand = torch.tensor([0.0, 2.0, 3.0])
    p = build_soft_plan(costs, demand, torch.ones(6, dtype=torch.bool),
                        torch.tensor([False, True, True]))
    ok &= p.active_task_ids.tolist() == [1, 2]
    ok &= float(p.capacity_task[0]) == 0.0 and float(p.capacity_task[1]) > 0
    # 还原矩阵的形状与压缩一致
    ok &= p.task_plan.shape == (6, 3) and p.full_plan.shape == (6, 4)
    check("A3 global ID mapping", ok)


# ---------------------------------------------------------------------------
# A4 资源竞争：N<M 时返回合法方案，允许任务未服务
# ---------------------------------------------------------------------------

def test_resource_competition():
    costs = torch.rand(2, 4) * 0.3
    demand = torch.ones(4)
    p = build_soft_plan(costs, demand, torch.ones(2, dtype=torch.bool),
                        torch.ones(4, dtype=torch.bool))
    ok = abs(float(p.full_plan.sum()) - 2.0) < 1e-3
    hard, energy, terms = solve_membership(p, np.array([-1, -1]))
    assign = hard_plan_to_assignment(hard)
    ok &= all((a == -1) or (0 <= a < 4) for a in assign)
    ok &= (assign >= 0).sum() <= 2
    check("A4 resource competition (N<M)", ok, f"assignment={assign.tolist()}")


# ---------------------------------------------------------------------------
# A5 硬优化目标：小规模穷举 vs 凸计数线性指派
# ---------------------------------------------------------------------------

def _energy_bruteforce(P_bar, X, prev, b_all, lam_sw, lam_n, eps=1e-8):
    E = 0.0
    for i in range(X.shape[0]):
        j = int(np.argmax(X[i]))
        E += -math_log(max(P_bar[i, j], eps))
        dest = -1 if j == X.shape[1] - 1 else j
        if dest != prev[i]:
            E += lam_sw
    counts = X.sum(axis=0)
    E += lam_n * float(((counts - b_all) ** 2).sum())
    return E


def math_log(x):
    import math
    return math.log(x)


def test_hard_solver_optimality():
    import math as _m
    global math_log
    math_log = _m.log
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(320):
        n = int(rng.integers(1, 5))
        m = int(rng.integers(0, 4))
        P_real = rng.uniform(1e-3, 1.0, size=(n, m + 1))
        P_real /= P_real.sum(axis=1, keepdims=True)
        b_task = rng.uniform(0.0, n, size=m)
        b_empty = max(0.0, n - b_task.sum())
        prev = rng.integers(-1, m, size=n)
        lam_sw, lam_n = float(rng.uniform(0, 1.5)), float(rng.uniform(0.1, 2.0))

        costs_t = torch.tensor(P_real[:, :m] if m else np.zeros((n, 0)))
        demand_t = torch.ones(m)
        p = build_soft_plan(
            torch.tensor(rng.uniform(0, 0.5, size=(n, max(m, 1)))[:, :m] if m else np.zeros((n, 1)))[:, :m] if m else torch.zeros((n, 0)),
            demand_t,
            torch.ones(n, dtype=torch.bool),
            torch.ones(m, dtype=torch.bool) if m else torch.zeros(0, dtype=torch.bool),
        )
        # 直接用受控 P_bar 覆盖（绕过 build_soft_plan 的容量耦合，专注求解器正确性）
        p.full_plan = torch.tensor(P_real)
        p.capacity_task = torch.tensor(b_task)
        p.capacity_idle = torch.tensor(b_empty)
        p.active_task_ids = torch.arange(m)

        hard, energy, _ = solve_membership(p, prev, lam_sw, lam_n)
        # 穷举全部 (m+1)^n 配置
        best = float("inf")
        for combo in itertools.product(range(m + 1), repeat=n):
            X = np.zeros((n, m + 1))
            for i, j in enumerate(combo):
                X[i, j] = 1.0
            E = _energy_bruteforce(P_real, X, prev,
                                   np.append(b_task, b_empty), lam_sw, lam_n)
            best = min(best, E)
        if abs(energy - best) > 1e-6 * max(1.0, abs(best)):
            ok = False
            print(f"  mismatch trial={trial} n={n} m={m} solver={energy:.6f} brute={best:.6f}")
            break
    check("A5 hard solver == brute force (320 cases)", ok)


# ---------------------------------------------------------------------------
# A6 二进制上下文一致性：同一 X 走统一聚合，置换等变
# ---------------------------------------------------------------------------

def test_context_consistency():
    H = torch.randn(4, 8)
    T = torch.randn(2, 8)
    X = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # 0,1->task0; 2->task1; 3 idle
    t_ctx, c_ctx = build_membership_context(X, H, T)
    ok = torch.allclose(c_ctx[0], (H[0] + H[1]) / 2, atol=1e-6)      # 同任务均值
    ok &= torch.allclose(c_ctx[2], H[2], atol=1e-6)                   # 单人任务=自身
    ok &= torch.allclose(c_ctx[3], torch.zeros(8), atol=1e-6)         # 未分配=零上下文
    # 置换等变：重排成员顺序，输出应相应置换
    perm = torch.tensor([2, 0, 3, 1])
    t2, c2 = build_membership_context(X[perm], H[perm], T)
    ok &= torch.allclose(t2, t_ctx[perm], atol=1e-6) and torch.allclose(c2, c_ctx[perm], atol=1e-6)
    # 软计划同一函数
    P = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.0, 0.0]])
    t3, c3 = build_membership_context(P, H, T)
    ok &= c3.shape == c_ctx.shape and t3.shape == t_ctx.shape
    check("A6 unified context consistency + permutation", ok)


# ---------------------------------------------------------------------------
# A7 梯度开关不改变前向：soft+grad vs soft+no_grad 特征一致
# ---------------------------------------------------------------------------

def test_grad_switch_forward():
    torch.manual_seed(0)
    head = ContinuousDemandHead(16, demand_init=3.0)
    emb = torch.randn(3, 16)
    mask = torch.tensor([True, True, False])
    d1 = head(emb, mask)
    d2 = head(emb.detach(), mask)
    demand_list = [round(float(v), 3) for v in d1.detach()]
    check("A7 grad switch same forward", torch.equal(d1, d2), "demand=%s" % demand_list)
    d0 = head(torch.zeros(1, 16), torch.ones(1, dtype=torch.bool))
    ok = bool((d1 >= 0).all()) and float(d1[2]) == 0.0
    ok &= abs(float(d0[0]) - 3.0) < 1e-3  # 末层权重零初始化：任意输入严格等于 demand_init
    check("A7 demand head init/nonneg/mask", ok)


# ---------------------------------------------------------------------------
# B-有限差分：costs 与 demand 的 Sinkhorn 梯度（固定迭代、float64）
# ---------------------------------------------------------------------------

def test_finite_difference():
    torch.manual_seed(1)
    n, m = 3, 2
    costs = (torch.rand(n, m, dtype=torch.float64) * 0.3).requires_grad_(True)
    demand = torch.tensor([2.3, 1.7], dtype=torch.float64).requires_grad_(True)
    masks_a = torch.ones(n, dtype=torch.bool)
    masks_t = torch.ones(m, dtype=torch.bool)
    plan = build_soft_plan(costs, demand, masks_a, masks_t, epsilon=0.1, iters=80)
    obj = (plan.full_plan ** 2).sum()
    obj.backward()
    eps_fd = 1e-6
    ok_c, ok_d = True, True
    for idx in [(0, 0), (2, 1)]:
        c2 = costs.detach().clone(); c2[idx] += eps_fd
        p2 = build_soft_plan(c2, demand.detach(), masks_a, masks_t, epsilon=0.1, iters=80)
        fd = ((p2.full_plan ** 2).sum() - obj.detach()) / eps_fd
        if abs(fd - costs.grad[idx]) > 1e-3 * max(1.0, abs(float(fd))):
            ok_c = False
    for idx in [0, 1]:
        d2 = demand.detach().clone(); d2[idx] += eps_fd
        p2 = build_soft_plan(costs.detach(), d2, masks_a, masks_t, epsilon=0.1, iters=80)
        fd = ((p2.full_plan ** 2).sum() - obj.detach()) / eps_fd
        if abs(fd - demand.grad[idx]) > 1e-3 * max(1.0, abs(float(fd))):
            ok_d = False
    check("B finite-diff d(obj)/d(costs)", ok_c)
    check("B finite-diff d(obj)/d(demand)", ok_d)


# ---------------------------------------------------------------------------
# 门控基本行为（A9 前置）
# ---------------------------------------------------------------------------

def test_gate_basics():
    n, m = 3, 1
    p = build_soft_plan(torch.rand(n, m) * 0.3, torch.tensor([2.0]),
                        torch.ones(n, dtype=torch.bool), torch.ones(m, dtype=torch.bool))
    hard, E_c, terms = solve_membership(p, np.array([-1, -1, -1]))
    prev = np.array([0, 0, -1])
    E_p, _ = _energy_of_prev(prev, p)
    # 驻留不足 -> 拒绝
    r = membership_gate(p, prev, hard, E_c, E_p, terms,
                        np.array([True, True, True]), dwell_steps_since_change=1,
                        capture_hold_active=False, min_dwell_steps=5)
    ok = (not r.accepted) and r.reason == "dwell"
    # 驻留足够但改善不足 -> 滞回拒绝
    r = membership_gate(p, prev, hard, E_c, E_p, terms,
                        np.array([True, True, True]), dwell_steps_since_change=10,
                        capture_hold_active=False, hysteresis=1e9, min_dwell_steps=5)
    ok &= (not r.accepted) and r.reason == "hysteresis"
    # 强制事件越过
    r = membership_gate(p, prev, hard, E_c, E_p, terms,
                        np.array([True, True, True]), dwell_steps_since_change=0,
                        capture_hold_active=False, event=GateEvent(force=True, reason="stagnation"),
                        min_dwell_steps=999)
    ok &= r.accepted and r.reason.startswith("force")
    # 捕获保持保留旧配置
    r = membership_gate(p, prev, hard, E_c, E_p, terms,
                        np.array([True, True, True]), dwell_steps_since_change=99,
                        capture_hold_active=True, min_dwell_steps=0)
    ok &= (not r.accepted) and r.reason == "capture_hold" and r.assignment.tolist() == [0, 0, -1]
    check("gate dwell/hysteresis/force/capture-hold", ok)




def test_gate_combined_conditions():
    """审查组合条件：失效修复 vs 驻留/捕获保持/强制事件的优先级。"""
    n, m = 4, 2
    p = build_soft_plan(torch.rand(n, m) * 0.3, torch.tensor([2.0, 2.0]),
                        torch.ones(n, dtype=torch.bool), torch.ones(m, dtype=torch.bool))
    hard, E_c, terms = solve_membership(p, np.array([-1, -1, -1, -1]))
    cand = hard_plan_to_assignment(hard)

    # a) 驻留期未结束 + 成员 1 离线（归属失效） -> 立即修复该成员，不等驻留
    prev = np.array([0, 0, 1, -1])
    E_p, _ = _energy_of_prev(prev, p)
    prev_valid = np.array([True, False, True, True])   # 成员1离线
    r = membership_gate(p, prev, hard, E_c, E_p, terms, prev_valid,
                        dwell_steps_since_change=1, capture_hold_active=False,
                        min_dwell_steps=5)
    ok = r.accepted and r.reason == "invalid_fixed" and r.assignment[1] == cand[1]
    ok &= r.assignment[0] == 0 and r.assignment[2] == 1  # 其余成员保持

    # b) 捕获保持开启 + 目标 0 已失活（其成员归属失效）-> 失效修复胜过捕获保持
    prev = np.array([0, 0, 1, -1])
    prev_valid = np.array([False, False, True, True])  # 任务0失活 -> 成员0,1归属失效
    r = membership_gate(p, prev, hard, E_c, E_p, terms, prev_valid,
                        dwell_steps_since_change=99, capture_hold_active=True,
                        min_dwell_steps=0)
    ok &= r.accepted and r.reason == "invalid_fixed"
    ok &= r.assignment[2] == 1                           # 合法成员保持

    # c) 普通保持条件成立（驻留中）+ 强制事件 -> force 不被驻留短路
    prev = np.array([0, 0, 1, -1])
    prev_valid = np.array([True, True, True, True])
    r = membership_gate(p, prev, hard, E_c, E_p, terms, prev_valid,
                        dwell_steps_since_change=2, capture_hold_active=False,
                        event=GateEvent(force=True, reason="stagnation"),
                        min_dwell_steps=5)
    ok &= r.accepted and r.reason.startswith("force")

    # d) 合法捕获保持 + 无事件 -> 保持
    r = membership_gate(p, prev, hard, E_c, E_p, terms, prev_valid,
                        dwell_steps_since_change=99, capture_hold_active=True,
                        min_dwell_steps=0)
    ok &= (not r.accepted) and r.reason == "capture_hold" and r.assignment.tolist() == [0, 0, 1, -1]
    check("gate combined conditions (review)", ok)


def _energy_of_prev(prev, p):
    hard = np.zeros_like(p.full_plan.cpu().numpy())
    m_max = hard.shape[1] - 1
    for i, a in enumerate(prev):
        hard[i, a if 0 <= a < m_max else m_max] = 1.0
    from crazyflie_examples.six_v_two_hgat.continuous_membership import _assignment_energy
    e, terms = _assignment_energy(
        p.full_plan.detach().cpu().numpy(), hard, prev,
        p.capacity_task.detach().cpu().numpy(),
        float(p.capacity_idle.detach().cpu().numpy()),
        switch_penalty=0.5, count_penalty=1.0)
    return e, terms


def main():
    test_cost_direction()
    test_conservation()
    test_global_ids()
    test_resource_competition()
    test_hard_solver_optimality()
    test_context_consistency()
    test_grad_switch_forward()
    test_finite_difference()
    test_gate_basics()
    test_gate_combined_conditions()
    print(f"\n==== {len(PASS)} PASS / {len(FAIL)} FAIL ====")
    if FAIL:
        print("FAILURES:", FAIL)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
