"""continuous_membership_v2 阶段 C 执行链路验收（审查表 7 项）。

运行：python -m crazyflie_examples.six_v_two_hgat.test_v2_integration
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if (_HERE / "continuous_membership.py").exists():
    sys.path.insert(0, str(_HERE.parent))

from crazyflie_examples.six_v_two_hgat.continuous_membership import (
    build_soft_plan,
    hard_plan_to_assignment,
    membership_gate,
    solve_membership,
    ContinuousDemandHead,
)
from crazyflie_examples.six_v_two_hgat.stage_env import StageCoalitionEnv
from crazyflie_examples.six_v_two_hgat.stage_networks import SharedActor, StageFeatureEncoder
from crazyflie_examples.six_v_two_hgat.v2_controller import V2Controller, v2_obs_dim
from crazyflie_examples.six_v_two_hgat.v2_env_adapter import (
    apply_membership, capture_members, members, membership_validity,
)
from crazyflie_examples.six_v_two_hgat.runtime_continuous_v2 import V2Runtime
from crazyflie_examples.six_v_two_hgat.continuous_membership import GateEvent

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def make_env(seed=7):
    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6,
                            max_evaders=3, d_cap=1.5, hold_steps=5)
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=seed)
    return env


def make_modules(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = StageFeatureEncoder(max_pursuers=6, max_evaders=3, num_obstacles=10,
                              map_size=8.0, hidden_dim=32, future_steps=10,
                              seq_len=10, node_dim=9, device=torch.device("cpu"),
                              use_ring_slot_features=False, use_stage_embedding=False,
                              use_role_embedding=False, use_graph_embedding=True)
    dem = ContinuousDemandHead(32, demand_init=3.0)
    actor = SharedActor(v2_obs_dim(6, 32, 10), hidden_dim=64)
    enc.eval()
    return enc, dem, actor


# ---------------------------------------------------------------------------
# C1 学习需求影响执行：固定状态与关系代价，改需求 -> 候选配置改变
# ---------------------------------------------------------------------------

def test_demand_affects_execution():
    costs = torch.tensor([[0.05, 0.30], [0.30, 0.05], [0.17, 0.18]])  # 0 偏好任务0, 1 偏好任务1
    masks_a = torch.ones(3, dtype=torch.bool)
    masks_t = torch.ones(2, dtype=torch.bool)

    plan_hi0 = build_soft_plan(costs, torch.tensor([3.0, 0.8]), masks_a, masks_t)
    plan_hi1 = build_soft_plan(costs, torch.tensor([0.8, 3.0]), masks_a, masks_t)
    h0, E0, _ = solve_membership(plan_hi0, np.array([-1, -1, -1]))
    h1, E1, _ = solve_membership(plan_hi1, np.array([-1, -1, -1]))
    a0, a1 = hard_plan_to_assignment(h0), hard_plan_to_assignment(h1)
    ok = not np.array_equal(np.sort(a0[a0 >= 0]), np.sort(a1[a1 >= 0])) or \
         (a0 >= 0).sum() != (a1 >= 0).sum()
    # 门控单独验证：驻留满足且改善 -> 接受
    prev = np.array([-1, -1, -1])
    r = membership_gate(plan_hi0, prev, h0, E0, E0 + 1.0, {"x": 0},
                        np.array([True, True, True]), dwell_steps_since_change=9,
                        capture_hold_active=False, min_dwell_steps=5)
    ok &= r.accepted and r.reason == "improved"
    check("C1 demand affects execution (candidate differs; gate separate)", ok,
          f"a(d_hi0)={a0.tolist()} a(d_hi1)={a1.tolist()}")


# ---------------------------------------------------------------------------
# C2 无隐藏角色差异：同一已分配成员集用于捕获/塑形/池化
# ---------------------------------------------------------------------------

def test_no_hidden_role_diff():
    env = make_env()
    assignment = np.array([0, 0, 0, 1, 1, -1])
    apply_membership(env, assignment)
    for j in range(env.num_evaders):
        m = members(env, j)
        ok_j = (m == capture_members(env, j)
                and m == env.coalition_of(j)
                and m == env.shaping_members(j)
                and m == env.core_members(j))
        if not ok_j:
            check("C2 same member set everywhere", False, f"j={j}")
            return
    check("C2 same member set everywhere", True,
          f"coalition_of(0)={env.coalition_of(0)}")


# ---------------------------------------------------------------------------
# C3 生命周期：目标完成 / 成员离线后旧配置及时修复（不等普通周期）
# ---------------------------------------------------------------------------

def test_lifecycle_immediate_fix():
    env = make_env(11)
    enc, dem, actor = make_modules(3)
    ctrl = V2Controller(enc, dem, actor, torch.device("cpu"), assign_interval=1000)
    for _ in range(3):
        ctrl.controller_step(env)
        env.step(np.zeros((6, 2), dtype=np.float32), max_steps=1000)
    # 让所有成员归属任务 0，然后使任务 0 失活
    env.task_mask[0] = False
    env.captured[0] = True
    actions, info = ctrl.controller_step(env)
    assigned_bad = any(a == 0 for a in info["assignment"])
    ok = (not assigned_bad) and info["reconfigured"]
    check("C3 lifecycle immediate fix (target deactivated)", ok,
          f"reason={info['gate_reason']} assignment={info['assignment']}")

    # 成员离线同样立即修复
    env2 = make_env(12)
    enc2, dem2, actor2 = make_modules(4)
    ctrl2 = V2Controller(enc2, dem2, actor2, torch.device("cpu"), assign_interval=1000)
    for _ in range(3):
        ctrl2.controller_step(env2)
        env2.step(np.zeros((6, 2), dtype=np.float32), max_steps=1000)
    before = list(ctrl2._assignment)
    env2.pursuer_mask[0] = False
    _, info2 = ctrl2.controller_step(env2)
    ok2 = info2["reconfigured"] and not any(
        (i == 0) for i in range(env2.num_pursuers)
        if not env2.pursuer_mask[i] and ctrl2._assignment[i] >= 0)
    check("C3 offline member immediate fix", ok2,
          f"reason={info2['gate_reason']} before={before}")


# ---------------------------------------------------------------------------
# C4 时间状态正确：每个物理时刻 H 只推进一次
# ---------------------------------------------------------------------------

def test_time_state_once():
    env = make_env(21)
    enc, dem, actor = make_modules(5)
    ctrl = V2Controller(enc, dem, actor, torch.device("cpu"))
    s0 = enc._step
    ctrl.controller_step(env)
    ok = enc._step == s0 + 1
    # 物理步推进后再次 controller_step：H 再推进一次（内部第二次 encode 命中缓存）
    env.step(np.zeros((6, 2), dtype=np.float32), max_steps=1000)
    ctrl.controller_step(env)
    ok &= enc._step == s0 + 2
    check("C4 graph state advances exactly once per physical step", ok,
          f"_step {s0} -> {enc._step}")


# ---------------------------------------------------------------------------
# C5 三入口一致：controller（训练路径）与 V2Runtime（回放路径）逐步一致
# ---------------------------------------------------------------------------

def test_three_entry_consistency():
    enc, dem, actor = make_modules(9)
    # 保存 V2 checkpoint 供 runtime 加载
    payload = {
        "architecture_version": "continuous_membership_v2",
        "observation_schema_version": "v2_no_stage_role",
        "task_plan_schema_version": "continuous_ot_v1",
        "membership_schema_version": "convex_count_assignment_v1",
        "capture_definition_version": "assigned_members_geometric_v1",
        "context_pool": "normalized_co_membership_mean",
        "target_context": "hard",
        "demand_parameterization": "scalar_softplus",
        "allocation_source": "learned_full_plan",
        "hidden_dim": 32, "future_steps": 10, "seq_len": 10,
        "d_cap": 1.5, "hold_steps": 5, "map_size": 8.0, "num_obstacles": 10,
        "max_pursuers": 6, "max_evaders": 3, "num_pursuers": 6, "num_evaders": 2,
        "args": {"demand_source": "learned_continuous", "demand_init": 3.0},
        "st_hgat": enc.st_hgat.state_dict(),
        "relation_head": enc.relation_head.state_dict(),
        "demand_head": dem.state_dict(),
        "actor": actor.state_dict(),
    }
    ckpt = "/tmp/v2_consistency_ckpt.pth"
    torch.save(payload, ckpt)

    env_a = make_env(33)
    ctrl_a = V2Controller(enc, dem, actor, torch.device("cpu"))
    rt = V2Runtime(ckpt, device="cpu", scenario_seed=33)

    ok = True
    for t in range(8):
        # 决策前捕获动捕帧（runtime 语义：写原始动捕 → controller_step）
        frame_pos = env_a.positions[: env_a.num_pursuers + env_a.num_evaders].copy()
        frame_vel = env_a.velocities[: env_a.num_pursuers + env_a.num_evaders].copy()
        act_a, info_a = ctrl_a.controller_step(env_a)      # 训练入口：内部推进生命周期+归属
        world_pos = frame_pos.copy()
        world_pos[:, :2] = frame_pos[:, :2] - env_a.length / 2.0   # world = local - L/2
        cmd_b, info_b = rt.step(world_pos, frame_vel, (0.0, 0.0))
        same_dem = np.allclose(info_a["demand_raw"], info_b["demand_raw"], atol=1e-5)
        same_assign = info_a["assignment"] == info_b["assignment"]
        # 输入侧 float32 动捕精度差异（rt.step 按 float32 写入）可接受，
        # 三入口语义一致性以 demand/assignment 精确相等 + 动作小容差判定。
        same_act = np.allclose(np.clip(act_a, -1.0, 1.0),
                               cmd_b[:6] / rt.config_max_speed(), atol=0.05)
        if not (same_dem and same_assign and same_act):
            ok = False
            print(f"  t={t} dem {info_a['demand_raw']} vs {info_b['demand_raw']} "
                  f"assign {info_a['assignment']} vs {info_b['assignment']} "
                  f"actmax {np.abs(np.clip(act_a,-1,1)-cmd_b[:6]/rt.config_max_speed()).max()}")
            break
        env_a.step(act_a, max_steps=1000)   # 参考物理推进（runtime 侧由下一帧动捕写入）
    check("C5 controller/runtime identical trajectory", ok)


def test_no_tier_leakage():
    env = make_env(41)
    enc, dem, actor = make_modules(6)
    ctrl = V2Controller(enc, dem, actor, torch.device("cpu"))
    ctrl.controller_step(env)
    obs_a = actor.last_obs if hasattr(actor, "last_obs") else None
    dem_a = np.array(ctrl.last_info["demand_raw"], dtype=np.float64)

    env2 = make_env(41)
    enc2, dem2, actor2 = make_modules(6)
    ctrl2 = V2Controller(enc2, dem2, actor2, torch.device("cpu"))
    # 扰动 legacy 分类字段（stages/roles），不动物理状态与 assignment
    env2.stages[: env2.num_evaders] = (env2.stages[: env2.num_evaders] + 1) % 5
    # 角色扰动保持与 assignment 一致（合法 legacy 配置），仅验证"V2 不消费这些字段"
    for i in range(env2.num_pursuers):
        env2.roles[i] = 0 if env2.assignment[i] >= 0 else 2
    ctrl2.controller_step(env2)
    dem_b = np.array(ctrl2.last_info["demand_raw"], dtype=np.float64)

    snap = env.snapshot()
    snap2 = env2.snapshot()
    check("C6 no tier leakage (demand unchanged)",
          np.allclose(dem_a, dem_b, atol=1e-6) and
          np.array_equal(snap["positions"], snap2["positions"]),
          f"dem_a={dem_a.round(4)} dem_b={dem_b.round(4)}")


# ---------------------------------------------------------------------------
# C7 ID 与空任务：编号不连续 / 全部目标完成时返回合法配置
# ---------------------------------------------------------------------------

def test_id_and_empty_task():
    env = make_env(55)
    env.task_mask[0] = False
    env.captured[0] = True
    enc, dem, actor = make_modules(7)
    ctrl = V2Controller(enc, dem, actor, torch.device("cpu"))
    _, info = ctrl.controller_step(env)
    ok = all(a != 0 for a in info["assignment"])          # 非连续编号不错列
    ok &= info["row_residual_max"] < 1e-3

    env.task_mask[1] = False
    env.captured[1] = True
    _, info2 = ctrl.controller_step(env)
    ok2 = all(a == -1 for a in info2["assignment"])       # 无活跃任务全部未分配
    ok2 &= info2["capacity_task"] == [0.0, 0.0]
    check("C7 ID / empty-task legal configs", ok and ok2,
          f"assign1={info['assignment']} assign2={info2['assignment']}")


def main():
    test_demand_affects_execution()
    test_no_hidden_role_diff()
    test_lifecycle_immediate_fix()
    test_time_state_once()
    test_three_entry_consistency()
    test_no_tier_leakage()
    test_id_and_empty_task()
    print(f"\n==== C-phase: {len(PASS)} PASS / {len(FAIL)} FAIL ====")
    if FAIL:
        print("FAILURES:", FAIL)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
