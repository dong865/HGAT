"""V2 环境适配层：assignment-only 成员定义，兼容旧环境字段。

两步迁移（交接文档 §9）：
1) 适配阶段：`apply_membership` 把所有已分配成员映射到旧存储的 CORE、
   未分配映射到 RES，不产生 SUPPORT——仅为复用旧环境的基线重置逻辑；
2) 查询统一：`members(j)` / `capture_members(j)` / `shaping_members(j)`
   在 V2 下指向同一成员定义（实际 assignment 成员集）。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .stage_env import ROLE_CORE, ROLE_RES


def apply_membership(env, assignment) -> None:
    """把 V2 assignment（全局任务 ID 或 -1）应用到旧环境存储。

    委托 env.apply_roles 完成全部副作用（距离基线、几何塑形基线重置），
    不重复实现。不产生 SUPPORT；未分配不是悬停（旧 RES 语义即移动预备）。
    """
    assignment = np.asarray(assignment).astype(np.int64)
    n = env.num_pursuers
    core = np.full(n, -1, dtype=np.int64)
    sup = np.full(n, -1, dtype=np.int64)
    roles = np.full(n, ROLE_RES, dtype=np.int64)
    for i in range(n):
        a = int(assignment[i]) if i < len(assignment) else -1
        if 0 <= a < env.num_evaders and bool(env.task_mask[a]):
            core[i] = a
            roles[i] = ROLE_CORE
    env.apply_roles(core, sup, roles, stages=None)


def members(env, evader_idx: int) -> List[int]:
    """V2 统一定义：可用且 assignment==j 的成员。"""
    return [
        i for i in range(env.num_pursuers)
        if bool(env.pursuer_mask[i]) and int(env.assignment[i]) == evader_idx
    ]


def capture_members(env, evader_idx: int) -> List[int]:
    """V2：捕获集合 = 实际已分配成员（与 members 同集）。"""
    return members(env, evader_idx)


def membership_validity(env, assignment) -> np.ndarray:
    """旧配置各成员归属是否仍合法（门控 prev_valid 输入）。

    非法 = 成员离线，或归属指向已完成/失活任务。
    """
    assignment = np.asarray(assignment).astype(np.int64)
    valid = np.zeros(env.num_pursuers, dtype=bool)
    for i in range(env.num_pursuers):
        if not bool(env.pursuer_mask[i]):
            continue
        a = int(assignment[i]) if i < len(assignment) else -1
        if a == -1:
            valid[i] = True                       # 未分配恒合法
        elif 0 <= a < env.num_evaders and bool(env.task_mask[a]):
            valid[i] = True
    return valid


def capture_hold_active(env) -> bool:
    """任一活跃任务当前成员集正满足物理捕获保持条件。"""
    for j in range(env.num_evaders):
        if bool(env.task_mask[j]) and not bool(env.captured[j]):
            if env._capture_hold_ok(j):
                return True
    return False
