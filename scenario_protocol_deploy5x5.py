"""实机部署用 5×5 m 场地场景规则（对 v4 dispersed 的**重新表征**）。

为什么要重表征
--------------
论文/仿真的 v4 协议把追捕者独立铺满整个 8×8 m 场地，并要求
``min_pursuer_axis_span = 4.0``（x、y 跨度都 ≥4 m），因此场景的两两最大距离
中位数 6.82 m、p95 8.05 m —— 5×5 m 动捕体积**装不下**（实测只有 10.8% 的场景
能塞进 6.0 m）。5×5 场地必须重定采样尺度。

设计原则（关键取舍）
--------------------
1. **不动 8 m 协议坐标系**：``StageFeatureEncoder(map_size=8.0)`` 用 8 m 做输入
   归一化，训练/评测都在这个尺度上。若把坐标系改成 5 m，策略看到的相对位置尺度
   会整体变化 → 分布偏移。所以仍用 8 m 坐标系，只把**采样盒**限制为
   4×4 m 居中盒子（[2,6]²）。实体包围盒 ≤4 m，放进 5 m 体积后每边留 0.5 m 余量。
2. **不动捕获判据**：``d_cap = 2.0``、``hold_steps = 5``、凸包判据、``max_steps`` 全部不变。
3. **不动模型输入维度**：障碍仍为 10 个（``num_obstacles=10`` 写进了编码器的输入宽度），
   障碍半径仍是 0.12~0.22 m，速度/控制频率不变。
4. **只重定"长度类"常数**：边距、间距、目标距离、跨度要求按 4 m 盒子重算
   （见 ``Deploy5x5Config`` 每项的注释：原值 → 新值 → 原因）。
5. **保留 v4 的"无刻意性"约束**：追捕者全场地独立均匀采样、跨度分散、每象限 ≤2、
   局部 ≤2 人，并拒绝"初始接近合围"（``initial_enclosure_buffer``）。

注意：``d_cap=2.0`` 相对 4 m 采样盒比在 8 m 场地里"更大"，因此这一版场景**更密、
更容易**——难度变化必须实测（见 ``gen_deploy_scenes_5x5.py`` 的仿真验证与直飞基线）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .scenario_protocol import (
    ScenarioGenerationError,
    ScenarioState,
    _angular_gap,
    _path_exists,
)

PROTOCOL_VERSION = "deploy_5x5_v1"

FRAME = 8.0          # 训练/协议坐标系边长（模型归一化尺度，不可改）
BOX = 4.0            # 采样盒边长（居中于坐标系中心）；实体包围盒 ≤4 m ⊂ 5×5 体积
CENTER = FRAME / 2.0


@dataclass(frozen=True)
class Deploy5x5Config:
    """5×5 场地版场景常数。每项都标注了相对 v4(8×8) 的变化与理由。"""

    protocol_version: str = PROTOCOL_VERSION
    length: float = FRAME              # 不变：模型归一化尺度
    flight_height: float = 1.0         # 不变
    # --- 障碍布置（混合式，见下方说明） ---
    # 编码器的障碍槽位数固定为 10（写进输入维度，不能改）。但在 4×4 m 盒里放 10 个
    # 障碍会把盒子占满：每个障碍的排斥盘 π(r+clearance)² ≈ 1.3 m²，×10 ≈ 整个盒面积，
    # 结果是实体↔障碍净距永远卡在协议下限、实机也不安全。
    # 因此：**可飞体积内只放 obstacles_inside 个**（实机真实摆放），其余放在体积外的
    # 环形区域（模型仍看到 10 个槽位，属训练分布内的"远处杂波"，实机无需摆放）。
    num_obstacles: int = 10            # 不变：编码器输入宽度写死 10
    obstacles_inside: int = 4          # 可飞体积内真实障碍数（实机按表摆放）
    volume: float = 5.0                # 实机动捕体积边长（4×4 盒 ⊂ 5×5 体积）
    obstacle_radius_min: float = 0.12  # 不变
    obstacle_radius_max: float = 0.22  # 不变
    obstacle_height: float = 3.0       # 不变
    agent_radius: float = 0.2          # 不变（栅格可达性用）
    collision_distance: float = 0.4    # 不变

    # --- 采样盒（新） ---
    box: float = BOX                   # 4.0 m，居中；实体包围盒 ≤4 m
    box_margin: float = 0.30           # 盒内留白：0.7(8m) → 0.30(4m)，等比 0.7/8*4≈0.35
    goal_box_margin: float = 0.50      # 目标点盒内留白：1.6(8m) → 0.50，避免目标贴体积边缘
    # 这两个是"8 m 场地外边界"用的（势场边界斥力 / 目标边界检查）。
    # 采样盒在场地中央，实体离外边界 ≥2.3 m，所以保持 v4 原值即可，不参与 5×5 约束。
    boundary_margin: float = 0.70
    goal_boundary_margin: float = 1.60

    # --- 间距类（按 4 m 盒重定） ---
    # 实机硬件：Crazyflie 机臂 u = 0.046 m，整机半径 ≈ u+桨径 ≈ 0.07 m。
    # 用户给定原则：「留出的安全距离比两个机臂长度(2u=0.092)稍长即可」，
    # 因此机间最小间距取 0.12 m（略大于 2u；两机桨尖间隙 ≈ 0.12-0.14=-0.02 →
    # 实际按整机半径 0.07×2=0.14 计，0.12 已属"紧贴"，故下面用 0.15 留一点余量）。
    min_agent_separation: float = 0.15
    min_pursuer_separation: float = 0.70    # 1.2 → 0.70
    min_pursuer_evader: float = 1.50        # 不变：物理量，且 < d_cap 会引入初始捕获
    min_evader_separation: float = 1.00     # 1.2 → 1.00
    min_goal_separation: float = 1.00       # 1.2 → 1.00
    # 实体(含目标点)↔障碍净距：用户要求"不要过大"，取 0.12 m ≈ 2.6u
    # （整机半径 ≈0.07 m → 桨尖到柱面 ≈0.05 m）。8 m 场地的 0.65 m 在 5×5 里既不可行也无必要。
    entity_clearance: float = 0.12
    entity_clearance_inner: float = 0.12
    obstacle_gap: float = 0.15              # 0.35 → 0.15（障碍之间净距，仅防止互相重叠）

    # --- 目标距离（按 4 m 盒重定） ---
    min_goal_distance: float = 1.50         # 3.0 → 1.50：4 m 盒内长距离目标不可行
    max_goal_distance: float = 3.20         # 5.5 → 3.20
    goal_radius: float = 0.25               # 不变（到达判定）

    # --- 分散性（保留 v4 意图，按盒缩小） ---
    min_pursuer_axis_span: float = 2.00     # 4.0 → 2.00（可用盒约 3.4 m）
    max_pursuers_per_quadrant: int = 2      # 不变（象限相对盒中心划分）
    local_pursuer_radius: float = 1.50      # 3.0 → 1.50
    max_local_pursuers_per_evader: int = 2  # 不变

    # --- 捕获判据（一律不变） ---
    capture_radius: float = 2.0             # = d_cap，论文协议
    initial_enclosure_buffer: float = 0.75  # 不变：拒绝 d_cap+0.75 内的接近合围

    # --- 栅格可达性（不变） ---
    grid_resolution: float = 0.1
    max_attempts: int = 2000

    # --- 势场逃逸参数（不变） ---
    max_speed: float = 0.35
    goal_weight: float = 3.0
    pursuer_weight: float = 0.3
    obstacle_weight: float = 1.8
    boundary_weight: float = 0.8
    sibling_weight: float = 0.25
    tangent_weight: float = 0.35
    pursuer_influence: float = 2.5
    max_pursuer_repulsion: float = 2.0
    obstacle_influence: float = 0.9

    @property
    def lo(self) -> float:
        return CENTER - self.box / 2.0 + self.box_margin      # 2.30

    @property
    def hi(self) -> float:
        return CENTER + self.box / 2.0 - self.box_margin      # 5.70

    @property
    def goal_lo(self) -> float:
        return CENTER - self.box / 2.0 + self.goal_box_margin  # 2.50

    @property
    def goal_hi(self) -> float:
        return CENTER + self.box / 2.0 - self.goal_box_margin  # 5.50


def validate_scenario(state: ScenarioState, cfg: Deploy5x5Config) -> Dict[str, object]:
    positions = np.vstack((state.pursuer_positions, state.evader_positions))
    errors = []
    if len(state.obstacles) != cfg.num_obstacles:
        errors.append("obstacle_count")
    for i in range(len(positions)):
        if np.any(positions[i, :2] < cfg.lo) or np.any(positions[i, :2] > cfg.hi):
            errors.append("entity_outside_box")
        for k in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i, :2] - positions[k, :2]) < cfg.min_agent_separation:
                errors.append("entity_separation")
    for goal in state.evader_goals:
        if np.any(goal[:2] < cfg.goal_lo) or np.any(goal[:2] > cfg.goal_hi):
            errors.append("goal_outside_box")
    for i, evader in enumerate(state.evader_positions):
        distance = np.linalg.norm(state.evader_goals[i, :2] - evader[:2])
        if not cfg.min_goal_distance <= distance <= cfg.max_goal_distance:
            errors.append("goal_distance")
        if not _path_exists(evader[:2], state.evader_goals[i, :2], state.obstacles, cfg):
            errors.append("unreachable_goal")
        close = state.pursuer_positions[
            np.linalg.norm(state.pursuer_positions[:, :2] - evader[:2], axis=1) <= cfg.capture_radius
        ]
        if len(close) >= 3 and _angular_gap(close[:, :2], evader[:2]) < np.pi:
            errors.append("initial_capture")
        near_ready = state.pursuer_positions[
            np.linalg.norm(state.pursuer_positions[:, :2] - evader[:2], axis=1)
            <= cfg.capture_radius + cfg.initial_enclosure_buffer
        ]
        if len(near_ready) >= 3 and _angular_gap(near_ready[:, :2], evader[:2]) < np.pi:
            errors.append("initial_near_ready_enclosure")
    for point in np.vstack((positions, state.evader_goals)):
        for x, y, radius, *_ in state.obstacles:
            if np.linalg.norm(point[:2] - np.array([x, y])) < radius + cfg.entity_clearance:
                errors.append("obstacle_clearance")
    return {"valid": not errors, "errors": tuple(sorted(set(errors)))}


class Deploy5x5Sampler:
    """4×4 m 居中采样盒版采样器（结构与 v4 一致，仅换盒与常数）。"""

    def __init__(self, config: Optional[Deploy5x5Config] = None):
        self.config = config or Deploy5x5Config()

    def sample(self, n_pursuers: int, n_evaders: int, seed: int) -> ScenarioState:
        cfg = self.config
        lo, hi = cfg.lo, cfg.hi
        glo, ghi = cfg.goal_lo, cfg.goal_hi
        z = cfg.flight_height
        for attempt in range(cfg.max_attempts):
            rng = np.random.default_rng(int(seed) + attempt * 104729)

            evaders = []
            for _ in range(n_evaders):
                placed = False
                for _ in range(500):
                    xy = rng.uniform(lo, hi, 2)
                    if any(np.linalg.norm(xy - other[:2]) < cfg.min_evader_separation
                           for other in evaders):
                        continue
                    evaders.append(np.array([xy[0], xy[1], z]))
                    placed = True
                    break
                if not placed:
                    break
            if len(evaders) != n_evaders:
                continue
            evaders = np.asarray(evaders)

            goals = []
            for evader in evaders:
                placed = False
                for _ in range(2000):
                    goal_xy = rng.uniform(glo, ghi, 2)
                    distance = float(np.linalg.norm(goal_xy - evader[:2]))
                    if not cfg.min_goal_distance <= distance <= cfg.max_goal_distance:
                        continue
                    if any(np.linalg.norm(goal_xy - other[:2]) < cfg.min_goal_separation
                           for other in goals):
                        continue
                    goals.append(np.array([goal_xy[0], goal_xy[1], z]))
                    placed = True
                    break
                if not placed:
                    break
            if len(goals) != n_evaders:
                continue
            goals_arr = np.asarray(goals)

            pursuers = []
            for _ in range(n_pursuers):
                placed = False
                for _ in range(2000):
                    xy = rng.uniform(lo, hi, 2)
                    if any(np.linalg.norm(xy - p[:2]) < cfg.min_pursuer_separation
                           for p in pursuers):
                        continue
                    if np.min(np.linalg.norm(evaders[:, :2] - xy, axis=1)) < cfg.min_pursuer_evader:
                        continue
                    if np.min(np.linalg.norm(goals_arr[:, :2] - xy, axis=1)) < cfg.entity_clearance:
                        continue
                    pursuers.append(np.array([xy[0], xy[1], z]))
                    placed = True
                    break
                if not placed:
                    break
            if len(pursuers) != n_pursuers:
                continue
            pursuers_arr = np.asarray(pursuers)

            if (np.ptp(pursuers_arr[:, 0]) < cfg.min_pursuer_axis_span
                    or np.ptp(pursuers_arr[:, 1]) < cfg.min_pursuer_axis_span):
                continue
            quadrants = (
                (pursuers_arr[:, 0] >= CENTER).astype(int)
                + 2 * (pursuers_arr[:, 1] >= CENTER).astype(int)
            )
            if np.max(np.bincount(quadrants, minlength=4)) > cfg.max_pursuers_per_quadrant:
                continue
            if any(
                np.count_nonzero(
                    np.linalg.norm(pursuers_arr[:, :2] - evader[:2], axis=1)
                    <= cfg.local_pursuer_radius
                ) > cfg.max_local_pursuers_per_evader
                for evader in evaders
            ):
                continue
            near_ready = False
            for evader in evaders:
                distances = np.linalg.norm(pursuers_arr[:, :2] - evader[:2], axis=1)
                close = pursuers_arr[
                    distances <= cfg.capture_radius + cfg.initial_enclosure_buffer
                ]
                if len(close) >= 3 and _angular_gap(close[:, :2], evader[:2]) < np.pi:
                    near_ready = True
                    break
            if near_ready:
                continue

            protected = np.vstack((pursuers_arr[:, :2], evaders[:, :2], goals_arr[:, :2]))
            obstacles = []
            # 混合布置：前 obstacles_inside 个在可飞体积内（采样盒，约束严、实机真摆），
            # 其余在体积外的环形区域（只作为模型的"远处杂波"槽位，实机无需摆放）。
            vol_lo = CENTER - cfg.volume / 2.0 - 0.30
            vol_hi = CENTER + cfg.volume / 2.0 + 0.30
            for index in range(cfg.num_obstacles):
                inside = index < cfg.obstacles_inside
                placed = False
                for _ in range(800):
                    radius = rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
                    if inside:
                        pos = rng.uniform(lo, hi, 2)
                        clearance = cfg.entity_clearance_inner
                    else:
                        pos = rng.uniform(cfg.boundary_margin, cfg.length - cfg.boundary_margin, 2)
                        # 体积外（含 0.30 m 缓冲）；同时仍与实体保持协议净距
                        if vol_lo <= pos[0] <= vol_hi and vol_lo <= pos[1] <= vol_hi:
                            continue
                        clearance = cfg.entity_clearance
                    if np.min(np.linalg.norm(protected - pos, axis=1)) < radius + clearance:
                        continue
                    if any(np.linalg.norm(pos - o[:2]) < radius + o[2] + cfg.obstacle_gap
                           for o in obstacles):
                        continue
                    obstacles.append(np.array([pos[0], pos[1], radius, cfg.obstacle_height]))
                    placed = True
                    break
                if not placed:
                    break
            if len(obstacles) != cfg.num_obstacles:
                continue

            state = ScenarioState(pursuers_arr, evaders, goals_arr, np.asarray(obstacles), int(seed))
            if validate_scenario(state, cfg)["valid"]:
                return state
        raise ScenarioGenerationError(
            f"failed to generate a valid {n_pursuers}v{n_evaders} 5x5 deploy scene for seed {seed}"
        )
