"""传统环导引基线：在 v4_dispersed 6v2 场景上测试任务可解性上限。

方法（无学习）：
1. 任务分配：按距离贪心，每目标 3 机（静态分配，不切换）；
2. 环导引：每机被分配到以逃逸者为圆心、均匀分布的环槽位（相位差 2π/3），
   分两相——逼近相（飞向半径 R_APPROACH 的环）→ 收缩相（全队就位后环半径
   以 0.3 m/s 内收到 0.85·d_cap，保持角间距）；
3. 斥力避障（障碍/队友/边界）。

捕获判据与环境一致（≥3 机、距离 ≤ d_cap、张角闭合、保持 5 步），
逃逸者用环境自身的目标驱动势场（v_max_e 可配，对齐学习臂的 0.20）。

运行：python -m crazyflie_examples.six_v_two_hgat.traditional_ring_baseline \
        --d-cap 1.5 --episodes 100 --seed 0
"""

from __future__ import annotations

import argparse
import numpy as np

from .stage_env import StageCoalitionEnv
from .v2_env_adapter import apply_membership


def greedy_assign(pos: np.ndarray, evaders: np.ndarray, per_task: int) -> np.ndarray:
    """按距离贪心分配，每任务至多 per_task 机；返回 assignment[Np]。 """
    n_p, n_e = len(pos), len(evaders)
    pairs = sorted(
        (float(np.linalg.norm(pos[i] - evaders[j])), i, j)
        for i in range(n_p) for j in range(n_e)
    )
    assignment = np.full(n_p, -1, dtype=np.int64)
    count = np.zeros(n_e, dtype=np.int64)
    for _, i, j in pairs:
        if assignment[i] == -1 and count[j] < per_task:
            assignment[i] = j
            count[j] += 1
    # 兜底：未分配者塞给人数最少的任务
    for i in range(n_p):
        if assignment[i] == -1:
            assignment[i] = int(np.argmin(count))
            count[assignment[i]] += 1
    return assignment


class RingGuidanceController:
    def __init__(self, d_cap: float, v_max: float = 0.35, n_p: int = 6):
        self.d_cap = float(d_cap)
        self.v_max = float(v_max)
        self.n_p = n_p
        self.r_approach = max(1.7, 1.15 * d_cap)
        self.r_inner = 0.85 * d_cap
        self.shrink_per_step = 0.008      # 0.4 m/s at 50 Hz
        self.k_guide = 1.2                # 槽位吸引增益
        self.assign = np.full(n_p, -1, dtype=np.int64)
        self.r_current: dict[int, float] = {}
        self.base_phase: dict[int, float] = {}

    def reset(self, env: StageCoalitionEnv):
        pos = env.pursuer_positions()[:, :2]
        ev = env.evader_positions()[:, :2]
        per = max(1, self.n_p // max(1, env.num_evaders))
        self.assign = greedy_assign(pos, ev, per)
        self.r_current = {}
        self.base_phase = {}
        for j in range(env.num_evaders):
            self.r_current[j] = self.r_approach
            # 基准相位对齐逃逸者速度方向，减少绕飞
            v = env.evader_velocities()[j][:2]
            self.base_phase[j] = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-3 else 0.0

    def act(self, env: StageCoalitionEnv) -> np.ndarray:
        n_e = env.num_evaders
        pos = env.pursuer_positions()[:, :2]
        ev = env.evader_positions()[:, :2]
        cmd = np.zeros((self.n_p, 2), dtype=np.float64)

        for j in range(n_e):
            members = [i for i in range(self.n_p) if self.assign[i] == j]
            if not members:
                continue
            # 比例导引：环心前移（拦截逃逸者的未来位置而非当前位置）
            ev_vel = env.evader_velocities()[j][:2]
            d0 = np.array([np.linalg.norm(pos[i] - ev[j]) for i in members])
            lead = ev_vel * np.minimum(d0.max() / self.v_max, 2.5) * 0.8
            center = ev[j] + lead
            d = np.array([np.linalg.norm(pos[i] - center) for i in members])
            d_max = float(d.max())
            # 近处成员直接瞄准内环，省去逼近相
            if d_max < 2.2:
                self.r_current[j] = self.r_inner
            elif d_max <= self.r_current[j] + 0.15:
                self.r_current[j] = max(self.r_inner, self.r_current[j] - self.shrink_per_step)
            r = self.r_current[j]
            # 稳定槽位：按当前角度排序，避免交换
            ang = np.array([np.arctan2(pos[i][1] - center[1], pos[i][0] - center[0]) for i in members])
            order = [members[k] for k in np.argsort(ang)]
            m = len(order)
            for k, i in enumerate(order):
                theta = self.base_phase[j] + k * (2.0 * np.pi / m)
                slot = center + r * np.array([np.cos(theta), np.sin(theta)])
                v = self.k_guide * (slot - pos[i])
                # 障碍斥力
                for o in env.obstacles:
                    diff = pos[i] - o.position[:2]
                    dist = float(np.linalg.norm(diff))
                    safe = o.radius + 0.45
                    if 1e-6 < dist < safe:
                        v += 0.8 * (1.0 / dist - 1.0 / safe) * diff / dist
                # 队友斥力
                for k2 in range(self.n_p):
                    if k2 == i:
                        continue
                    diff = pos[i] - pos[k2][:2] if pos[k2].ndim > 1 else pos[i] - pos[k2]
                    dist = float(np.linalg.norm(diff))
                    if 1e-6 < dist < 0.5:
                        v += 0.5 * (1.0 / dist - 2.0) * diff / dist
                # 边界
                L = env.length
                margin = 0.6
                if pos[i][0] < margin: v[0] += (margin - pos[i][0]) * 3
                if pos[i][0] > L - margin: v[0] -= (pos[i][0] - (L - margin)) * 3
                if pos[i][1] < margin: v[1] += (margin - pos[i][1]) * 3
                if pos[i][1] > L - margin: v[1] -= (pos[i][1] - (L - margin)) * 3
                # 限速→归一化
                norm = float(np.linalg.norm(v))
                if norm > self.v_max:
                    v = v / norm * self.v_max
                cmd[i] = v / self.v_max  # env 需要 [-1,1]
        return cmd


class ContinuousRingGuidanceController:
    """Slot-free radial closure with continuous angular gap balancing.

    Members keep their current cyclic order.  Radial feedback moves every
    assigned member inside the capture radius while nearest-neighbour angular
    gap feedback spreads the coalition around the live target.  No discrete
    slot identity or slot reassignment is used.
    """

    def __init__(self, d_cap: float, v_max: float = 0.35, n_p: int = 6):
        self.d_cap = float(d_cap)
        self.v_max = float(v_max)
        self.n_p = int(n_p)
        self.r_target = 0.85 * self.d_cap
        self.k_radial = 1.2
        self.k_angular = 0.45
        self.assign = np.full(self.n_p, -1, dtype=np.int64)

    def reset(self, env: StageCoalitionEnv):
        pos = env.pursuer_positions()[:, :2]
        evaders = env.evader_positions()[:, :2]
        per_task = max(1, self.n_p // max(1, env.num_evaders))
        self.assign = greedy_assign(pos, evaders, per_task)

    def act(self, env: StageCoalitionEnv) -> np.ndarray:
        pos = env.pursuer_positions()[:, :2]
        evaders = env.evader_positions()[:, :2]
        evader_velocities = env.evader_velocities()[:, :2]
        cmd = np.zeros((self.n_p, 2), dtype=np.float64)

        for task_id in range(env.num_evaders):
            members = np.flatnonzero(self.assign == task_id)
            if members.size == 0:
                continue
            center = evaders[task_id]
            offsets = pos[members] - center
            radii = np.linalg.norm(offsets, axis=1).clip(min=1e-6)
            radial = offsets / radii[:, None]
            angles = np.arctan2(offsets[:, 1], offsets[:, 0])
            order = np.argsort(angles)
            ordered_angles = angles[order]
            gaps = np.mod(np.roll(ordered_angles, -1) - ordered_angles,
                          2.0 * np.pi)
            angular_velocity = np.empty(len(members), dtype=np.float64)
            for rank, local_index in enumerate(order):
                previous_gap = gaps[(rank - 1) % len(gaps)]
                next_gap = gaps[rank]
                angular_velocity[local_index] = self.k_angular * (
                    next_gap - previous_gap
                )

            tangent = np.stack([-radial[:, 1], radial[:, 0]], axis=1)
            radial_velocity = self.k_radial * (self.r_target - radii)
            velocities = (
                radial_velocity[:, None] * radial
                + (radii * angular_velocity)[:, None] * tangent
                + evader_velocities[task_id]
            )

            for local_index, member in enumerate(members):
                velocity = velocities[local_index]
                for obstacle in env.obstacles:
                    difference = pos[member] - obstacle.position[:2]
                    distance = float(np.linalg.norm(difference))
                    safe = obstacle.radius + 0.45
                    if 1e-6 < distance < safe:
                        velocity += (0.8 * (1.0 / distance - 1.0 / safe)
                                     * difference / distance)
                for other in range(self.n_p):
                    if other == member:
                        continue
                    difference = pos[member] - pos[other]
                    distance = float(np.linalg.norm(difference))
                    if 1e-6 < distance < 0.5:
                        velocity += (0.5 * (1.0 / distance - 2.0)
                                     * difference / distance)
                margin = 0.6
                if pos[member, 0] < margin:
                    velocity[0] += (margin - pos[member, 0]) * 3.0
                if pos[member, 0] > env.length - margin:
                    velocity[0] -= (pos[member, 0] - (env.length - margin)) * 3.0
                if pos[member, 1] < margin:
                    velocity[1] += (margin - pos[member, 1]) * 3.0
                if pos[member, 1] > env.length - margin:
                    velocity[1] -= (pos[member, 1] - (env.length - margin)) * 3.0
                speed = float(np.linalg.norm(velocity))
                if speed > self.v_max:
                    velocity = velocity / speed * self.v_max
                cmd[member] = velocity / self.v_max
        return cmd


def run(env: StageCoalitionEnv, controller: RingGuidanceController,
        episodes: int, seed: int, max_steps: int) -> dict:
    full = 0
    single = 0
    rewards = []
    times = []
    goals = 0
    for k in range(episodes):
        env.reset(num_pursuers=6, num_evaders=2, scenario_seed=seed + 50000 + k)
        controller.reset(env)
        apply_membership(env, controller.assign)
        done = False
        t = 0
        total_r = 0.0
        cap_time = None
        while not done and t < max_steps:
            cmd = controller.act(env)
            padded = np.zeros((env.max_pursuers, 2), dtype=np.float64)
            padded[:6] = cmd
            r, done, _ = env.step(padded, max_steps=max_steps)
            apply_membership(env, controller.assign)
            total_r += r
            t += 1
            if cap_time is None and env.all_captured():
                cap_time = t * env.time_step
        cap_ev = int(np.count_nonzero(env.captured[:2]))
        full += int(cap_ev == 2)
        single += int(cap_ev >= 1)
        rewards.append(total_r)
        times.append(cap_time)
        goals += int(getattr(env, "outcome", "") == "evader_goal_success")
    n = max(1, episodes)
    finite = [x for x in times if x is not None]
    return {
        "full_capture_rate": full / n,
        "single_capture_rate": single / n,
        "evader_goal_rate": goals / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_capture_time": float(np.mean(finite)) if finite else float("inf"),
        "captures": full,
        "episodes": episodes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-cap", type=float, default=1.5)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--evader-speed", type=float, default=0.20,
                   help="对齐学习臂配置；0.35 为同速更难档")
    args = p.parse_args()

    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6,
                            max_evaders=3, d_cap=args.d_cap, hold_steps=5, seed=args.seed,
                            v_max_e=float(args.evader_speed))
    ctrl = RingGuidanceController(d_cap=args.d_cap)
    res = run(env, ctrl, args.episodes, args.seed, args.max_steps)
    print(f"[传统环导引] d_cap={args.d_cap} evader={args.evader_speed} "
          f"episodes={res['episodes']}")
    for k_, v_ in res.items():
        print(f"  {k_}: {v_}")


if __name__ == "__main__":
    main()
