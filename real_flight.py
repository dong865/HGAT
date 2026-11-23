"""Deploy a retrained 6V2 checkpoint through the Crazyflie setpoint streaming API.

方法为论文主线 SDGCR（``stage_feasible_dgcr``）。本文件是精简后的**实机部署专用**入口，
基线方法（IBR / MAPPO / MADDPG / G2ANet-AC / CBBA-IBR 及旧路线 ST-HGAT+ACPC-MADDPG）
与其训练/评测代码已从本包移除。

本文件是**闭环在线决策**：每个 50 Hz 控制周期读一次动捕 → ``runtime.act()`` →
下发 ``cmdVelocityWorld``（与 3V1 控制器一致的速度控制）。捕获判定与评测器完全同源：生命周期由
``StageCoalitionRuntime`` 内部的 ``env.advance_lifecycle()`` 用同一套判据推进
（≥3 个核心成员进入 ``d_cap=2.0``、逃逸者落在其 XY 凸包内、连续 ``hold_steps=5`` 步），
因此 ``env.captured`` / ``env.task_mask`` 就是机载捕获状态，评测、日志、论文口径一致，
不是另写一套近似判据。

高度（2026-10-16 起）：
  * 物理飞行高度 = ``--height``，默认 **1.0 m = 训练高度**（``StageCoalitionEnv.flight_height``）；
  * 每周期下发 ``cmdVelocityWorld([vx, vy, vz])``：xy 为策略速度指令，vz 由高度 PD 保持 1.0 m；
    脚本不再用 PD 算 vz，飞控持有绝对高度基准，因此不会自己往上爬；
  * 另加高度看门狗 ``--height-window``（默认 ±0.25 m），任一机越窗立即中止并降落，
    防止撞天花板。
  * 捕获后行为与训练/评测完全一致：不插入任何停顿、等待或额外下降流程；捕获
    只由 ``env.advance_lifecycle()`` 标记，被捕获逃逸机按环境 EXIT 速度（原地停住）
    继续，整局结束时统一 land。

用法（建议顺序）：
  1) ``--dry-run``  只算不下发、不上锁不起飞，验证动捕/模型/协议与日志通路；
  2) 单机悬停 → 2 机（``--evader-mode stationary``）→ 全量 ``4v2``。
"""

import argparse
import hashlib
import json
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(4)  # CPU perf: avoid thread oversubscription on tiny nets
from crazyflie_py import Crazyswarm

from .config import SixVTwoConfig
from .mocap_bridge import Nokov
from .runtime_stage import StageCoalitionRuntime
from .runtime_v3_hybrid import V3HybridRuntime
from .runtime_v3_locked import V3LockedRuntime
from .safety import filter_velocity

try:  # 口径自审计：结果文件必须能事后分辨场景协议版本
    from .scenario_protocol import PROTOCOL_VERSION
except Exception:  # pragma: no cover - legacy checkouts
    PROTOCOL_VERSION = "unknown"

# 现场动捕体积半边长（m）。默认 5x5 m 场地，可在命令行按实际场地调整。
FIELD_HALF_EXTENT_M = 2.5

# ---------------------------------------------------------------------------
# 实飞高度（2026-10-16 变更）
# ---------------------------------------------------------------------------
# 物理飞行高度 = **训练高度 1.0 m**（= StageCoalitionEnv.flight_height），不做任何换算。
# 速度控制（与 3V1 real_flight 一致）：每周期发 cmdVelocityWorld([vx, vy, vz])，
# 飞控持有绝对高度基准并把它按在 1.0 m，脚本不再自己算 vz —— 因此不会自己往上爬。
# 另加软件高度看门狗 HEIGHT_WINDOW_M，越窗立刻中止并降落，防撞天花板。
FLIGHT_HEIGHT_M = 1.0
HEIGHT_WINDOW_M = 0.25


def _yaws_from_state(state_rows):
    """从 Nokov 状态行的四元数 (w,x,y,z) 取当前偏航角（rad）。

    ``cmdVelocityWorld`` 用 yawRate=0 保持当前航向，不需要绝对 yaw。
    传实测偏航角即可原地保持航向（yaw 对本任务的平面追逃无影响，保持即可）。
    """
    q = np.asarray(state_rows, dtype=np.float64)[:, 3:7]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _send_velocity(cf, vx, vy, vz):
    """速度控制：与 3V1 real_flight 相同的 cmdVelocityWorld 接口。"""
    cf.cmdVelocityWorld([float(vx), float(vy), float(vz)], yawRate=0.0)


def _height_vz(z, vz, height, kp, kd, vmax):
    """高度 PD：把高度保持在 height 附近，输出垂直速度（m/s）。"""
    return float(np.clip(kp * (height - z) - kd * vz, -vmax, vmax))


def _select_crazyflies(crazyflies, names):
    by_name = {cf.prefix.lstrip("/"): cf for cf in crazyflies}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"Crazyflie names not present in swarm configuration: {missing}")
    return [by_name[name] for name in names]


def _load_obstacles(path):
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (float(item["x"]), float(item["y"]), float(item["radius"]))
        for item in payload.get("obstacles", [])
    ]


def _apply_capture_override(runtime, d_cap=None, hold_steps=None) -> None:
    """实机试飞用的捕获判据覆盖（默认不动论文口径）。

    d_cap 同时作用于三处，必须一起改，否则判据/分配器/策略观测会不一致：
      * env.d_cap      —— 捕获判定 _capture_hold_ok + 奖励/可行性
      * allocator.d_cap—— 联盟可行性 f_cap 与 κ(m)
      * env.snapshot() 会带出 d_cap，编码器的 ring-slot 特征也按它构造
    """
    if d_cap is not None:
        d = float(d_cap)
        runtime.d_cap = d
        if getattr(runtime, "env", None) is not None:
            runtime.env.d_cap = d
        for name in ("allocator", "demand_probe", "feasibility_probe"):
            obj = getattr(runtime, name, None)
            if obj is not None and hasattr(obj, "d_cap"):
                obj.d_cap = d
    if hold_steps is not None:
        h = int(hold_steps)
        runtime.hold_steps = h
        if getattr(runtime, "env", None) is not None:
            runtime.env.hold_steps = h
    print(
        f"[capture] ⚠ 已覆盖捕获判据: d_cap={float(getattr(runtime, 'd_cap', 2.0)):.2f} m, "
        f"hold_steps={int(getattr(runtime, 'hold_steps', 5))}（论文口径 d_cap=2.0 / hold_steps=5）；"
        "收紧属偏离论文口径的实验设定", flush=True,
    )


def _mocap_published_frames(mocap) -> list:
    """动捕当前实际发布的刚体名（用于 preflight 报错时对照 --roles）。"""
    log = getattr(mocap, "log", None)
    if log is None:
        return []
    try:
        return sorted({str(t.child_frame_id).lstrip("/") for t in log.transforms})
    except Exception:  # noqa: BLE001
        return []


def _missing_rigid_bodies(mocap, names) -> list:
    return [names[i] for i, seen in enumerate(mocap.seen) if not seen]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", help="SDGCR checkpoint")
    parser.add_argument("--method", choices=["stage_feasible_dgcr"], default="stage_feasible_dgcr")
    parser.add_argument("--runtime", choices=("stage", "v3hybrid", "v3hybrid_locked"),
                        default="stage",
                        help="决策运行时：stage=论文主线 SDGCR（默认，行为不变）；"
                             "v3hybrid=V3 在线分配；v3hybrid_locked=V3 锁定分配（73016 用）")
    parser.add_argument("--lock-assignment", type=int, nargs=4, default=None,
                        metavar=("P0", "P1", "P2", "P3"),
                        help="v3hybrid_locked 的 4v2 初始任务分配；默认 1 0 1 1")
    parser.add_argument("--scale", choices=("4v2", "5v2", "6v2", "6v3"), default="6v2")
    parser.add_argument("--roles", default="cf1,cf2,cf3,cf4,cf5,cf6,cf7,cf8")
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument("--scenario-seed", type=int, default=None,
                        help="固定场景种子：复现 docs/DEPLOY_SCENARIOS_5x5.md 里某个场景"
                             "（逃逸机目标点 + 障碍布局都由它决定）必须显式指定；"
                             "不指定则沿用训练默认种子（root_seed+episode_index），会飞到另一个场景。"
                             "注意：与 --obstacles-file 同时使用会覆盖该场景自带的障碍布局。")
    parser.add_argument("--no-recenter-scene", action="store_true",
                        help="关闭「部署时场景居中」。默认开启：把采样场景整体平移，使出生点"
                             "质心落在 arena 中心，这样现场可以直接用「相对出生点质心」的坐标，"
                             "否则策略内部相对障碍/目标点会整体偏 arena/2-C（实测 0.3~0.8 m）。")
    parser.add_argument("--deploy-5x5", action="store_true",
                        help="强制使用 deploy_5x5_v1 采样（4×4 m 居中盒）。V3 deploy 模型"
                             "会按检查点里的 scenario_protocol=deploy5x5 自动启用；此开关仅用于"
                             "强制覆盖（例如拿 2.0 m 成熟模型按 5×5 场景摆位时）。")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="控制时长上限（秒），到时自动进入降落流程")
    parser.add_argument("--swap-xy", action="store_true",
                        help="物理布局 x/y 对调时使用：动捕坐标先换回再送策略，输出速度再换回物理系；"
                             "--plan-only 打印的也是对调后的物理坐标")
    parser.add_argument("--max-accel", type=float, default=None,
                        help="V3 runtime 的加速度上限（m/s^2）。默认 None = 不做加速度限幅，与仿真一致；"
                             "实飞想更保守可显式给值（例如 0.5）")
    parser.add_argument("--transition-duration", type=float, default=0.0,
                        help="V3 runtime 分配切换时的速度平滑时长（s）。默认 0 = 与仿真一致")
    parser.add_argument("--height", type=float, default=FLIGHT_HEIGHT_M,
                        help="物理飞行高度（m）。默认 1.0 = 训练高度；高度由飞控位置环闭环。")
    parser.add_argument("--height-window", type=float, default=HEIGHT_WINDOW_M,
                        help="高度看门狗窗口（m）：任一机偏离目标高度超过该值就中止并降落，"
                             "防止高度爬升撞天花板。默认 ±0.25")
    parser.add_argument("--target-center", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="场景中心的动捕绝对坐标。给了它 => 场景固定在动捕坐标系里，"
                             "无人机可以从任意位置起飞、由预位飞到各自目标点；"
                             "不给 => 场景中心 = 起飞点质心（旧行为，现场要按质心摆障碍）。")
    parser.add_argument("--preposition", dest="preposition", action="store_true",
                        default=True, help="起飞后先把各机飞到位再开始实验（默认开启）")
    parser.add_argument("--no-preposition", dest="preposition", action="store_false",
                        help="跳过预位（仅调试；此时现场必须已经摆好）")
    parser.add_argument("--preposition-tolerance", type=float, default=0.15,
                        help="预位 xy 到位判据（m），默认 0.15")
    parser.add_argument("--preposition-z-tolerance", type=float, default=0.12,
                        help="预位 z 到位判据（m），默认 0.12")
    parser.add_argument("--preposition-timeout", type=float, default=45.0,
                        help="预位超时（s），默认 45；sequential 模式下是**每架**的预算")
    parser.add_argument("--preposition-mode", choices=("sequential", "simultaneous"),
                        default="sequential",
                        help="sequential=一架一架飞（默认，任意摆放都安全）；"
                             "simultaneous=同时飞（快，但起点形状与目标差异大时路径会交叉）")
    parser.add_argument("--preposition-safety-distance", type=float, default=0.25,
                        help="预位阶段机间距中止阈值（m），默认 0.25（机体物理接触约 0.09 m）。"
                             "起点摆放越接近目标形状越不会触发；触发即中止并降落。")
    parser.add_argument("--approach-speed", type=float, default=0.30,
                        help="预位飞行速度（m/s），用来算 goTo 的时长，默认 0.30")
    parser.add_argument("--plan-only", action="store_true",
                        help="只打印 [field] 现场布置计划（绝对坐标）然后退出："
                             "不连飞机、不上锁、不起飞、不下发。用于提前规划场地。")
    parser.add_argument("--prepare-wait", dest="prepare_wait", action="store_true",
                        default=True,
                        help="打印 [field] 计划后等操作员把圆柱/标记摆好再继续（默认开启）")
    parser.add_argument("--no-prepare-wait", dest="prepare_wait", action="store_false",
                        help="不等操作员（现场已摆好时用）")
    parser.add_argument("--prepare-timeout", type=float, default=0.0,
                        help="准备阶段最长等待秒数；0（默认）= 一直等 Enter。"
                             "非交互终端下建议设一个值（例如 60）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evader-mode", choices=["escape", "stationary"], default="escape")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读动捕并计算指令，不上锁/不起飞/不下发（台架验证用）")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="平面速度硬上限（m/s），默认取 --scale 对应的协议速度")
    parser.add_argument("--mocap-timeout", type=float, default=15.0,
                        help="等待动捕刚体的超时秒数（默认 15）；超时会列出动捕实际发布的刚体名")
    parser.add_argument("--d-cap", type=float, default=None,
                        help="覆盖捕获判定半径（m）。默认 = 检查点里的论文口径 2.0；"
                             "收紧后捕获率会下降（实测 1.5→约一半、1.2→约 1/8、1.0→约 1/8 且只对特定场景有效），"
                             "属于偏离论文口径的实验设定，需自行评估")
    parser.add_argument("--hold-steps", type=int, default=None,
                        help="覆盖捕获连续保持步数（默认 = 检查点值 5）")
    parser.add_argument("--min-separation", type=float, default=0.40,
                        help="两机间距低于该值时消去相互接近分量（m）；<=0 关闭")
    parser.add_argument("--force-min-separation", type=float, default=0.40,
                        help="硬保护间距：低于该值时两机该周期指令归零（m）；<=0 关闭")
    parser.add_argument("--log-dir", default="logs/real_flight")
    parser.add_argument("--tag", default=None, help="写入日志文件名的标签")
    return parser.parse_args()


def _capture_snapshot(env, n_p, n_e):
    """与评测器同源的捕获量测（直接用 env 的判据，不另写近似）。"""
    pursuers = env.positions[:n_p, :2]
    evaders = env.positions[n_p:n_p + n_e, :2]
    distance = np.linalg.norm(pursuers[:, None, :] - evaders[None, :, :], axis=-1)
    return {
        "min_dist": [float(distance[:, j].min()) for j in range(n_e)],
        "n_within_dcap": [int((distance[:, j] <= float(env.d_cap)).sum()) for j in range(n_e)],
        "hold_ok": [bool(env._capture_hold_ok(j)) for j in range(n_e)],
        "captured": [bool(env.captured[j]) for j in range(n_e)],
        "task_mask": [bool(env.task_mask[j]) for j in range(n_e)],
        "stages": [int(env.stages[j]) for j in range(n_e)],
    }


def _apply_separation_guard(command_xy, positions, hard_radius, soft_radius):
    """消去接近分量；间距低于硬阈值时该周期归零（最后一道机载保护）。"""
    count = len(positions)
    tripped = []
    if hard_radius > 0.0:
        for i in range(count):
            for k in range(i + 1, count):
                delta = positions[i, :2] - positions[k, :2]
                distance = float(np.linalg.norm(delta))
                if distance < hard_radius:
                    command_xy[i] = 0.0
                    command_xy[k] = 0.0
                    tripped.append((i, k, distance))
    if soft_radius > 0.0:
        for i in range(count):
            for k in range(i + 1, count):
                delta = positions[i, :2] - positions[k, :2]
                distance = float(np.linalg.norm(delta))
                if distance < soft_radius and distance > 1e-6:
                    direction = delta / distance
                    for index, axis in ((i, direction), (k, -direction)):
                        closing = float(np.dot(command_xy[index], axis))
                        if closing < 0.0:
                            command_xy[index] = command_xy[index] - closing * axis
    return tripped


def _print_scene(runtime, config, n_evaders, scenario_seed):
    """把运行时内部的场景打印成「相对出生点质心」的现场摆放坐标。

    ``recenter_scene`` 开启时（实飞默认），这些坐标就是
    ``docs/DEPLOY_SCENARIOS_5x5.md`` 里「相对 6 架出生点质心」的表格值，可直接照摆。
    """
    env = getattr(runtime, "env", None)
    if env is None:
        return
    # 注意：arena_size/2 只用于「绝对 arena 坐标 -> 相对质心」的换算（虚拟场地是 8x8），
    #      判定"是否落在动捕体积内"必须用真实场地半边长 2.5 m（5x5），两者不是一个量。
    half = config.arena_size / 2.0
    volume_half = FIELD_HALF_EXTENT_M
    print("[scene] scenario_seed=%s | recenter_scene=%s"
          % (scenario_seed, getattr(runtime, "recenter_scene", False)), flush=True)
    goals = np.asarray(env.evader_goals, dtype=np.float64)[:n_evaders, :2] - half
    for j in range(n_evaders):
        print("  逃逸机目标标记 T%d: (%+.2f, %+.2f)" % (j + 1, goals[j, 0], goals[j, 1]))
    near, far = [], []
    for obstacle in env.obstacles:
        x = float(obstacle.position[0]) - half
        y = float(obstacle.position[1]) - half
        radius = float(obstacle.radius)
        inside = abs(x) <= volume_half and abs(y) <= volume_half
        (near if inside else far).append((x, y, radius))
    near.sort(key=lambda value: value[0] ** 2 + value[1] ** 2)
    print("  %gx%g 动捕体积内障碍 %d 个（按下面坐标摆真实圆柱）:"
          % (2 * volume_half, 2 * volume_half, len(near)))
    for index, (x, y, radius) in enumerate(near):
        print("    O%d: (%+.2f, %+.2f)  r=%.2f m" % (index + 1, x, y, radius))
    print("  体积外障碍 %d 个（协议自带远处槽位，无需摆放）" % len(far))


def _scene_offsets(runtime, config, n_evaders):
    """从已 reset 的运行时里取出本场景的「相对起飞点质心」坐标表。

    ``recenter_scene`` 打开时场景出生点质心正好落在 arena 中心，所以
    ``绝对arena坐标 - arena_size/2`` 就是文档里「相对出生点质心」的偏移（均值 0）。
    返回 (agents[n,2], markers[n_e,2], obstacles[m,3])。
    """
    env = runtime.env
    half = config.arena_size / 2.0
    n_agents = runtime.num_pursuers + runtime.num_evaders
    agents = np.asarray(env.positions[:n_agents, :2], dtype=np.float64) - half
    markers = np.asarray(env.evader_goals[:n_evaders, :2], dtype=np.float64) - half
    obstacles = np.asarray([[o.position[0], o.position[1], o.radius]
                            for o in env.obstacles], dtype=np.float64)
    if obstacles.size:
        obstacles[:, :2] -= half
    return agents, markers, obstacles.reshape(-1, 3)


def _print_field_plan(names, targets, marker_offsets, obstacle_offsets, center, height):
    """打印**动捕绝对坐标**的现场布置计划，供起飞前摆圆柱、贴地面标记。

    只列**落在动捕体积内**的圆柱；体积外的槽位是协议自带的远处障碍，不用摆。
    """
    print(f"[field] 场景中心（动捕绝对坐标）= ({center[0]:+.3f}, {center[1]:+.3f})")
    print("[field] 无人机目标位置（绝对坐标；预位会把每架飞到这里）:")
    for name, tgt in zip(names, targets):
        print(f"    {name}: ({tgt[0]:+.3f}, {tgt[1]:+.3f})   z={height:.2f}")
    for index, (mx, my) in enumerate(marker_offsets):
        print(f"[field] 地面标记 T{index + 1}: ({center[0] + mx:+.3f}, {center[1] + my:+.3f})")
    inside, outside = [], []
    for ox, oy, radius in obstacle_offsets:
        (inside if abs(ox) <= FIELD_HALF_EXTENT_M and abs(oy) <= FIELD_HALF_EXTENT_M
         else outside).append((ox, oy, radius))
    print(f"[field] 需摆放的圆柱障碍 {len(inside)} 个（绝对坐标，"
          f"动捕体积 ±{FIELD_HALF_EXTENT_M:g} m 内）:")
    for index, (ox, oy, radius) in enumerate(inside):
        print(f"    O{index + 1}: ({center[0] + ox:+.3f}, {center[1] + oy:+.3f})   "
              f"r={radius:.2f} m")
    if not inside:
        print("    （本场景体积内无圆柱）")
    if outside:
        print(f"[field] 另 {len(outside)} 个障碍槽位在体积之外（协议自带远处障碍，**不要摆**）")
    # 包络：只统计**真正要摆出来**的实体（机体目标点 + 地面标记 + 体积内圆柱），
    # 体积外的远处障碍槽位是虚拟的、不用摆，不计入。
    rel = list(np.asarray(targets, dtype=np.float64) - np.asarray(center, dtype=np.float64)[None, :])
    rel += [(float(m[0]), float(m[1])) for m in marker_offsets]
    rel += [(float(o[0]), float(o[1])) for o in inside]
    if rel:
        xs = [p[0] for p in rel]
        ys = [p[1] for p in rel]
        radius = max(float(np.hypot(x, y)) for x, y in rel)
        print(f"[field] 相对中心的包络: x ∈ [{min(xs):+.2f}, {max(xs):+.2f}]，"
              f"y ∈ [{min(ys):+.2f}, {max(ys):+.2f}]；最大半径 {radius:.2f} m"
              f"（动捕体积是方形，请以逐轴范围为准）")
        print(f"[field] 绝对范围: x ∈ [{center[0] + min(xs):+.2f}, "
              f"{center[0] + max(xs):+.2f}]，y ∈ [{center[1] + min(ys):+.2f}, "
              f"{center[1] + max(ys):+.2f}]  ← 请核对是否在你的动捕有效区内")


def _preposition(cfs, names, mocap, targets, height, tolerance, z_tolerance,
                 timeout, approach_speed, safety_distance, mode="sequential"):
    """用高层 goTo 把每架机送到指定绝对位置，等全部到位再返回。

    为什么要分模式（实测依据）：
      * 起点「任意摆放」+ 直线飞向目标 => 六条直线路径会**交叉**，
        实测过程中最小机间距可低到 0.137 m（远低于 0.3 m 安全线）。
      * 起点≈目标形状（平移/轻微扰动）时，过程最小间距 0.915 m，安全。
    所以默认 **sequential：一架一架飞**（其余悬停在原地），从任意摆放都安全；
    只有当你确认起点形状与目标形状相近时，才用 simultaneous 省时间。

    三重保护：动捕断流、机间距过小、超时 —— 任一触发都抛错并走降落流程。
    """
    def check(state):
        if mocap.stale(0.25):
            raise RuntimeError("预位阶段动捕 /tf 数据中断")
        pair = min(float(np.linalg.norm(state[i, :2] - state[j, :2]))
                   for i in range(len(names)) for j in range(i + 1, len(names)))
        if pair < float(safety_distance):
            raise RuntimeError(
                f"预位阶段机间距过小（{pair:.3f} m < {safety_distance:.3f} m），中止并降落")

    def launch(index):
        state = mocap.get_drone_state().numpy()
        distance = float(np.linalg.norm(targets[index] - state[index, :2]))
        duration = max(3.0, distance / max(0.05, approach_speed))
        yaw = float(_yaws_from_state(state)[index])
        cfs[index].goTo(
            [float(targets[index, 0]), float(targets[index, 1]), float(height)],
            yaw, duration)
        print(f"[preposition] {names[index]} -> ({targets[index, 0]:+.3f}, "
              f"{targets[index, 1]:+.3f}, {height:.2f})  "
              f"距离 {distance:.2f} m / {duration:.1f} s", flush=True)

    def wait_one(index, budget):
        deadline = time.monotonic() + float(budget)
        err = float("inf")
        while time.monotonic() < deadline:
            state = mocap.get_drone_state().numpy()
            check(state)
            err = float(np.linalg.norm(state[index, :2] - targets[index]))
            z_err = float(abs(state[index, 2] - height))
            if err <= tolerance and z_err <= z_tolerance:
                print(f"[preposition] {names[index]} 就位（xy 误差 {err:.3f} m）",
                      flush=True)
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"预位超时：{names[index]} 还差 {err:.3f} m > {tolerance} m，中止并降落")

    if mode == "simultaneous":
        for index in range(len(cfs)):
            launch(index)
        deadline = time.monotonic() + float(timeout)
        errors = np.full(len(names), np.inf)
        while time.monotonic() < deadline:
            state = mocap.get_drone_state().numpy()
            check(state)
            errors = np.linalg.norm(state[:, :2] - targets, axis=1)
            z_errors = np.abs(state[:, 2] - height)
            if np.all(errors <= tolerance) and np.all(z_errors <= z_tolerance):
                print(f"[preposition] {len(names)} 机就位（xy 最大误差 "
                      f"{errors.max():.3f} m，z 最大误差 {z_errors.max():.3f} m）",
                      flush=True)
                return state
            time.sleep(0.05)
        worst = int(np.argmax(errors))
        raise RuntimeError(
            f"预位超时：{names[worst]} 还差 {errors[worst]:.3f} m > {tolerance} m，中止并降落")

    # sequential：一架一架飞，其余悬停等待
    for index in range(len(cfs)):
        launch(index)
        wait_one(index, timeout)
    state = mocap.get_drone_state().numpy()
    errors = np.linalg.norm(state[:, :2] - targets, axis=1)
    print(f"[preposition] {len(names)} 机全部就位（xy 最大误差 {errors.max():.3f} m）",
          flush=True)
    return state


def _prepare_wait(args, what="圆柱 / 地面标记"):
    """打印完 [field] 计划后，等操作员把现场摆好再继续。

    * 交互终端：按 Enter 提前继续；设了 --prepare-timeout 就最多等这么久。
    * 非交互（重定向/后台）：按 --prepare-timeout 睡够就继续；没设则跳过并告警。
    * --no-prepare-wait：完全不等待。
    """
    if not args.prepare_wait:
        print("[prepare] 跳过等待（--no-prepare-wait）", flush=True)
        return
    timeout = float(args.prepare_timeout)
    if not sys.stdin.isatty():
        if timeout > 0:
            print(f"[prepare] 非交互终端：{timeout:.0f} s 后自动继续", flush=True)
            time.sleep(timeout)
        else:
            print("[prepare] ⚠ 非交互终端且未设 --prepare-timeout：跳过等待，"
                  "请自行确认现场已摆好", flush=True)
        return
    if timeout > 0:
        print(f"[prepare] 请照上面 [field] 的绝对坐标摆好{what}；"
              f"{timeout:.0f} s 后自动继续，摆好可按 Enter 提前继续", flush=True)
    else:
        print(f"[prepare] 请照上面 [field] 的绝对坐标摆好{what}，"
              f"然后按 Enter 继续（Ctrl-C 取消）", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout if timeout > 0 else None)
    if ready:
        sys.stdin.readline()
    else:
        print("[prepare] 等待超时，自动继续", flush=True)


def _quat_rows_to_rpy(quats):
    """(T, N, 4) 四元数 -> (T, N, 3) 欧拉角（rad）。"""
    q = np.asarray(quats, dtype=np.float64)
    if q.size == 0:
        return np.zeros(q.shape[:2] + (3,), dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def _npz_from_frames(frames):
    """把逐帧记录转成与 3V1 实飞记录一致的数值数组，便于 numpy/绘图分析。"""
    if not frames:
        empty = np.zeros((0, 0, 3))
        return {"timestamps": np.zeros(0), "positions": empty, "velocities": empty,
                "commanded_velocities": empty, "attitudes": empty,
                "quaternions": np.zeros((0, 0, 4))}
    arrays = {
        "timestamps": np.asarray([f["t"] for f in frames], dtype=np.float64),
        "positions": np.asarray([f["positions"] for f in frames], dtype=np.float64),
        "velocities": np.asarray([f["velocities"] for f in frames], dtype=np.float64),
        "commanded_velocities": np.asarray([f["command"] for f in frames], dtype=np.float64),
        "quaternions": np.asarray([f["quaternions"] for f in frames], dtype=np.float64),
    }
    arrays["attitudes"] = _quat_rows_to_rpy(arrays["quaternions"])
    return arrays


def main():
    args = _arguments()
    scale_map = {"4v2": (4, 2), "5v2": (5, 2), "6v2": (6, 2), "6v3": (6, 3)}
    n_p, n_e = scale_map[args.scale]
    names = tuple(value.strip().lstrip("/") for value in args.roles.split(","))
    if len(names) != n_p + n_e or len(set(names)) != n_p + n_e:
        raise ValueError(
            f"--roles must contain {n_p + n_e} distinct Crazyflie names for {args.scale}"
        )
    config = SixVTwoConfig(
        num_pursuers=n_p,
        num_evaders=n_e,
        pursuer_names=names[:n_p],
        evader_names=names[n_p:],
        flight_height=args.height,
    )
    max_speed = float(args.max_speed) if args.max_speed else float(config.max_pursuer_speed)
    local_obstacles = _load_obstacles(args.obstacles_file)

    if not args.checkpoint:
        raise ValueError("--method stage_feasible_dgcr requires a checkpoint")
    if args.runtime == "v3hybrid_locked":
        runtime = V3LockedRuntime(
            args.checkpoint,
            config=config,
            device=args.device,
            obstacles=local_obstacles,
            scenario_seed=args.scenario_seed,
            evader_mode=args.evader_mode,
            lock_assignment=(args.lock_assignment
                             if args.lock_assignment is not None else (1, 0, 1, 1)),
            max_acceleration=(1e6 if args.max_accel is None else float(args.max_accel)),
            transition_duration=float(args.transition_duration),
        )
    elif args.runtime == "v3hybrid":
        runtime = V3HybridRuntime(
            args.checkpoint,
            config=config,
            device=args.device,
            obstacles=local_obstacles,
            scenario_seed=args.scenario_seed,
            evader_mode=args.evader_mode,
            max_acceleration=(1e6 if args.max_accel is None else float(args.max_accel)),
            transition_duration=float(args.transition_duration),
        )
    else:
        runtime = StageCoalitionRuntime(
            args.checkpoint,
            config=config,
            device=args.device,
            obstacles=local_obstacles,
            evader_mode=args.evader_mode,
            scenario_seed=args.scenario_seed,
            recenter_scene=not args.no_recenter_scene,
        )
    if args.d_cap is not None or args.hold_steps is not None:
        _apply_capture_override(runtime, args.d_cap, args.hold_steps)

    if args.deploy_5x5:
        from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
        runtime.env.scenario_sampler = Deploy5x5Sampler()
        print("[protocol] 强制 deploy_5x5_v1 采样（4×4 m 居中盒）", flush=True)

    if args.plan_only:
        # 只打印计划：需要检查点来取场景，但**不连飞机**
        runtime.reset()
        plan_center = (np.asarray(args.target_center, dtype=np.float64)
                       if args.target_center is not None else np.zeros(2, dtype=np.float64))
        if args.target_center is None:
            print("[plan-only] 未给 --target-center：场景中心按 (0,0) 计算；"
                  "现场请把它设成你实际要用的中心")
        agent_off, marker_off, obstacle_off = _scene_offsets(runtime, config, n_e)
        if args.swap_xy:
            agent_off = agent_off[:, [1, 0]]
            marker_off = marker_off[:, [1, 0]]
            if obstacle_off.size:
                obstacle_off = obstacle_off[:, [1, 0, 2]]
        _print_field_plan(names, plan_center[None, :] + agent_off, marker_off,
                          obstacle_off, plan_center, config.flight_height)
        print("[plan-only] 只打印计划：未连接飞机、未上锁、未起飞、未下发任何指令", flush=True)
        print(f"[plan-only] 正式飞时用同样的参数（去掉 --plan-only）即可，"
              f"脚本会在起飞前再打印一次并等你确认。", flush=True)
        return

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    metadata = {
        "scale": args.scale,
        "num_pursuers": n_p,
        "num_evaders": n_e,
        "roles": list(names),
        "method": args.method,
        "runtime": args.runtime,
        "lock_assignment": (list(args.lock_assignment)
                            if args.lock_assignment is not None else None),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "scenario_protocol_version": PROTOCOL_VERSION,
        "scenario_seed": args.scenario_seed,
        "deploy_5x5_forced": bool(args.deploy_5x5),
        "recenter_scene": not args.no_recenter_scene,
        "lock_on_close3": bool(getattr(runtime, "lock_on_close3", None)),
        "d_cap": float(getattr(runtime, "d_cap", getattr(config, "d_cap", 2.0))),
        "d_cap_override": (None if args.d_cap is None else float(args.d_cap)),
        "hold_steps_override": (None if args.hold_steps is None else int(args.hold_steps)),
        "hold_steps": int(getattr(runtime, "hold_steps", 5)),
        "control_hz": float(config.control_hz),
        "max_speed": max_speed,
        "min_separation": float(args.min_separation),
        "force_min_separation": float(args.force_min_separation),
        "evader_mode": args.evader_mode,
        "dry_run": bool(args.dry_run),
        "duration_s": float(args.duration),
        "flight_height": float(config.flight_height),
        "height_window": float(args.height_window),
        "swap_xy": bool(args.swap_xy),
        "max_accel": (None if args.max_accel is None else float(args.max_accel)),
        "transition_duration": float(args.transition_duration),
        "obstacles": [list(value) for value in local_obstacles],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("[preflight] " + json.dumps(
        {key: metadata[key] for key in (
            "scale", "roles", "control_hz", "d_cap", "hold_steps", "lock_on_close3",
            "scenario_protocol_version", "scenario_seed", "recenter_scene",
            "max_speed", "dry_run", "checkpoint_sha256",
        )},
        ensure_ascii=False), flush=True)

    swarm = Crazyswarm()
    cfs = _select_crazyflies(swarm.allcfs.crazyflies, names)
    helper = swarm.timeHelper
    mocap = Nokov(cfs)
    airborne = False
    streaming = False
    # 记录用：即使起飞前就异常，这些也必须是已定义的名字
    center = np.zeros(2, dtype=np.float64)
    targets = np.zeros((len(names), 2), dtype=np.float64)
    marker_off = np.zeros((n_e, 2), dtype=np.float64)
    obstacle_off = np.zeros((0, 3), dtype=np.float64)
    frames = []
    capture_events = []
    deadline_misses = 0
    separation_trips = 0
    try:
        print(f"[preflight] 等待 {len(names)} 个动捕刚体（超时 {args.mocap_timeout:.0f}s）：{names}",
              flush=True)
        deadline = time.monotonic() + float(args.mocap_timeout)
        last_report = 0.0
        while not mocap.ready() and time.monotonic() < deadline:
            mocap.get_drone_state()
            now = time.monotonic()
            if now - last_report >= 3.0:
                last_report = now
                published = _mocap_published_frames(mocap)
                print(f"[preflight] 仍缺刚体: {_missing_rigid_bodies(mocap, names)}；"
                      f"动捕当前发布: {published}", flush=True)
            time.sleep(0.05)
        if not mocap.ready():
            missing = _missing_rigid_bodies(mocap, names)
            published = _mocap_published_frames(mocap)
            raise RuntimeError(
                "动捕里没看到这些刚体：" + str(missing) + "\n"
                "  动捕当前实际发布的刚体：" + str(published) + "\n"
                "  请依次确认：1) 该机已上电、在动捕视野内（反光球/LED 正常、未被遮挡）；"
                "2) 动捕软件里的刚体名与 --roles 完全一致；"
                "3) 若只是名字不同，可在动捕软件里改名，或把 --roles 换成实际刚体名后重跑"
            )

        initial = mocap.get_drone_state().numpy()
        if args.swap_xy:
            origin = initial[:, [1, 0]].mean(axis=0)
        else:
            origin = initial[:, :2].mean(axis=0)
        runtime.reset()
        metadata["origin"] = [float(value) for value in origin]
        print(f"[preflight] origin={origin.tolist()}, roles={names}, control=50 Hz, method={args.method}")
        _print_scene(runtime, config, n_e, args.scenario_seed is not None)

        # ---- 场景坐标表（相对质心）-> 动捕绝对目标位置 ----
        agent_off, marker_off, obstacle_off = _scene_offsets(runtime, config, n_e)
        if args.swap_xy:
            agent_off = agent_off[:, [1, 0]]
            marker_off = marker_off[:, [1, 0]]
            if obstacle_off.size:
                obstacle_off = obstacle_off[:, [1, 0, 2]]
        if args.target_center is not None:
            center = np.asarray(args.target_center, dtype=np.float64)
        else:
            center = origin.copy()
            print("[field] 未给 --target-center：场景中心 = 起飞点质心 "
                  f"({center[0]:+.3f}, {center[1]:+.3f})；现场按下面的绝对坐标摆即可")
        targets = center[None, :] + agent_off
        _print_field_plan(names, targets, marker_off, obstacle_off, center,
                          config.flight_height)
        metadata["target_center"] = [float(value) for value in center]
        metadata["preposition"] = bool(args.preposition)

        if args.dry_run:
            print("[dry-run] 不上锁、不起飞、不下发指令；只校验动捕→模型→日志通路")
        else:
            _prepare_wait(args)
            for cf in cfs:
                cf.arm(True)
                helper.sleep(0.25)
            helper.sleep(1.2)
            # 速度起飞：不依赖高层 commander（上一次速度降落后 commander 会停在
            # low-level，高层 takeoff 会被忽略，表现为"有的起、有的不起"）。
            # 同时监控高度上限，防止冲高。
            airborne = True
            print("[takeoff] 速度起飞（cmdVelocityWorld）", flush=True)
            deadline = time.monotonic() + 10.0
            stuck_since = None
            while time.monotonic() < deadline:
                state_now = mocap.get_drone_state().numpy()
                heights = state_now[:, 2]
                too_high = np.flatnonzero(heights > args.height + 0.35)
                if too_high.size:
                    raise RuntimeError(
                        f"takeoff overshoot: {[names[i] for i in too_high]} "
                        f"z={np.round(heights, 2).tolist()}")
                if all(float(h) >= 0.90 * config.flight_height for h in heights):
                    break
                # 部分已到高度、部分还在地面 -> 2 s 后判定为失败，交给 finally 降落
                if (np.any(heights >= 0.90 * config.flight_height)
                        and np.any(heights < 0.10 * config.flight_height)):
                    if stuck_since is None:
                        stuck_since = time.monotonic()
                    elif time.monotonic() - stuck_since > 2.0:
                        break
                else:
                    stuck_since = None
                for i, cf in enumerate(cfs):
                    z = float(heights[i])
                    vz = float(np.clip(
                        1.5 * (config.flight_height - z)
                        - 0.25 * float(state_now[i, 9]), -0.25, 0.30))
                    cf.cmdVelocityWorld([0.0, 0.0, vz], yawRate=0.0)
                time.sleep(0.02)
            _h = mocap.get_drone_state().numpy()[:, 2]
            low = [i for i in range(len(cfs))
                   if float(_h[i]) < 0.5 * config.flight_height]
            if low:
                raise RuntimeError(
                    f"velocity takeoff failed: {[names[i] for i in low]} "
                    f"z={np.round(_h, 2).tolist()}")
            print("[takeoff] 速度起飞完成", flush=True)
            if args.preposition:
                _preposition(cfs, names, mocap, targets, config.flight_height,
                             args.preposition_tolerance, args.preposition_z_tolerance,
                             args.preposition_timeout, args.approach_speed,
                             args.preposition_safety_distance,
                             args.preposition_mode)
                # 预位后重算 origin：此时实际位置 ≈ 目标位置，origin ≈ 场景中心
                _post = mocap.get_drone_state().numpy()
                origin = (_post[:, [1, 0]].mean(axis=0) if args.swap_xy
                          else _post[:, :2].mean(axis=0))
                metadata["origin"] = [float(value) for value in origin]
                print(f"[preposition] 重新标定 origin = "
                      f"({origin[0]:+.3f}, {origin[1]:+.3f})", flush=True)

        total = round(args.duration * config.control_hz)

        def _log_frame(step, state, positions_phys, velocity_phys, command,
                       info, snapshot):
            frames.append({
                "step": step,
                "t": step / config.control_hz,
                "positions": positions_phys.tolist(),
                "velocities": velocity_phys.tolist(),
                "command": command.tolist(),
                "quaternions": state[:, 3:7].tolist(),
                "roles": info.get("roles"),
                "core_assignment": info.get("core_assignment"),
                "stages": info.get("stages"),
                "capture": snapshot,
            })

        for step in range(total):
            started = time.monotonic()
            state = mocap.get_drone_state().numpy()
            if mocap.stale(config.mocap_timeout):
                raise RuntimeError("motion-capture stream is stale")
            positions_phys = state[:, :3].copy()
            velocity_phys = np.nan_to_num(state[:, 7:10]).copy()
            if args.swap_xy:
                positions = positions_phys[:, [1, 0, 2]]
                measured_velocity = velocity_phys[:, [1, 0, 2]]
            else:
                positions = positions_phys
                measured_velocity = velocity_phys

            # 与训练/评测完全一致：每周期推理一次并直接执行，不插入停顿/下降流程
            command_xy, info = runtime.act(positions, measured_velocity, origin)
            command_xy = filter_velocity(positions, command_xy, origin, config)
            command_xy = np.asarray(command_xy, dtype=np.float64).copy()
            speed = np.linalg.norm(command_xy[:, :2], axis=1)
            over = speed > max_speed
            if np.any(over):
                command_xy[over, :2] *= (max_speed / speed[over])[:, None]
            tripped = _apply_separation_guard(
                command_xy[:, :2], positions, args.force_min_separation, args.min_separation,
            )
            if tripped:
                separation_trips += 1
            if args.swap_xy:
                command_xy = command_xy[:, [1, 0]]
            command = np.zeros((len(names), 3), dtype=np.float32)
            command[:, :2] = command_xy[:, :2]
            if not np.isfinite(command).all():
                raise RuntimeError("policy generated a non-finite command")

            snapshot = _capture_snapshot(runtime.env, n_p, n_e)
            for j in range(n_e):
                if snapshot["captured"][j] and not any(e["evader"] == j for e in capture_events):
                    capture_events.append({
                        "evader": j,
                        "step": step,
                        "time_s": step / config.control_hz,
                        "min_dist": snapshot["min_dist"][j],
                    })
                    print(f"[capture] evader {j} captured at "
                          f"t={step / config.control_hz:.2f}s "
                          f"(min_dist={snapshot['min_dist'][j]:.3f} m)", flush=True)

            height_error = np.abs(positions_phys[:, 2] - args.height)
            if np.any(height_error > args.height_window):
                worst = int(np.argmax(height_error))
                raise RuntimeError(
                    f"height guard tripped: {names[worst]} "
                    f"z={positions_phys[worst, 2]:.3f} m, target={args.height:.3f} m, "
                    f"window=±{args.height_window:.3f} m")

            vz_command = np.clip(
                float(config.height_kp) * (float(args.height) - positions_phys[:, 2])
                - float(config.height_kd) * velocity_phys[:, 2],
                -float(config.max_vertical_speed), float(config.max_vertical_speed),
            )

            _log_frame(step, state, positions_phys, velocity_phys, command, info, snapshot)
            if not args.dry_run:
                for idx, cf in enumerate(cfs):
                    _send_velocity(cf, command[idx, 0], command[idx, 1],
                                   float(vz_command[idx]))
                streaming = True
            if step % 50 == 0:
                print(
                    f"[flight] t={step/config.control_hz:.1f}s "
                    f"assignment={info.get('assignment')} stages={info.get('stages')} "
                    f"min_dist={[round(value, 2) for value in snapshot['min_dist']]}"
                )
            remaining = config.dt - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                deadline_misses += 1
                if step % 50 == 0:
                    print("[warning] 50 Hz control deadline missed")
    except KeyboardInterrupt:
        print("[stop] interrupted by operator")
    finally:
        if streaming:
            for _ in range(10):
                for cf in cfs:
                    _send_velocity(cf, 0.0, 0.0, 0.0)
                time.sleep(config.dt)
            for cf in cfs:
                try:
                    cf.notifySetpointsStop()
                except Exception:
                    pass
        if airborne:
            for cf in cfs:
                try:
                    cf.land(targetHeight=0.05, duration=2.0)
                except Exception:
                    pass
            helper.sleep(3.0)
        if not args.dry_run:
            for cf in cfs:
                try:
                    cf.arm(False)
                except Exception:
                    pass
        mocap.close()

        captured = [bool(runtime.env.captured[j]) for j in range(n_e)]
        if n_e and all(captured):
            outcome = "all_captured"
        elif any(captured):
            outcome = "partial_capture"
        else:
            outcome = "no_capture"
        summary = dict(metadata)
        summary.update({
            "frames": len(frames),
            "samples": len(frames),
            "duration_s": (len(frames) / float(config.control_hz)) if frames else 0.0,
            "outcome": outcome,
            "n_captured": int(sum(bool(value) for value in captured)),
            "capture_events": capture_events,
            "capture_time_s": capture_events[0]["time_s"] if capture_events else None,
            "deadline_miss_frames": deadline_misses,
            "separation_guard_frames": separation_trips,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # ---- 现场布置（绝对坐标，便于事后复盘重建现场）----
            "agent_order": list(names),
            "targets_abs": np.asarray(targets, dtype=float).tolist(),
            "markers_abs": (np.asarray(center, dtype=float)[None, :]
                            + np.asarray(marker_off, dtype=float)).tolist(),
            "obstacles_abs": (np.column_stack([
                np.asarray(center, dtype=float)[None, :] + np.asarray(obstacle_off, dtype=float)[:, :2],
                np.asarray(obstacle_off, dtype=float)[:, 2],
            ]).tolist() if len(obstacle_off) else []),
            "obstacles_in_field": int(sum(
                1 for ox, oy, _ in obstacle_off
                if abs(ox) <= FIELD_HALF_EXTENT_M and abs(oy) <= FIELD_HALF_EXTENT_M)),
            "height_window": float(args.height_window),
            "preposition_mode": args.preposition_mode if args.preposition else None,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"flight_{args.scale}_{tag}.json"
        payload = json.dumps({"summary": summary, "frames": frames},
                             indent=1, ensure_ascii=False)
        log_path.write_text(payload, encoding="utf-8")
        # 数值数组（与 3V1 实飞记录同名字段，便于 numpy / 绘图直接读）
        npz_path = log_path.with_suffix(".npz")
        np.savez_compressed(npz_path, **_npz_from_frames(frames))
        # 归档一份（每次实飞都留，不覆盖）
        archive_dir = log_dir / "runs"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive_stem = archive_dir / f"flight_{args.scale}_{tag}_{stamp}"
        archive_stem.with_suffix(".json").write_text(payload, encoding="utf-8")
        np.savez_compressed(archive_stem.with_suffix(".npz"), **_npz_from_frames(frames))
        print(f"[done] {len(names)} vehicles landed and disarmed; "
              f"outcome={outcome} captured={summary['n_captured']}/{n_e}")
        print(f"[log] 实飞记录: {log_path} / {npz_path}"
              f"（{summary['samples']} 帧）+ runs/ 归档", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
