"""V3 hybrid 实飞候选场景筛选。

用与训练/评测完全一致的 V3 决策闭环（``HybridPolicy.decision``：逐步 encode、
按 assign_interval/membership 触发 allocate、apply_membership、
hard_K_from_assignment、build_v2_obs_torch、actor×mask，动作 (6,2) pad 零后
env.step，与 ``evaluate_v3_hybrid.run_episode_eval`` 逐步循环逐行对齐）扫描
实飞候选场景种子，流程：

1. 扫描阶段：每个种子 reset 后跑满一局，记录 captured/full/捕获用时、初始
   6 机位置 + 2 目标点 + 全部障碍（供人工摆位）、初始 assignment/容量、
   全程最小机间距（deploy5x5 安全核对 ≥0.15）。
2. 重复阶段：对至少捕获 1 个目标的种子同种子重复 --reps 次，校验确定性
   （XY 动力学应确定；不一致则在 JSON 标注 unstable）。
3. 排序输出：full capture > any capture > 捕获用时升序，写 JSON。

默认使用 deploy5x5 协议采样器（4×4 m 居中采样盒，天然可塞进 5×5 m 动捕
体积）；``--no-deploy-5x5`` 回退到 env 默认 v4 采样器。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .evaluate_v3_hybrid import build_evaluation_env, load_v3_system, parse_scale
from .scenario_protocol_deploy5x5 import PROTOCOL_VERSION, Deploy5x5Sampler
from .train_v3_hybrid import HybridPolicy

# Deploy5x5Config.min_agent_separation：实机安全机间距下限。
DEPLOY_MIN_SEPARATION = 0.15


def _min_pair_distance(env) -> float:
    """全部存活机体（追捕 + 逃逸）两两 XY 间距的最小值。"""
    n, m = env.num_pursuers, env.num_evaders
    points = env.positions[: n + m, :2]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n + m, k=1)
    return float(dist[iu].min())


def _capture_geometry(env) -> dict:
    """初始摆位表：6 机位置、逃逸机目标点、全部障碍 (x, y, radius)。"""
    obstacles = [[round(float(o.position[0]), 3), round(float(o.position[1]), 3),
                  round(float(o.radius), 3)] for o in env.obstacles]
    return {
        "pursuers": [[round(float(v), 3) for v in p]
                     for p in env.pursuer_positions()],
        "evaders": [[round(float(v), 3) for v in p]
                    for p in env.evader_positions()],
        "goals": [[round(float(v), 3) for v in env.evader_goals[j]]
                  for j in range(env.num_evaders)],
        "obstacles": obstacles,
    }


def run_episode(env, modules, config, seed, scale) -> dict:
    """跑一个固定种子整局；逐步循环与 ``run_episode_eval`` 保持一致。"""
    num_agents, num_tasks = int(scale[0]), int(scale[1])
    env.reset(num_pursuers=num_agents, num_evaders=num_tasks,
              scenario_seed=int(seed))
    env._initial_geometry = _capture_geometry(env)
    for module in modules.values():
        module.eval()
    policy = HybridPolicy(modules, config)
    max_steps = int(config["max_steps"])

    min_dist = _min_pair_distance(env)  # 初始摆位也计入全程最小间距
    initial_assignment = None
    initial_capacity = None
    first_capture = None
    steps = 0
    decision = policy.decision(env)
    for step in range(max_steps):
        if initial_assignment is None:
            initial_assignment = [int(a) for a in decision["assignment"]]
            initial_capacity = [float(c) for c in
                                decision["allocation"]["capacity"]
                                .detach().cpu().numpy()]
        actions = decision["actions"]
        reward, done, info = env.step(actions, max_steps=max_steps)
        if not np.isfinite(actions).all() or not np.isfinite(reward):
            raise FloatingPointError("nonfinite policy action or environment reward")
        steps += 1
        if first_capture is None and info["capture_events"]:
            first_capture = env.episode_step_count * env.time_step
        min_dist = min(min_dist, _min_pair_distance(env))
        if done:
            break
        decision = policy.decision(env)
    captured = int(env.captured[:num_tasks].sum())
    full = bool(captured == num_tasks)
    duration = float(env.episode_step_count * env.time_step)
    return {
        "captured": captured, "full": full,
        "first_capture_time": first_capture,
        # full：整局用时（= 全员捕获时刻）；any 非 full：首次捕获时刻。
        "capture_time": duration if full else first_capture,
        "steps": steps, "duration": duration, "outcome": env.outcome,
        "min_pair_distance": round(min_dist, 4),
        "min_pair_distance_ok": bool(min_dist >= DEPLOY_MIN_SEPARATION),
        "initial_assignment": initial_assignment,
        "initial_capacity": initial_capacity,
    }


def _consistent(a: dict, b: dict) -> bool:
    """两次运行是否确定一致（捕获数/是否 full/步数相同 ⇒ 轨迹一致）。"""
    return (a["captured"] == b["captured"] and a["full"] == b["full"]
            and a["steps"] == b["steps"])


def _sort_key(record) -> tuple:
    scan = record["scan"]
    return (0 if scan["full"] else (1 if scan["captured"] > 0 else 2),
            scan["capture_time"] if scan["capture_time"] is not None else float("inf"),
            record["seed"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_hybrid_4v2_flight/best_4v2.pth",
                        help="trained V3 checkpoint (strict loader)")
    parser.add_argument("--scale", default="4v2", choices=("4v2", "6v2"),
                        help="episode scale (num pursuers v num evaders)")
    parser.add_argument("--deploy-5x5", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="用 Deploy5x5Sampler(4x4 盒)；--no-deploy-5x5 用 env 默认 v4 采样器")
    parser.add_argument("--scan-seeds", type=int, default=200,
                        help="扫描种子数（seed = seed-base + k）")
    parser.add_argument("--seed-base", type=int, default=70000, help="种子起点")
    parser.add_argument("--reps", type=int, default=3,
                        help="对 captured>=1 的种子额外重复次数（确定性确认）")
    parser.add_argument("--out", default="checkpoints/v3_flight_screen.json",
                        help="JSON 输出路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env-seed", type=int, default=0, help="环境构造种子")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    scale = parse_scale(args.scale)
    modules, config = load_v3_system(args.checkpoint, device=args.device)
    env = build_evaluation_env(config, seed=args.env_seed)
    protocol = "default(env sampler)"
    if args.deploy_5x5:
        env.scenario_sampler = Deploy5x5Sampler()
        protocol = PROTOCOL_VERSION
    print(f"[协议] {protocol} | scale={scale[0]}v{scale[1]} | "
          f"device={torch.device(args.device)} | max_steps={int(config['max_steps'])}",
          flush=True)

    records = []
    try:
        # ---- 阶段1：种子扫描 ----
        for k in range(args.scan_seeds):
            seed = args.seed_base + k
            scan = run_episode(env, modules, config, seed, scale)
            record = {"seed": int(seed), "scan": scan,
                      "geometry": _capture_geometry(env),
                      "reps": [], "stable": None}
            records.append(record)
            if (k + 1) % 25 == 0:
                hits = sum(1 for r in records if r["scan"]["captured"] > 0)
                print(f"[scan] {k + 1}/{args.scan_seeds} 种子, "
                      f"{hits} 个至少捕获 1 个目标", flush=True)
        hits = [r for r in records if r["scan"]["captured"] > 0]
        print(f"[scan 完成] {args.scan_seeds} 个种子, {len(hits)} 个至少捕获 1 个目标 "
              f"({len(hits) / max(1, args.scan_seeds):.0%})", flush=True)

        # ---- 阶段2：成功种子重复确认 ----
        if args.reps > 0:
            print(f"[reps] 对 {len(hits)} 个成功种子各重复 {args.reps} 次:", flush=True)
        for record in hits:
            seed = record["seed"]
            for _ in range(args.reps):
                rep = run_episode(env, modules, config, seed, scale)
                record["reps"].append(rep)
            record["stable"] = all(_consistent(record["scan"], rep)
                                   for rep in record["reps"])
            if not record["stable"]:
                print(f"  [warn] 种子 {seed}: reps 结果不一致, 标注 unstable", flush=True)
    finally:
        env.close()

    # ---- 排序：full > any > 捕获用时升序 ----
    candidates = sorted(hits, key=_sort_key)
    for rank, record in enumerate(candidates, start=1):
        record["rank"] = rank
    for record in records:
        record.setdefault("rank", None)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_schema": "v3_sparse_demand_value_hybrid_v1",
        "scale": f"{scale[0]}v{scale[1]}",
        "deploy_5x5": bool(args.deploy_5x5),
        "protocol": protocol,
        "seed_base": int(args.seed_base),
        "scan_seeds": int(args.scan_seeds),
        "reps": int(args.reps),
        "max_steps": int(config["max_steps"]),
        "device": str(torch.device(args.device)),
        "safety_min_separation": DEPLOY_MIN_SEPARATION,
        "sort_order": "full_capture > any_capture > capture_time asc",
        "summary": {
            "scanned": len(records),
            "captured_any": len(hits),
            "full_capture": sum(1 for r in hits if r["scan"]["full"]),
            "stable": sum(1 for r in hits if r["stable"]),
            "unstable": sum(1 for r in hits if r["stable"] is False),
            "min_separation_violations": sum(
                1 for r in records if not r["scan"]["min_pair_distance_ok"]),
        },
        "records": records,
        "candidates": candidates,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")

    print(f"\n=== 候选清单（{len(candidates)} 个, 排序: full > any > 捕获用时升序）===",
          flush=True)
    for record in candidates:
        scan = record["scan"]
        stable_text = ("稳定" if record["stable"] else "UNSTABLE")
        time_text = (f"{scan['capture_time']:.2f}s" if scan["capture_time"] is not None
                     else "未捕获")
        print(f"  #{record['rank']} 种子{record['seed']}: "
              f"full={scan['full']} captured={scan['captured']}/2 "
              f"用时={time_text} {stable_text} "
              f"最小间距={scan['min_pair_distance']:.2f}m"
              f"{'✓' if scan['min_pair_distance_ok'] else '✗'}", flush=True)
    print(f"已保存 {out}", flush=True)
    return payload


if __name__ == "__main__":
    main()
