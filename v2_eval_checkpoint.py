"""V2 checkpoint 独立评测：任意 evader_speed / d_cap / 局数下的成功率。

用法：
  python -m crazyflie_examples.six_v_two_hgat.v2_eval_checkpoint \
      --checkpoint checkpoints/sdgcr_v2_dcap20/checkpoint_010900.pth \
      --evader-speed 0.15 --episodes 100
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .capture_diagnostics import target_capture_diagnostics
from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .v2_controller import V2Controller, v2_obs_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario-seed", type=int, default=None,
                   help="固定场景种子；设置后每局重放完全相同的场景")
    p.add_argument("--report-episodes", action="store_true",
                   help="逐局输出场景种子、结果、捕获数和步数")
    p.add_argument("--output", default=None, help="可选：保存完整逐局诊断 JSON")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--evader-speed", type=float, default=None,
                   help="覆盖评测逃逸者速度；None=用训练值")
    p.add_argument("--legacy-evader-attribute", type=float, default=None,
                   help="复现旧训练 bug：构造后仅覆盖 env.v_max_e 属性")
    p.add_argument("--d-cap", type=float, default=None,
                   help="覆盖捕获判据；None=用 checkpoint 值")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sinkhorn-iters", type=int, default=20,
                   help="Sinkhorn 预算（自适应续跑至 300 上限，精度不变）")
    args = p.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    assert ck.get("architecture_version") == "continuous_membership_v2", "非 V2 checkpoint"
    hidden = int(ck.get("hidden_dim", 32))
    d_cap = float(args.d_cap if args.d_cap is not None else ck.get("d_cap", 1.5))
    hold = int(ck.get("hold_steps", 5))
    v_max_e = float(args.evader_speed if args.evader_speed is not None
                    else ck.get("args", {}).get("evader_speed", 0.2))
    demand_source = str(ck.get("args", {}).get("demand_source", "fixed"))
    fixed_demand = float(ck.get("args", {}).get("fixed_demand", 3.0))
    map_size = float(ck.get("map_size", 8.0))
    num_obstacles = int(ck.get("num_obstacles", 10))

    # 关键：v_max_e 必须经构造函数传入——scenario_config 在 __init__ 冻结，
    # 事后改属性不影响逃逸者势场速度（2026-11-15 发现的配置坑）
    env = StageCoalitionEnv(length=map_size, num_obstacle=num_obstacles,
                            max_pursuers=6, max_evaders=3,
                            d_cap=d_cap, hold_steps=hold, seed=args.seed,
                            v_max_e=v_max_e)
    if args.legacy_evader_attribute is not None:
        env.v_max_e = float(args.legacy_evader_attribute)

    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=num_obstacles,
        map_size=map_size, hidden_dim=hidden, future_steps=10, seq_len=10,
        node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True
    demand_head = ContinuousDemandHead(hidden, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, hidden, 10), hidden_dim=64).to(device)
    encoder.st_hgat.load_state_dict(ck["st_hgat"])
    encoder.relation_head.load_state_dict(ck["relation_head"])
    demand_head.load_state_dict(ck["demand_head"])
    actor.load_state_dict(ck["actor"])
    encoder.eval(); demand_head.eval(); actor.eval()

    controller = V2Controller(
        encoder, demand_head, actor, device,
        demand_source=demand_source, fixed_demand=fixed_demand,
        sinkhorn_iters=args.sinkhorn_iters,
        target_context=str(ck.get("target_context", "hard")),
    )

    full = single = goals = 0
    rewards = []
    times = []
    episode_records = []
    for k in range(args.episodes):
        scenario_seed = (args.scenario_seed if args.scenario_seed is not None
                         else args.seed + 50000 + k)
        env.reset(num_pursuers=6, num_evaders=2, scenario_seed=scenario_seed)
        controller.reset()
        done = False
        t = 0
        r_sum = 0.0
        cap_t = None
        max_hold = [0, 0]
        ever_enclosed = [False, False]
        min_max_member_distance = [float("inf"), float("inf")]
        assignment_switches = 0
        previous_assignment = None
        while not done and t < args.max_steps:
            actions, info = controller.controller_step(env)
            assignment = np.asarray(info["assignment"], dtype=np.int64)
            if previous_assignment is not None and not np.array_equal(
                assignment, previous_assignment
            ):
                assignment_switches += 1
            previous_assignment = assignment.copy()
            r, done, _ = env.step(actions, max_steps=args.max_steps)
            r_sum += r
            t += 1
            for row in target_capture_diagnostics(env, assignment):
                j = row["target"]
                max_hold[j] = max(max_hold[j], row["hold_counter"])
                ever_enclosed[j] = ever_enclosed[j] or row["enclosed"]
                distance = row["max_member_distance"]
                if distance is not None:
                    min_max_member_distance[j] = min(
                        min_max_member_distance[j], distance
                    )
            if cap_t is None and env.all_captured():
                cap_t = t * env.time_step
        ce = int(np.count_nonzero(env.captured[:2]))
        full += int(ce == 2)
        single += int(ce >= 1)
        goals += int(getattr(env, "outcome", "") == "evader_goal_success")
        rewards.append(r_sum)
        times.append(cap_t)
        terminal_targets = target_capture_diagnostics(
            env,
            previous_assignment
            if previous_assignment is not None
            else np.full(6, -1, dtype=np.int64),
        )
        record = {
            "episode": k,
            "scenario_seed": scenario_seed,
            "outcome": getattr(env, "outcome", ""),
            "n_captured": ce,
            "steps": t,
            "capture_time": cap_t,
            "reward": r_sum,
            "assignment_switches": assignment_switches,
            "max_hold": max_hold,
            "ever_enclosed": ever_enclosed,
            "min_max_member_distance": [
                None if not np.isfinite(value) else value
                for value in min_max_member_distance
            ],
            "terminal_targets": terminal_targets,
        }
        episode_records.append(record)
        if args.report_episodes:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    n = max(1, args.episodes)
    finite = [x for x in times if x is not None]
    result = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "d_cap": d_cap,
        "evader_speed": v_max_e,
        "legacy_evader_attribute": args.legacy_evader_attribute,
        "full_capture_rate": full / n,
        "single_capture_rate": single / n,
        "evader_goal_rate": goals / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_capture_time": float(np.mean(finite)) if finite else float("inf"),
        "training_episode_of_ckpt": int(ck.get("episode", -1)),
        "fixed_scenario_seed": args.scenario_seed,
        "episode_records": episode_records,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
