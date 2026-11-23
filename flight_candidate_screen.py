"""实飞候选场景筛选：两阶段重复测试。

阶段1: 种子扫描（1次/种子）找出成功场景；
阶段2: 成功场景 x 10 次重复，测稳定成功率；
输出: 重复成功率 >= --min-rep-rate 的实飞候选清单。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .v2_controller import V2Controller, v2_obs_dim


def load_controller(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    h = int(ck.get("hidden_dim", 32))
    d_cap = float(ck.get("d_cap", 2.0))
    enc = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=10, map_size=8.0,
        hidden_dim=h, future_steps=10, seq_len=10, node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True).to(device)
    enc._skip_legacy_plan = True
    dh = ContinuousDemandHead(h, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, h, 10), hidden_dim=64).to(device)
    enc.st_hgat.load_state_dict(ck["st_hgat"])
    enc.relation_head.load_state_dict(ck["relation_head"])
    dh.load_state_dict(ck["demand_head"])
    actor.load_state_dict(ck["actor"])
    enc.eval(); dh.eval(); actor.eval()
    ctrl = V2Controller(enc, dh, actor, device, demand_source="fixed", fixed_demand=3.0)
    return ctrl, d_cap


def run_once(env, ctrl, seed, max_steps):
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=seed)
    ctrl.reset()
    done = False
    t = 0
    cap_t = None
    while not done and t < max_steps:
        actions, _ = ctrl.controller_step(env)
        _, done, _ = env.step(actions, max_steps=max_steps)
        t += 1
        if cap_t is None and env.all_captured():
            cap_t = t * env.time_step
    ce = int(np.count_nonzero(env.captured[:2]))
    return (2 if ce == 2 else (1 if ce == 1 else 0)), cap_t


def geo_of(env):
    pos = env.pursuer_positions()[:, :2]
    out = []
    for j in range(2):
        d = np.sort(np.linalg.norm(pos - env.evader_positions()[j][:2], axis=1))[:3]
        out.append((float(d.mean()), float(env.evader_goals[j][2] and 0 or 0)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sdgcr_v2_dcap20/checkpoint_010900.pth")
    p.add_argument("--scan-seeds", type=int, default=200, help="阶段1扫描种子数(从50000起)")
    p.add_argument("--reps", type=int, default=10, help="阶段2每场景重复次数")
    p.add_argument("--min-rep-rate", type=float, default=0.7, help="实飞候选重复成功率阈值")
    p.add_argument("--evader-speed", type=float, default=0.35)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=50000)
    p.add_argument("--deploy-5x5", action="store_true",
                   help="用 5×5 部署协议采样器(4×4 盒)替代默认 v4 8×8 采样器")
    p.add_argument("--out", default="checkpoints/flight_candidates.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl, d_cap = load_controller(args.checkpoint, device)
    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6, max_evaders=3,
                            d_cap=d_cap, hold_steps=5, seed=0,
                            v_max_e=float(args.evader_speed))
    if args.deploy_5x5:
        env.scenario_sampler = Deploy5x5Sampler()
        print("[协议] deploy_5x5_v1 (4x4 采样盒, d_cap=2.0 不变)", flush=True)

    # ---- 阶段1: 扫描 ----
    winners = {}
    geo = {}
    for k in range(args.scan_seeds):
        seed = args.seed_base + k
        outcome, cap_t = run_once(env, ctrl, seed, args.max_steps)
        if outcome >= 1:
            winners[seed] = {"single": int(outcome >= 1), "full": int(outcome == 2),
                             "cap_t": cap_t}
            pos = env.pursuer_positions()[:, :2]
            g = []
            for j in range(2):
                d = np.sort(np.linalg.norm(pos - env.evader_positions()[j][:2], axis=1))[:3]
                g.append(dict(mean3=round(float(d.mean()), 2),
                              goal=round(float(np.linalg.norm(
                                  env.evader_goals[j][:2] - env.evader_positions()[j][:2])), 2)))
            geo[seed] = g
        if (k + 1) % 25 == 0:
            print(f"[scan] {k+1}/{args.scan_seeds} 种子, 已发现 {len(winners)} 个成功场景",
                  flush=True)
    print(f"\n[阶段1完成] {args.scan_seeds} 个场景中 {len(winners)} 个至少成功1次 "
          f"({len(winners)/args.scan_seeds:.0%})", flush=True)

    # ---- 阶段2: 重复测试 ----
    print(f"\n[阶段2] 对 {len(winners)} 个成功场景各重复 {args.reps} 次:", flush=True)
    candidates = []
    for seed, first in winners.items():
        full_hits = single_hits = 0
        times = []
        for _ in range(args.reps):
            outcome, cap_t = run_once(env, ctrl, seed, args.max_steps)
            full_hits += int(outcome == 2)
            single_hits += int(outcome >= 1)
            if cap_t is not None:
                times.append(cap_t)
        rep_full = full_hits / args.reps
        rep_single = single_hits / args.reps
        g = geo[seed]
        marker = " ★实飞候选" if rep_full >= args.min_rep_rate else ""
        mean_time = float(np.mean(times)) if times else None
        time_text = f"{mean_time:.1f}s" if mean_time is not None else "未完整捕获"
        print(f"  种子{seed}: 完整={rep_full:.0%} 单目标={rep_single:.0%} "
              f"捕获用时={time_text}"
              f" | 目标0近3机{g[0]['mean3']}m/目标{g[0]['goal']}m"
              f" | 目标1近3机{g[1]['mean3']}m/目标{g[1]['goal']}m{marker}", flush=True)
        if rep_full >= args.min_rep_rate:
            candidates.append(dict(seed=seed, rep_full=rep_full, rep_single=rep_single,
                                   mean_capture_time=mean_time,
                                   geometry=g))

    print(f"\n=== 实飞候选（重复完整捕获率 ≥ {args.min_rep_rate:.0%}）: "
          f"{len(candidates)} 个 ===", flush=True)
    for c in candidates:
        print(f"  种子 {c['seed']}: 稳定完整捕获 {c['rep_full']:.0%}, "
              f"平均 {c['mean_capture_time']:.1f}s", flush=True)
    with open(args.out, "w") as fh:
        json.dump(candidates, fh, indent=2, ensure_ascii=False)
    print(f"已保存 {args.out}", flush=True)


if __name__ == "__main__":
    main()
