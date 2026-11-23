"""快速测试：6v2 训练的模型在 4v2 场景下的适应情况。

只改 num_pursuers=4，模型/控制器/协议与 6v2 筛选完全一致。
指标: 捕获数(0/1/2，结构性预期为0)、每任务最近2机最小距离、
      最近2机同时在 d_cap 内的累计时间、 episode 长度。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .flight_candidate_screen import load_controller
from .stage_env import StageCoalitionEnv


def run_once(env, ctrl, seed, max_steps):
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=seed)
    ctrl.reset()
    done = False
    t = 0
    min_d2 = [np.inf, np.inf]          # 每任务最近2机距离的最小值
    hold2_time = [0.0, 0.0]            # 最近2机同时在 d_cap 内的累计时间
    while not done and t < max_steps:
        actions, _ = ctrl.controller_step(env)
        if actions.shape[0] < env.max_pursuers:
            pad = np.zeros((env.max_pursuers - actions.shape[0], actions.shape[1]),
                           dtype=actions.dtype)
            actions = np.vstack([actions, pad])
        _, done, _ = env.step(actions, max_steps=max_steps)
        t += 1
        P = env.pursuer_positions()[:, :2]
        for j in range(2):
            d = np.sort(np.linalg.norm(P - env.evader_positions()[j][:2], axis=1))[:2]
            min_d2[j] = min(min_d2[j], float(d.mean()))
            if d.max() <= env.d_cap:
                hold2_time[j] += env.time_step
    n_cap = int(np.count_nonzero(env.captured[:2]))
    return n_cap, min_d2, hold2_time, t * env.time_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="checkpoints/sdgcr_v2_dcap20/checkpoint_010900.pth")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed-base", type=int, default=60000)
    p.add_argument("--evader-speed", type=float, default=0.35)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--pair-dcap", type=float, default=0.0,
                   help=">0 时反事实测试: 判据放宽为 >=2 名分配成员在该距离内(跳过凸包)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl, d_cap = load_controller(args.checkpoint, device)
    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6,
                            max_evaders=3, d_cap=d_cap, hold_steps=5, seed=0,
                            v_max_e=float(args.evader_speed))
    if args.pair_dcap > 0:
        pair_dcap = float(args.pair_dcap)
        def _relaxed_capture_ok(j, _self=env, _r=pair_dcap):
            coalition = _self.coalition_of(j)
            if len(coalition) < 2:
                return False
            evader_pos = _self.positions[_self.num_pursuers + j]
            pts = _self.positions[coalition]
            return bool(float(np.max(np.linalg.norm(pts - evader_pos, axis=1))) <= _r)
        env._capture_hold_ok = _relaxed_capture_ok
        print(f"[反事实] 判据放宽: >=2 名分配成员 <= {pair_dcap} m (跳过凸包)", flush=True)

    caps, mins0, mins1, hold0, hold1, times = [], [], [], [], [], []
    for k in range(args.seeds):
        seed = args.seed_base + k
        n_cap, min_d2, hold2, T = run_once(env, ctrl, seed, args.max_steps)
        caps.append(n_cap)
        mins0.append(min_d2[0])
        mins1.append(min_d2[1])
        hold0.append(hold2[0])
        hold1.append(hold2[1])
        times.append(T)
        print(f"seed {seed}: captured={n_cap}/2  min_d2=({min_d2[0]:.2f},{min_d2[1]:.2f})"
              f"  in_dcap_t=({hold2[0]:.1f}s,{hold2[1]:.1f}s)  T={T:.0f}s", flush=True)

    caps = np.asarray(caps)
    print("\n=== 4v2 适配汇总 (6v2 模型, N=4, 需求3.0→缩放到2/2) ===", flush=True)
    print(f"episode 数: {len(caps)}", flush=True)
    print(f"完整捕获: {np.sum(caps==2)}/{len(caps)}   单目标: {np.sum(caps==1)}/{len(caps)}"
          f"   (判据需>=3机, 4机2/2分配下结构性不可达, 预期为0)", flush=True)
    print(f"任务0 最近2机最小距离: mean={np.mean(mins0):.2f} min={np.min(mins0):.2f} max={np.max(mins0):.2f}", flush=True)
    print(f"任务1 最近2机最小距离: mean={np.mean(mins1):.2f} min={np.min(mins1):.2f} max={np.max(mins1):.2f}", flush=True)
    print(f"任务0 双机在d_cap内时间: mean={np.mean(hold0):.1f}s  任务1: mean={np.mean(hold1):.1f}s", flush=True)


if __name__ == "__main__":
    main()
