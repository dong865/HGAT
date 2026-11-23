"""绘制 V3 模型的 4v2 围捕轨迹图（deploy5x5 协议）。

重放指定种子，记录每步位置，输出 XY 轨迹：
追捕机实线(P0-P3)、逃逸机粗虚线(E0/E1)，o=起点 x/■=终点，
灰盘=障碍，红/橙点线圈=捕获时刻 d_cap 圆。
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate_v3_hybrid import build_evaluation_env, load_v3_system
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .train_v3_hybrid import HybridPolicy


def record(env, policy, seed, max_steps, n_pursuers=4):
    env.reset(num_pursuers=n_pursuers, num_evaders=2, scenario_seed=seed)
    policy.reset()
    P = [env.pursuer_positions()[:n_pursuers].copy()]
    E = [env.evader_positions()[:2].copy()]
    t = 0
    cap_t = None
    decision = policy.decision(env)
    while t < max_steps:
        actions = decision["actions"]
        _, done, info = env.step(actions, max_steps=max_steps)
        P.append(env.pursuer_positions()[:n_pursuers].copy())
        E.append(env.evader_positions()[:2].copy())
        t += 1
        if cap_t is None and info.get("capture_events"):
            cap_t = env.episode_step_count * env.time_step
        if done:
            break
        decision = policy.decision(env)
    n_cap = int(np.count_nonzero(env.captured[:2]))
    return np.asarray(P), np.asarray(E), cap_t, n_cap


def plot(P, E, cap_t, n_cap, seed, d_cap, obstacles, out, crop=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    n_p, n_e = P.shape[1], E.shape[1]
    colors_p = plt.cm.viridis(np.linspace(0.1, 0.85, n_p))
    colors_e = ["crimson", "darkorange"]
    for i in range(n_p):
        ax.plot(P[:, i, 0], P[:, i, 1], color=colors_p[i], lw=1.4, label=f"P{i}")
        ax.plot(P[0, i, 0], P[0, i, 1], "o", color=colors_p[i], ms=7)
        ax.plot(P[-1, i, 0], P[-1, i, 1], "x", color=colors_p[i], ms=9, mew=2.2)
    for j in range(n_e):
        ax.plot(E[:, j, 0], E[:, j, 1], "--", color=colors_e[j], lw=2.4, label=f"E{j}")
        ax.plot(E[0, j, 0], E[0, j, 1], "o", color=colors_e[j], ms=10, mec="k", mew=1.2)
        ax.plot(E[-1, j, 0], E[-1, j, 1], "s", color=colors_e[j], ms=9, mec="k", mew=1.2)
        for g in getattr(plot, "_goals", [])[j:j+1]:
            ax.plot(g[0], g[1], "*", color=colors_e[j], ms=15, mec="k")
    for x, y, r, *_ in obstacles:
        if crop and not (crop[0] - 0.6 <= x <= crop[1] + 0.6 and crop[2] - 0.6 <= y <= crop[3] + 0.6):
            continue  # 只画可飞体积内的真实障碍
        ax.add_patch(plt.Circle((x, y), r, color="gray", alpha=0.35, zorder=0))
    for j in range(min(n_cap, n_e)):
        ax.add_patch(plt.Circle((E[-1, j, 0], E[-1, j, 1]), d_cap, fill=False,
                                ls=":", ec=colors_e[j], lw=1.6))
    box = plt.Rectangle((2, 2), 4, 4, fill=False, ls="-.", ec="navy", lw=1.0)
    ax.add_patch(box)
    if crop:
        ax.add_patch(plt.Rectangle((crop[0], crop[2]), crop[1]-crop[0], crop[3]-crop[2],
                                   fill=False, ec="navy", lw=1.4))
    if n_cap == n_e:
        title = f"seed {seed}  FULL capture ({n_cap}/{n_e}) @ {cap_t:.1f}s"
    elif n_cap > 0:
        title = f"seed {seed}  partial capture ({n_cap}/{n_e}) @ {cap_t:.1f}s"
    else:
        title = f"seed {seed}  no capture"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x (m, 8m protocol frame)"); ax.set_ylabel("y (m)")
    if crop:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
    else:
        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/v3_hybrid_4v2_flight/best_4v2.pth")
    p.add_argument("--seeds", type=int, nargs="+", default=[70001, 70000])
    p.add_argument("--out-dir", default="checkpoints/traj_figures_v3")
    p.add_argument("--crop", choices=("none", "volume"), default="none",
                   help="volume=裁到5x5可飞体积[1.5,6.5]并只画体积内障碍")
    p.add_argument("--d-cap", type=float, default=None, help="覆盖捕获半径（默认取ckpt config）")
    p.add_argument("--no-deploy-5x5", action="store_true", help="用默认v4采样器(8x8)代替deploy5x5")
    p.add_argument("--v-mult", type=float, default=1.0, help="追捕机速度倍率(默认1.0同速)")
    p.add_argument("--num-pursuers", type=int, default=4)
    args = p.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modules, config = load_v3_system(args.checkpoint, device=device)
    env = build_evaluation_env(config, v_max_e=config.get("v_max_e", 0.35), seed=0)
    if not args.no_deploy_5x5:
        env.scenario_sampler = Deploy5x5Sampler()
    if args.d_cap is not None:
        env.d_cap = float(args.d_cap)
    if args.v_mult != 1.0:
        env.v_max = env.v_max * float(args.v_mult)
    policy = HybridPolicy(modules, config)
    d_cap = float(args.d_cap) if args.d_cap is not None else config.get("d_cap", 2.0)

    for seed in args.seeds:
        P, E, cap_t, n_cap = record(env, policy, seed, config.get("max_steps", 2000), args.num_pursuers)
        plot._goals = [env.evader_goals[j][:2].copy() for j in range(2)]
        obstacles = [(o.position[0], o.position[1], o.radius, o.height) for o in env.obstacles]
        crop = (1.5, 6.5, 1.5, 6.5) if args.crop == "volume" else None
        out = f"{args.out_dir}/v3_traj_seed{seed}{'_5x5' if crop else ''}.png"
        plot(P, E, cap_t, n_cap, seed, d_cap, obstacles, out, crop=crop)
        print(f"seed {seed}: captured={n_cap}/2 full_t={cap_t}", flush=True)


if __name__ == "__main__":
    main()
