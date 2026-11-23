"""绘制实飞候选种子的围捕轨迹图。

在 v4_dispersed(8×8) 筛选协议下重放指定种子，记录每步位置，
输出二维 XY 轨迹图：追捕机实线、逃逸机粗虚线，起点 o、终点 x，
并在完整捕获时刻画 d_cap 圆。
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .flight_candidate_screen import load_controller
from .stage_env import StageCoalitionEnv


def record(env, ctrl, seed, max_steps=1000):
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=seed)
    ctrl.reset()
    P, E = [env.pursuer_positions().copy()], [env.evader_positions().copy()]
    done = False
    t = 0
    cap_t = None
    while not done and t < max_steps:
        actions, _ = ctrl.controller_step(env)
        _, done, _ = env.step(actions, max_steps=max_steps)
        P.append(env.pursuer_positions().copy())
        E.append(env.evader_positions().copy())
        t += 1
        if cap_t is None and env.all_captured():
            cap_t = t * env.time_step
    n_captured = int(np.count_nonzero(env.captured[:2]))
    return np.asarray(P), np.asarray(E), cap_t, n_captured, t * env.time_step


def plot(P, E, cap_t, seed, d_cap, out, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    n_p, n_e = P.shape[1], E.shape[1]
    colors_p = plt.cm.viridis(np.linspace(0.1, 0.85, n_p))
    colors_e = ["crimson", "darkorange"]
    for i in range(n_p):
        ax.plot(P[:, i, 0], P[:, i, 1], color=colors_p[i], lw=1.3,
                label=f"P{i}")
        ax.plot(P[0, i, 0], P[0, i, 1], "o", color=colors_p[i], ms=6)
        ax.plot(P[-1, i, 0], P[-1, i, 1], "x", color=colors_p[i], ms=8, mew=2)
    for j in range(n_e):
        ax.plot(E[:, j, 0], E[:, j, 1], "--", color=colors_e[j % 2], lw=2.2,
                label=f"E{j}")
        ax.plot(E[0, j, 0], E[0, j, 1], "o", color=colors_e[j % 2], ms=9,
                mec="k", mew=1.2)
        ax.plot(E[-1, j, 0], E[-1, j, 1], "s", color=colors_e[j % 2], ms=8,
                mec="k", mew=1.2)
    for obs in getattr(plot, "_obstacles", []):
        ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color="gray",
                                alpha=0.35, zorder=0))
    if cap_t is not None:
        for j in range(n_e):
            ax.add_patch(plt.Circle((E[-1, j, 0], E[-1, j, 1]), d_cap,
                                    fill=False, ls=":", ec=colors_e[j % 2],
                                    lw=1.5))
        ax.set_title(f"seed {seed}  full capture @ {cap_t:.1f}s{title_suffix}",
                     fontsize=13)
    else:
        ax.set_title(f"seed {seed}  no full capture{title_suffix}",
                     fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[50045, 50084])
    p.add_argument("--checkpoint",
                   default="checkpoints/sdgcr_v2_dcap20/checkpoint_010900.pth")
    p.add_argument("--evader-speed", type=float, default=0.35)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out-dir", default="checkpoints/traj_figures")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl, d_cap = load_controller(args.checkpoint, device)
    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6,
                            max_evaders=3, d_cap=d_cap, hold_steps=5, seed=0,
                            v_max_e=float(args.evader_speed))
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        P, E, cap_t, n_cap, T = record(env, ctrl, seed, args.max_steps)
        plot._obstacles = [(o.position[0], o.position[1], o.radius)
                           for o in env.obstacles]
        out = f"{args.out_dir}/traj_seed{seed}.png"
        plot(P, E, cap_t, seed, d_cap, out)
        print(f"seed {seed}: captured={n_cap}/2 full_t={cap_t} T={T:.1f}s "
              f"steps={len(P)}", flush=True)


if __name__ == "__main__":
    main()
