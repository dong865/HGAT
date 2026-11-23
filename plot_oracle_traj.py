"""oracle（锁定静态分配）轨迹绘制：记录每步位置并出 XY 图。"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .train_v3_hybrid import build_system, load_v3_hybrid_checkpoint
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment


def load_modules(path, device):
    payload = load_v3_hybrid_checkpoint(path)
    config = dict(payload["config"])
    for k in ("init_policy_from", "init_value_from", "init_v3_from"):
        config[k] = None
    config.pop("warm_start_v3_from", None)
    config["device"] = str(device)
    modules, _, _ = build_system(config)
    load_v3_hybrid_checkpoint(path, modules)
    return modules, config


def record_oracle(modules, config, seed, assignment, d_cap=0.8, max_steps=700):
    from .train_v3_hybrid import make_environment
    env = make_environment(config, d_cap=d_cap)
    env.scenario_sampler = Deploy5x5Sampler()
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=int(seed))
    modules["encoder"].reset()
    P = [env.pursuer_positions()[:4].copy()]
    E = [env.evader_positions()[:2].copy()]
    cap_events = []
    t = 0
    for step in range(max_steps):
        active = np.flatnonzero(env.task_mask[:2])
        if not len(active):
            break
        cur = np.asarray(assignment, dtype=np.int64).copy()
        cur[~np.isin(cur, active)] = int(active[0])
        apply_membership(env, cur)
        snapshot = env.snapshot()
        with torch.no_grad():
            encoded = modules["encoder"].encode(snapshot, differentiable=False)
            membership = hard_K_from_assignment(cur, env.task_mask, 6, 3,
                                                encoded["pursuer_embeddings"].device)
            obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
            mask = torch.as_tensor(env.pursuer_mask, dtype=torch.float32, device=obs.device)
            actions = (modules["actor"](obs) * mask.unsqueeze(-1)).cpu().numpy()
        _, done, info = env.step(actions, max_steps=max_steps)
        P.append(env.pursuer_positions()[:4].copy())
        E.append(env.evader_positions()[:2].copy())
        t += 1
        if info.get("capture_events"):
            cap_events.append((t * env.time_step, list(info["capture_events"])))
        if done:
            break
    return np.asarray(P), np.asarray(E), cap_events, env


def plot(P, E, cap_events, seed, assignment, d_cap, env, out):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors_p = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
    colors_e = ["crimson", "darkorange"]
    for i in range(4):
        ax.plot(P[:, i, 0], P[:, i, 1], color=colors_p[i], lw=1.4, label=f"P{i}")
        ax.plot(P[0, i, 0], P[0, i, 1], "o", color=colors_p[i], ms=7)
        ax.plot(P[-1, i, 0], P[-1, i, 1], "x", color=colors_p[i], ms=9, mew=2.2)
    for j in range(2):
        ax.plot(E[:, j, 0], E[:, j, 1], "--", color=colors_e[j], lw=2.4, label=f"E{j}")
        ax.plot(E[0, j, 0], E[0, j, 1], "o", color=colors_e[j], ms=10, mec="k", mew=1.2)
        ax.plot(E[-1, j, 0], E[-1, j, 1], "s", color=colors_e[j], ms=9, mec="k", mew=1.2)
        ax.plot(env.evader_goals[j][0], env.evader_goals[j][1], "*", color=colors_e[j], ms=15, mec="k")
    for o in env.obstacles:
        if 1.2 <= o.position[0] <= 6.8 and 1.2 <= o.position[1] <= 6.8:
            ax.add_patch(plt.Circle((o.position[0], o.position[1]), o.radius,
                                    color="gray", alpha=0.35, zorder=0))
    ax.add_patch(plt.Rectangle((2, 2), 4, 4, fill=False, ls="-.", ec="navy", lw=1.0))
    ax.add_patch(plt.Rectangle((1.5, 1.5), 5, 5, fill=False, ec="navy", lw=1.4))
    caps = "; ".join(f"t={t:.1f}s E{ev}" for t, ev in cap_events)
    ax.set_title(f"seed {seed}  ORACLE locked {list(assignment)}  d_cap={d_cap}\n{caps}", fontsize=12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(1.5, 6.5); ax.set_ylim(1.5, 6.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/v3_08_parallel/promoted/actor/winner.pth")
    p.add_argument("--seed", type=int, default=73016)
    p.add_argument("--assignment", type=int, nargs=4, default=[1, 0, 1, 1])
    p.add_argument("--d-cap", type=float, default=0.8)
    p.add_argument("--out", default="checkpoints/v3_08_parallel/evaluation/oracle_traj_73016.png")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modules, config = load_modules(args.checkpoint, device)
    P, E, cap_events, env = record_oracle(modules, config, args.seed, args.assignment, args.d_cap)
    print(f"captured={int(env.captured[:2].sum())} events={cap_events}")
    plot(P, E, cap_events, args.seed, args.assignment, args.d_cap, env, args.out)


if __name__ == "__main__":
    main()
