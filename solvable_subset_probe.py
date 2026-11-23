"""学习模型的"可解子集"刻画：哪些场景它能解、靠什么特征区分。

对每局记录出生几何与结局，按特征分箱统计成功率，找出判别阈值。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .v2_controller import V2Controller, v2_obs_dim


def birth_features(env: StageCoalitionEnv):
    """每目标：最近3机距离均值、第3近机距离、3机集结跨度、目标距离。"""
    pos = env.pursuer_positions()[:, :2]
    ev = env.evader_positions()[:, :2]
    feats = []
    for j in range(env.num_evaders):
        d = np.sort(np.linalg.norm(pos - ev[j], axis=1))
        near3 = d[:3]
        idx3 = np.argsort(np.linalg.norm(pos - ev[j], axis=1))[:3]
        span = max(
            np.linalg.norm(pos[a] - pos[b])
            for a in idx3 for b in idx3 if a < b
        ) if len(idx3) >= 2 else 0.0
        goal = np.linalg.norm(env.evader_goals[j][:2] - ev[j][:2])
        feats.append(dict(mean3=float(near3.mean()), third=float(near3[2]),
                          span=float(span), goal=float(goal)))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sdgcr_v2_dcap20/checkpoint_010900.pth")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evader-speed", type=float, default=0.35)
    p.add_argument("--max-steps", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden = int(ck.get("hidden_dim", 32))
    d_cap = float(ck.get("d_cap", 2.0))
    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6, max_evaders=3,
                            d_cap=d_cap, hold_steps=5, seed=args.seed,
                            v_max_e=float(args.evader_speed))
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=10, map_size=8.0,
        hidden_dim=hidden, future_steps=10, seq_len=10, node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True).to(device)
    encoder._skip_legacy_plan = True
    demand_head = ContinuousDemandHead(hidden, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, hidden, 10), hidden_dim=64).to(device)
    encoder.st_hgat.load_state_dict(ck["st_hgat"])
    encoder.relation_head.load_state_dict(ck["relation_head"])
    demand_head.load_state_dict(ck["demand_head"])
    actor.load_state_dict(ck["actor"])
    encoder.eval(); demand_head.eval(); actor.eval()
    controller = V2Controller(encoder, demand_head, actor, device,
                              demand_source="fixed", fixed_demand=3.0)

    rows = []
    for k in range(args.episodes):
        env.reset(num_pursuers=6, num_evaders=2, scenario_seed=args.seed + 50000 + k)
        f = birth_features(env)
        controller.reset()
        done = False
        t = 0
        while not done and t < args.max_steps:
            actions, _ = controller.controller_step(env)
            _, done, _ = env.step(actions, max_steps=args.max_steps)
            t += 1
        ce = int(np.count_nonzero(env.captured[:2]))
        rows.append(dict(
            full=int(ce == 2), single=int(ce >= 1),
            worst_third=max(x["third"] for x in f),      # 每目标第3近机距离取最差
            worst_mean3=max(x["mean3"] for x in f),
            worst_span=max(x["span"] for x in f),
            worst_goal=max(x["goal"] for x in f),
        ))

    n = len(rows)
    full_rate = np.mean([r["full"] for r in rows])
    single_rate = np.mean([r["single"] for r in rows])
    print(f"\n=== 总成功率: full={full_rate:.2f} single={single_rate:.2f} (n={n}) ===")

    for key in ("worst_third", "worst_mean3", "worst_span", "worst_goal"):
        vals = np.array([r[key] for r in rows])
        order = np.argsort(vals)
        third = n // 3
        print(f"\n[{key}] 按三分位分箱:")
        for b in range(3):
            idx = order[b * third:(b + 1) * third] if b < 2 else order[b * third:]
            sub = [rows[i] for i in idx]
            print(f"  箱{b+1} {vals[idx].min():.2f}-{vals[idx].max():.2f}m: "
                  f"full={np.mean([r['full'] for r in sub]):.2f} "
                  f"single={np.mean([r['single'] for r in sub]):.2f} (n={len(sub)})")
        # 找 best 阈值：单目标率 >=0.5 的最宽子集
        best = None
        for thr in np.quantile(vals, np.linspace(0.1, 0.9, 17)):
            sub = [r for r in rows if r[key] <= thr]
            if len(sub) >= 10:
                s = np.mean([r["single"] for r in sub])
                if s >= 0.5 and (best is None or len(sub) > best[1]):
                    best = (float(thr), len(sub), float(s))
        if best:
            print(f"  → 可解子集: {key}<={best[0]:.2f} 覆盖 {best[1]/n:.0%} 分布, "
                  f"子集内 single={best[2]:.2f}")

    with open("checkpoints/solvable_subset_rows.json", "w") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()
