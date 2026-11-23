"""Offline diagnostic: load a dcap15 checkpoint and measure representation health.

Read-only: builds modules, loads state dicts from the checkpoint, runs a few
fresh env snapshots, and reports the metrics that the (disabled) TensorBoard
would normally show -- obs scale, assignment entropy/capacity error, demand
distribution, action norms/saturation, and Q-value scale.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if (_HERE / "stage_env.py").exists():
    sys.path.insert(0, str(_HERE.parent))

from crazyflie_examples.six_v_two_hgat.stage_env import StageCoalitionEnv
from crazyflie_examples.six_v_two_hgat.stage_networks import (
    SharedActor,
    SetInvariantCritic,
    StageFeatureEncoder,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--d-cap", type=float, default=1.5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=8, help="fresh scenarios to probe")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    print(f"checkpoint episode: {ckpt.get('episode')}")
    print(f"saved d_cap={saved.get('d_cap')}, run args d_cap={args.d_cap}")

    env = StageCoalitionEnv(
        length=args.map_size, num_obstacle=args.num_obstacles,
        max_pursuers=6, max_evaders=3, d_cap=args.d_cap,
    )
    env.v_max_e = 0.2  # match training config
    encoder = StageFeatureEncoder(
        hidden_dim=args.hidden_dim, node_dim=9, max_pursuers=6, max_evaders=3,
        future_steps=args.future_steps, seq_len=args.seq_len,
        map_size=args.map_size, num_obstacles=args.num_obstacles, device=device,
        use_ring_slot_features=True,          # main_stage_end_to_end hardcodes this
        output_layer_norm=True, output_gain=0.8,
    )
    actor = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    critic = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)

    encoder.st_hgat.load_state_dict(ckpt["st_hgat"])
    encoder.relation_head.load_state_dict(ckpt["relation_head"])
    encoder.demand_head.load_state_dict(ckpt["demand_head"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])

    encoder.eval(); actor.eval(); critic.eval()

    rng = np.random.default_rng(args.seed)
    stats = {k: [] for k in [
        "obs_abs_mean", "obs_std", "entropy", "capacity_err", "demand",
        "rel_cost_mean", "rel_cost_std", "act_norm", "saturation",
        "q_policy", "q_rand",
    ]}

    for ep in range(args.episodes):
        env.reset(num_pursuers=6, num_evaders=2, scenario_seed=int(rng.integers(2**31)))
        encoder.reset(step=0)
        for t in range(30):
            snap = env.snapshot()
            with torch.no_grad():
                feats = encoder.encode(snap, differentiable=True)
                obs = feats["obs"].unsqueeze(0)
                masks = feats["masks"].unsqueeze(0)
                plan = feats["soft_assignment"]
                tmask = torch.as_tensor(snap["task_mask"], device=device)[: env.num_evaders]
                acts = actor(obs)
                q_pol = critic(obs, acts, masks)
                q_rnd = critic(obs, torch.rand_like(acts) * 2 - 1, masks)

            active = plan[:, : env.num_evaders] * tmask.unsqueeze(0)
            probs = active / active.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            ent = (-(probs.clamp_min(1e-8).log() * probs).sum(-1)).mean()
            cap_err = (active.sum(dim=0) - feats["demand_hat"][: env.num_evaders]).abs().mean()

            stats["obs_abs_mean"].append(float(obs.abs().mean()))
            stats["obs_std"].append(float(obs.std()))
            stats["entropy"].append(float(ent))
            stats["capacity_err"].append(float(cap_err))
            stats["demand"].append(feats["demand_hat"][: env.num_evaders].mean().item())
            rc = feats["relation_cost"][:, : env.num_evaders]
            stats["rel_cost_mean"].append(float(rc.mean()))
            stats["rel_cost_std"].append(float(rc.std()))
            stats["act_norm"].append(float(torch.linalg.vector_norm(acts, dim=-1).mean()))
            stats["saturation"].append(float((acts.abs() >= 0.98).float().mean()))
            stats["q_policy"].append(float(q_pol.mean()))
            stats["q_rand"].append(float(q_rnd.mean()))
            env.step(acts.cpu().numpy()[0])

    print("\n=== diagnostic averages over {} episodes x 30 steps ===".format(args.episodes))
    for k, v in stats.items():
        if v:
            print(f"{k:16s}: mean={np.mean(v):10.5g}  std={np.std(v):10.5g}  "
                  f"min={np.min(v):10.5g}  max={np.max(v):10.5g}")

    print("\n=== interpretation hints ===")
    print("- obs_abs_mean/std: output LayerNorm+gain 0.8 should pin obs_std near 0.8.")
    print("- entropy: near 0 => plan collapsed (one-hot); near ln(2)~0.69 => uniform.")
    print("- capacity_err: how far column sums are from predicted demand.")
    print("- q scale: with critic_reward_scale=0.1 and sparse +600 capture bonus,")
    print("  |Q| in the ones range is healthy; 1e-2 or 1e+3 indicates drift.")
    print("- saturation: fraction of |a|>=0.98; >0.2 is concerning.")


if __name__ == "__main__":
    main()
