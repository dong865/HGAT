"""continuous_membership_v2 主训练入口（阶段 D 交付）。

采集、评估共用 v2_controller.V2Controller；更新走 v2_training 三阶段。
旧入口与旧 checkpoint 不受影响；本入口只写 architecture_version=
continuous_membership_v2 的 checkpoint。
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .encoder_pretraining import load_encoder_checkpoint
from .stage_env import StageCoalitionEnv, StageRewardConfig
from .stage_networks import SetInvariantCritic, SharedActor, StageFeatureEncoder
from .v2_controller import (
    ALLOCATION_SOURCE,
    ARCHITECTURE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    V2Controller,
    v2_obs_dim,
)
from .v2_training import V2Replay, V2UpdateConfig, v2_update

SCHEMA_PAYLOAD = {
    "architecture_version": ARCHITECTURE_VERSION,
    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
    "task_plan_schema_version": "continuous_ot_v1",
    "membership_schema_version": "convex_count_assignment_v1",
    "capture_definition_version": "assigned_members_geometric_v1",
    "context_pool": "normalized_co_membership_mean",
    "demand_parameterization": "scalar_softplus",
    "allocation_source": ALLOCATION_SOURCE,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", default="continuous_membership_v2")
    p.add_argument("--demand-source", choices=("learned_continuous", "fixed"),
                   default="learned_continuous")
    p.add_argument("--membership-solver", default="convex_count_assignment")
    p.add_argument("--context-pool", default="normalized_co_membership_mean")
    p.add_argument("--target-context", choices=("hard", "soft"), default="hard")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--hidden-dim", type=int, default=32, help="ST-HGAT/关系头/需求头宽度")
    p.add_argument("--actor-hidden", type=int, default=64, help="Actor/Critic 宽度（容量测试用）")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--num-pursuers", type=int, default=6)
    p.add_argument("--num-evaders", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--map-size", type=float, default=8.0)
    p.add_argument("--num-obstacles", type=int, default=10)
    p.add_argument("--d-cap", type=float, default=1.5)
    p.add_argument("--hold-steps", type=int, default=5)
    p.add_argument("--pursuer-speed", type=float, default=0.35)
    p.add_argument("--closure-weight", type=float, default=0.5)
    p.add_argument("--closure-band", type=float, default=2.5)
    p.add_argument("--ring-weight", type=float, default=0.0)
    p.add_argument("--ring-radius", type=float, default=1.5)
    p.add_argument("--angular-closure-weight", type=float, default=0.3)
    p.add_argument("--angular-closure-band", type=float, default=1.5)
    p.add_argument("--evader-speed", type=float, default=0.20)
    # 分配 / 门控
    p.add_argument("--assign-interval", type=int, default=5)
    p.add_argument("--count-penalty", type=float, default=1.0)
    p.add_argument("--switch-penalty", type=float, default=0.5)
    p.add_argument("--hysteresis", type=float, default=0.05)
    p.add_argument("--min-dwell-steps", type=int, default=5)
    p.add_argument("--epsilon-ot", type=float, default=0.5)
    p.add_argument("--sinkhorn-iters", type=int, default=80)
    p.add_argument("--demand-init", type=float, default=3.0)
    p.add_argument("--fixed-demand", type=float, default=3.0)
    # 消融（仅明确对照时使用）
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--stop-plan-gradient", action="store_true")
    # 学习
    p.add_argument("--lr-encoder", type=float, default=2e-4)
    p.add_argument("--demand-lr", type=float, default=None,
                   help="S1：需求头独立学习率（默认随 lr-encoder 联合）")
    p.add_argument("--critic-stop-plan-gradient", action="store_true",
                   help="S2：critic 阶段切断 P 梯度，需求只从 Actor 目标学习")
    p.add_argument("--demand-scale-anchor", type=float, default=0.0,
                   help="S3：尺度锚定 λ(Σd-N)²，0=关闭")
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument(
        "--actor-los-weight", type=float, default=0.0,
        help="training-only auxiliary weight for pointing each assigned agent "
             "toward its target; 0 disables it",
    )
    p.add_argument(
        "--actor-ring-weight", type=float, default=0.0,
        help="training-only auxiliary weight for distinct three-agent ring-slot "
             "directions; 0 disables it",
    )
    p.add_argument("--lr-critic", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--segment-length", type=int, default=20,
                   help="训练片段长度 L（采 L+1 条 transition）")
    p.add_argument("--max-replay-episodes", type=int, default=200)
    p.add_argument("--updates-per-episode", type=int, default=10)
    p.add_argument("--exploration-noise", type=float, default=0.2)
    p.add_argument("--final-exploration-noise", type=float, default=0.05)
    p.add_argument("--exploration-decay-episodes", type=int, default=3000)
    p.add_argument("--warmup-episodes", type=int, default=2)
    # 预训练 warmup（抗塌缩；主配置建议启用）
    p.add_argument("--resume", default=None,
                   help="从 V2 checkpoint 恢复权重并把 episode 计数续到该点之后（replay 重新暖机）")
    p.add_argument("--init-policy-from", default=None,
                   help="从 V2 checkpoint 暖启动全部网络（actor/critic/encoder/demand），"
                        "优化器与 replay 重建，episode 从 0 重新计数")
    p.add_argument("--init-encoder-from", default=None,
                   help="关系预训练 checkpoint（st_hgat+relation_head；demand 头重新初始化）")
    # 评估 / 保存
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--output", default="checkpoints/sdgcr_v2_main")
    p.add_argument("--run-name", default="sdgcr_v2_main")
    return p


def exploration_noise(episode: int, args) -> float:
    if args.exploration_decay_episodes <= 0:
        return float(args.final_exploration_noise)
    prog = min(1.0, episode / float(args.exploration_decay_episodes))
    return float(args.exploration_noise
                 + prog * (args.final_exploration_noise - args.exploration_noise))


def build_env(args) -> StageCoalitionEnv:
    """Build the training/evaluation environment with effective dynamics."""
    return StageCoalitionEnv(
        length=args.map_size,
        num_obstacle=args.num_obstacles,
        max_pursuers=6,
        max_evaders=3,
        d_cap=args.d_cap,
        hold_steps=args.hold_steps,
        seed=args.seed,
        v_max_e=float(args.evader_speed),
        reward_config=StageRewardConfig(
            closure_weight=args.closure_weight,
            closure_band=args.closure_band,
            ring_weight=args.ring_weight,
            ring_radius=args.ring_radius,
            angular_closure_weight=args.angular_closure_weight,
            angular_closure_band=args.angular_closure_band,
        ),
    )


def model_selection_key(metrics: dict) -> tuple[float, float, float, float]:
    """Prefer full capture, then any-target capture, partial capture, reward."""
    return tuple(float(metrics[name]) for name in (
        "capture_rate",
        "target_capture_rate",
        "partial_capture_rate",
        "mean_reward",
    ))


def evaluate(controller: V2Controller, env: StageCoalitionEnv, episodes: int,
             max_steps: int, seed: int) -> dict:
    captured = 0
    partial = 0
    target_cap = 0
    rewards = []
    invalid = 0
    for k in range(episodes):
        env.reset(num_pursuers=env.num_pursuers, num_evaders=env.num_evaders,
                  scenario_seed=seed + 50000 + k)
        controller.reset()
        ep_r = 0.0
        done = False
        t = 0
        while not done and t < max_steps:
            actions, info = controller.controller_step(env)
            r, done, step_info = env.step(actions, max_steps=max_steps)
            ep_r += r
            t += 1
        cap_ev = int(np.count_nonzero(env.captured[: env.num_evaders]))
        captured += int(cap_ev == env.num_evaders)
        target_cap += int(cap_ev >= 1)
        partial += int(cap_ev >= max(1, env.num_evaders - 1))
        rewards.append(ep_r)
        invalid += int(step_info.get("invalid", 0) > 0)
    n = max(1, episodes)
    return {
        "capture_rate": captured / n,
        "target_capture_rate": target_cap / n,
        "partial_capture_rate": partial / n,
        "mean_reward": float(np.mean(rewards)),
        "invalid_rate": invalid / n,
        "evader_goal_rate": 1.0 - target_cap / n,
    }


def main():
    args = build_parser().parse_args()
    assert args.architecture == "continuous_membership_v2"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(args)

    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=args.num_obstacles,
        map_size=args.map_size, hidden_dim=args.hidden_dim, future_steps=10, seq_len=10,
        node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True   # V2 不需要 encoder 内部旧需求/软计划
    if args.init_encoder_from:
        payload = load_encoder_checkpoint(args.init_encoder_from, encoder, device)
        print(f"[v2] encoder warm start from {args.init_encoder_from} "
              f"(step={payload.get('step')}); demand 头重新初始化", flush=True)
    demand_head = ContinuousDemandHead(args.hidden_dim, demand_init=args.demand_init).to(device)
    actor = SharedActor(v2_obs_dim(6, args.hidden_dim, 10), hidden_dim=args.actor_hidden).to(device)
    critic1 = SetInvariantCritic(v2_obs_dim(6, args.hidden_dim, 10), hidden_dim=args.actor_hidden).to(device)
    critic2 = SetInvariantCritic(v2_obs_dim(6, args.hidden_dim, 10), hidden_dim=args.actor_hidden).to(device)

    if args.init_policy_from:
        payload = torch.load(args.init_policy_from, map_location=device, weights_only=False)
        loaded = []
        for name, module in (("st_hgat", encoder.st_hgat),
                             ("relation_head", encoder.relation_head),
                             ("demand_head", demand_head),
                             ("actor", actor),
                             ("critic1", critic1),
                             ("critic2", critic2)):
            state = payload.get(name)
            if state is None:
                raise KeyError(f"init-policy-from checkpoint 缺少 {name}")
            module.load_state_dict(state)
            loaded.append(name)
        print(f"[v2] policy warm start from {args.init_policy_from}: "
              f"loaded={loaded}（优化器/replay 重建，episode 重新计数）", flush=True)

    encoder_target = copy.deepcopy(encoder).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    critic1_target = copy.deepcopy(critic1).to(device)
    critic2_target = copy.deepcopy(critic2).to(device)

    if args.freeze_encoder:
        for p_ in encoder.parameters():
            p_.requires_grad_(False)
        print("[v2][ablation] freeze-encoder：仅 actor/critic 学习", flush=True)
    if args.stop_plan_gradient:
        print("[v2][ablation] stop-plan-gradient：上下文入口 detach P", flush=True)

    modules = {"encoder": encoder, "demand_head": demand_head, "actor": actor,
               "critic1": critic1, "critic2": critic2, "encoder_target": encoder_target,
               "actor_target": actor_target, "critic1_target": critic1_target,
               "critic2_target": critic2_target}
    if args.demand_lr is not None:
        # S1：需求头独立优化器；encoder 组剔除 demand 头（st_hgat+relation_head 仍在内）
        encoder_params = list(encoder.parameters())
        optimizers = {
            "encoder": torch.optim.Adam(encoder_params, lr=args.lr_encoder),
            "demand": torch.optim.Adam(demand_head.parameters(), lr=args.demand_lr),
            "actor": torch.optim.Adam(actor.parameters(), lr=args.lr_actor),
            "critics": torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()),
                lr=args.lr_critic),
        }
        print(f"[v2] demand 独立优化器 lr={args.demand_lr}", flush=True)
    else:
        encoder_params = list(encoder.parameters()) + list(demand_head.parameters())
        optimizers = {
            "encoder": torch.optim.Adam(encoder_params, lr=args.lr_encoder),
            "actor": torch.optim.Adam(actor.parameters(), lr=args.lr_actor),
            "critics": torch.optim.Adam(
                list(critic1.parameters()) + list(critic2.parameters()),
                lr=args.lr_critic),
        }
    config = V2UpdateConfig(
        gamma=args.gamma, tau=args.tau, policy_delay=args.policy_delay,
        target_context=args.target_context,
        freeze_encoder=args.freeze_encoder,
        stop_plan_gradient=args.stop_plan_gradient,
        critic_stop_plan_gradient=args.critic_stop_plan_gradient,
        demand_scale_anchor=args.demand_scale_anchor,
        actor_los_weight=args.actor_los_weight,
        actor_ring_weight=args.actor_ring_weight,
        d_cap=args.d_cap,
    )
    controller = V2Controller(
        encoder, demand_head, actor, device,
        epsilon_ot=args.epsilon_ot, sinkhorn_iters=args.sinkhorn_iters,
        count_penalty=args.count_penalty, switch_penalty=args.switch_penalty,
        hysteresis=args.hysteresis, min_dwell_steps=args.min_dwell_steps,
        assign_interval=args.assign_interval,
        demand_source=args.demand_source, fixed_demand=args.fixed_demand,
        stop_plan_gradient=args.stop_plan_gradient,
        target_context=args.target_context,
    )
    replay = V2Replay(max_episodes=args.max_replay_episodes)

    start_episode = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.st_hgat.load_state_dict(payload["st_hgat"])
        encoder.relation_head.load_state_dict(payload["relation_head"])
        actor.load_state_dict(payload["actor"])
        critic1.load_state_dict(payload["critic1"])
        critic2.load_state_dict(payload["critic2"])
        encoder_target.load_state_dict(payload["encoder_target"])
        actor_target.load_state_dict(payload["actor_target"])
        critic1_target.load_state_dict(payload["critic1_target"])
        critic2_target.load_state_dict(payload["critic2_target"])
        if "demand_head" in payload:
            demand_head.load_state_dict(payload["demand_head"])
        start_episode = int(payload.get("episode", 0)) + 1
        print(f"[v2] resume from {args.resume} at episode {start_episode} "
              f"(replay 重新暖机，优化器状态重置)", flush=True)
    print(f"[v2] {ARCHITECTURE_VERSION} start; schema={OBSERVATION_SCHEMA_VERSION}; "
          f"allocation_source={ALLOCATION_SOURCE}; target_context={args.target_context}",
          flush=True)
    started = time.time()
    update_step = 0
    last_logs: dict = {}
    best_key = None
    for episode in range(start_episode, args.episodes):
        env.reset(num_pursuers=args.num_pursuers, num_evaders=args.num_evaders,
                  scenario_seed=args.seed + episode)
        controller.reset()
        transitions = []
        ep_reward = 0.0
        done = False
        t = 0
        noise = exploration_noise(episode, args)
        while not done and t < args.max_steps:
            snap = env.snapshot()
            actions, info = controller.controller_step(env)
            noisy = np.clip(actions + rng.normal(0.0, noise, size=actions.shape),
                            -1.0, 1.0).astype(np.float32)
            r, done, _ = env.step(noisy, max_steps=args.max_steps)
            transitions.append({
                "snapshot": snap,               # 决策时刻 post-assignment 状态
                "assignment": list(info["assignment"]),
                "action": noisy,
                "reward": float(r),
                "done": bool(done),
                "episode_id": episode,
                "step_id": t,
                "reconfigured": bool(info["reconfigured"]),
                "gate_reason": info["gate_reason"],
            })
            ep_reward += r
            t += 1
        replay.add_episode(transitions)

        if episode + 1 >= args.warmup_episodes:
            for _ in range(args.updates_per_episode):
                try:
                    seg = replay.sample_segment(args.segment_length, rng)
                except ValueError:
                    break
                update_step += 1
                last_logs = v2_update(
                    modules, optimizers, seg, config,
                    plan_length=args.segment_length,
                    demand_source=args.demand_source, fixed_demand=args.fixed_demand,
                    epsilon_ot=args.epsilon_ot, sinkhorn_iters=args.sinkhorn_iters,
                    max_pursuers=6, max_evaders=3,
                    update_index=update_step,
                )

        if episode % max(1, args.eval_interval) == 0 or episode == args.episodes - 1:
            metrics = evaluate(controller, env, args.eval_episodes,
                               args.max_steps, args.seed)
            (out / "latest_eval.json").write_text(json.dumps(
                {"episode": episode, **metrics}, indent=2))
            key = model_selection_key(metrics)
            if best_key is None or key > best_key:
                best_key = key
                torch.save(_ckpt_payload(modules, args, episode, metrics),
                           out / "best_model.pth")
            dem = last_logs and controller.last_info.get("demand_raw")
            print(
                f"[v2] ep={episode}/{args.episodes - 1} reward={ep_reward:.2f} "
                f"cap={metrics['capture_rate']:.3f} tgt={metrics['target_capture_rate']:.3f} "
                f"part={metrics['partial_capture_rate']:.3f} "
                f"evalR={metrics['mean_reward']:.1f} inv={metrics['invalid_rate']:.2f} "
                f"upd={update_step} "
                f"lc={last_logs.get('loss/critic', float('nan')):.4g} "
                f"la={last_logs.get('loss/actor', float('nan')):.4g} "
                f"lr_={last_logs.get('loss/relation', float('nan')):.4g} "
                f"dem={[round(float(d), 2) for d in (dem or [])]} "
                f"noise={noise:.3f} elapsed={time.time() - started:.1f}s",
                flush=True)
            torch.save(_ckpt_payload(modules, args, episode, metrics),
                       out / f"checkpoint_{episode:06d}.pth")

    torch.save(_ckpt_payload(modules, args, args.episodes - 1, {}),
               out / "final_model.pth")


def _ckpt_payload(modules: Dict[str, torch.nn.Module], args, episode: int,
                  metrics: dict) -> dict:
    payload = dict(SCHEMA_PAYLOAD)
    payload.update({
        "target_context": args.target_context,
        "hidden_dim": int(args.hidden_dim), "future_steps": 10, "seq_len": 10,
        "d_cap": args.d_cap, "hold_steps": args.hold_steps,
        "map_size": args.map_size, "num_obstacles": args.num_obstacles,
        "max_pursuers": 6, "max_evaders": 3,
        "num_pursuers": args.num_pursuers, "num_evaders": args.num_evaders,
        "episode": int(episode),
        "metrics": dict(metrics),
        "args": vars(args),
        "st_hgat": modules["encoder"].st_hgat.state_dict(),
        "relation_head": modules["encoder"].relation_head.state_dict(),
        "demand_head": modules["demand_head"].state_dict(),
        "actor": modules["actor"].state_dict(),
        "critic1": modules["critic1"].state_dict(),
        "critic2": modules["critic2"].state_dict(),
        "encoder_target": modules["encoder_target"].state_dict(),
        "actor_target": modules["actor_target"].state_dict(),
        "critic1_target": modules["critic1_target"].state_dict(),
        "critic2_target": modules["critic2_target"].state_dict(),
    })
    return payload


if __name__ == "__main__":
    main()
