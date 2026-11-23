"""Experimental raw-sequence end-to-end SDGCR training probe."""

import argparse
import copy
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .encoder_pretraining import load_encoder_checkpoint
from .end_to_end_training import (
    EndToEndUpdateConfig,
    RawTrajectoryBuffer,
    end_to_end_update,
)
from .main_stage_feasible import (
    evaluate, reward_config_from_args, scale_reward_for_critic, write_eval_summary,
)
from .stage_coalitions import StageCoalitionAllocator
from .stage_env import StageCoalitionEnv, StageRewardConfig
from .stage_networks import SetInvariantCritic, SharedActor, StageFeatureEncoder
from .training_protocol import DEFAULT_D_CAP, DEFAULT_HOLD_STEPS


def exploration_noise_at_episode(
    episode: int, decay_episodes: int, initial: float, final: float,
) -> float:
    """Linearly anneal exploration while retaining a non-zero deployment margin."""
    if decay_episodes <= 0:
        return float(final)
    progress = float(np.clip(float(episode) / float(decay_episodes), 0.0, 1.0))
    if progress >= 1.0:
        return float(final)
    return float(initial + progress * (final - initial))


def model_selection_key(metrics: dict) -> tuple:
    """Rank checkpoints by task success before proxy reward."""
    return (
        float(metrics.get("capture_rate", 0.0)),
        float(metrics.get("target_capture_rate", 0.0)),
        float(metrics.get("partial_capture_rate", 0.0)),
        float(metrics.get("near_capture_rate", 0.0)),
        -float(metrics.get("invalid_rate", 0.0)),
        float(metrics.get("mean_reward", float("-inf"))),
    )


def validate_training_args(args: argparse.Namespace) -> None:
    if int(args.segment_length) <= int(args.future_steps):
        raise ValueError(
            "--segment-length must be greater than --future-steps so relation "
            "targets exist inside every sampled sequence"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # No source means a genuine from-scratch run: encoder, actor and critics
    # all keep their random initialization.  Supplying both sources remains
    # invalid because resume and encoder-only warm start have different
    # semantics.
    initialization = parser.add_mutually_exclusive_group(required=False)
    initialization.add_argument("--init-encoder-from")
    initialization.add_argument(
        "--resume-from",
        help="continue from a complete end-to-end checkpoint (all online and target networks)",
    )
    parser.add_argument(
        "--freeze-encoder", action="store_true",
        help="matched-pipeline control: load the encoder checkpoint and freeze "
             "ST-HGAT plus relation head, so only Actor/Critics learn. Everything "
             "else (segments, phases, losses) matches the joint-update arm, which "
             "isolates the effect of encoder gradients.",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="freeze ONLY the ST-HGAT backbone (pretrained features stay diverse); "
             "relation and demand heads remain trainable so the soft allocation "
             "still receives policy gradients through P_t. Empirically the backbone "
             "re-collapses under joint RL even from a warmup start (cos-dist "
             "0.30->0.05 within 1200 episodes at lr 2e-4).",
    )
    parser.add_argument(
        "--encoder-output-norm", action="store_true",
        help="LayerNorm the encoder output. Pins the observation scale, which "
             "otherwise drifts up ~40%% and drives the actor into tanh saturation.",
    )
    parser.add_argument(
        "--logit-action-penalty", action="store_true",
        help="apply the action-magnitude penalty to pre-tanh logits so it keeps "
             "gradient in the saturated region",
    )
    parser.add_argument(
        "--action-norm-weight", type=float, default=0.1,
        help="weight of the action-magnitude penalty. 0.1 was too weak to stop "
             "the actor's output gain growing (norm kept climbing past 1.0 even "
             "with the pre-tanh form), so this is the knob for B''.",
    )
    parser.add_argument(
        "--encoder-output-gain", type=float, default=1.0,
        help="scale applied after --encoder-output-norm; LayerNorm alone only "
             "pins the scale at unit variance (~0.8), so a gain < 1 is what "
             "actually lowers it",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-pursuers", type=int, default=6, choices=(6,))
    parser.add_argument("--num-evaders", type=int, default=2, choices=(2,))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--d-cap", type=float, default=DEFAULT_D_CAP)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--pursuer-speed", type=float, default=0.35)
    parser.add_argument("--evader-speed", type=float, default=0.20)
    parser.add_argument(
        "--closure-weight", type=float, default=1.0,
        help="dense reward weight for contracting an assigned coalition's max "
             "distance inside the closure band (capture rule d_cap/hold unchanged); "
             "0.0 = off",
    )
    parser.add_argument("--closure-band", type=float, default=3.0,
                        help="meters inside which closure contraction is rewarded")
    parser.add_argument(
        "--angular-closure-weight", type=float, default=1.0,
        help="potential-difference reward for shrinking the best nearby "
             "three-agent maximum angular gap toward pi; 0.0 = off",
    )
    parser.add_argument(
        "--angular-closure-band", type=float, default=2.5,
        help="maximum pursuer-evader distance included in angular shaping",
    )
    parser.add_argument(
        "--ring-weight", type=float, default=0.0,
        help="potential-difference reward for converging to an evenly spaced "
             "ring around the looked-ahead evader (IBR-style); 0.0 = off",
    )
    parser.add_argument("--ring-radius", type=float, default=1.5,
                        help="desired ring radius (m) for ring shaping")
    parser.add_argument("--ring-lookahead", type=float, default=0.4,
                        help="evader-velocity lookahead factor for the ring center")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--segment-length", type=int, default=32)
    parser.add_argument("--max-replay-episodes", type=int, default=200)
    parser.add_argument("--max-elite-episodes", type=int, default=40)
    parser.add_argument("--success-replay-fraction", type=float, default=0.25,
                        help="probability of drawing a segment from a successful episode")
    parser.add_argument(
        "--elite-replay-fraction", type=float, default=0.25,
        help="probability of drawing a retained near-capture episode",
    )
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument(
        "--policy-delay", type=int, default=2,
        help="apply Actor and target-network updates every N critic updates; "
             "use 2 for the standard TD3 delayed-policy schedule",
    )
    parser.add_argument(
        "--target-encoder", action=argparse.BooleanOptionalAction, default=True,
        help="use a Polyak-averaged encoder for TD bootstrap targets; disable "
             "only for the legacy-behavior causal control",
    )
    parser.add_argument(
        "--policy-anchor-weight", type=float, default=0.0,
        help="MSE penalty to a frozen copy of the resume-point Actor; this "
             "limits catastrophic policy drift without freezing the encoder",
    )
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.20)
    parser.add_argument("--final-exploration-noise", type=float, default=0.05)
    parser.add_argument("--exploration-decay-episodes", type=int, default=3000)
    parser.add_argument("--critic-reward-scale", type=float, default=0.1)
    parser.add_argument("--critic-reward-clip", type=float, default=10.0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output", default="checkpoints/end_to_end_6v2_probe")
    parser.add_argument("--tensorboard-dir", default="runs")
    parser.add_argument("--run-name", default="SDGCR_end_to_end_6v2_probe")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return resolved


def _save(
    path: Path, modules, args, episode: int, metrics, *, reward_config=None,
    optimizers=None, replay=None, update_step: int = 0, total_env_steps: int = 0,
    rng=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 3,
        "method": "sdgcr_end_to_end_probe",
        "args": vars(args),
        "reward_config": (
            reward_config.to_dict() if reward_config is not None
            else reward_config_from_args(args).to_dict()
        ),
        "episode": int(episode),
        "metrics": dict(metrics),
        "st_hgat": modules["encoder"].st_hgat.state_dict(),
        "relation_head": modules["encoder"].relation_head.state_dict(),
        "demand_head": modules["encoder"].demand_head.state_dict(),
        "actor": modules["actor"].state_dict(),
        "actor_anchor": modules["actor_anchor"].state_dict(),
        "actor_target": modules["actor_target"].state_dict(),
        "critic": modules["critic1"].state_dict(),
        "critic2": modules["critic2"].state_dict(),
        "critic_target": modules["critic1_target"].state_dict(),
        "critic2_target": modules["critic2_target"].state_dict(),
    }
    if "encoder_target" in modules:
        payload["encoder_target"] = modules["encoder_target"].state_dict()
    if "encoder_anchor" in modules:
        payload["encoder_anchor"] = modules["encoder_anchor"].state_dict()
    if optimizers is not None and replay is not None and rng is not None:
        payload["training_state"] = {
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in optimizers.items()
            },
            "replay": replay.state_dict(),
            "update_step": int(update_step),
            "total_env_steps": int(total_env_steps),
            "rng_state": copy.deepcopy(rng.bit_generator.state),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
    torch.save(payload, path)


def load_training_checkpoint(path, modules, device: torch.device) -> int:
    """Restore every learned online/target module and return the next episode id.

    The original 400-episode probe checkpoints predate optimizer persistence, so
    continuation intentionally starts fresh Adam moments while preserving every
    learned network tensor. Existing checkpoint files are never modified.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    key_to_module = {
        "st_hgat": modules["encoder"].st_hgat,
        "relation_head": modules["encoder"].relation_head,
        "actor": modules["actor"],
        "actor_target": modules["actor_target"],
        "critic": modules["critic1"],
        "critic2": modules["critic2"],
        "critic_target": modules["critic1_target"],
        "critic2_target": modules["critic2_target"],
    }
    missing = [key for key in key_to_module if key not in payload]
    if missing:
        raise KeyError(f"resume checkpoint missing keys: {missing}")
    for key, module in key_to_module.items():
        module.load_state_dict(payload[key])
    if "demand_head" in payload:
        modules["encoder"].demand_head.load_state_dict(payload["demand_head"])
    else:
        print("[migration] legacy checkpoint has no demand_head; using initialized weights")
    def _load_encoder(module, state):
        missing, _ = module.load_state_dict(state, strict=False)
        non_norm_missing = [k for k in missing if not k.startswith("output_norm")]
        if non_norm_missing:
            raise RuntimeError(f"encoder loading missing keys: {non_norm_missing}")

    if "encoder_target" in modules:
        _load_encoder(
            modules["encoder_target"],
            payload.get("encoder_target", modules["encoder"].state_dict()),
        )
    if "actor_anchor" in modules:
        modules["actor_anchor"].load_state_dict(
            payload.get("actor_anchor", payload["actor"]),
        )
    if "encoder_anchor" in modules:
        if "encoder_anchor" in payload:
            _load_encoder(modules["encoder_anchor"], payload["encoder_anchor"])
        else:
            # Legacy checkpoints did not persist a reference encoder.  Build a
            # coherent functional anchor at the exact resume point.  Its saved
            # Actor-only anchor belongs to a different representation and must
            # not be paired with this newly created reference encoder.
            modules["encoder_anchor"].load_state_dict(modules["encoder"].state_dict())
            if "actor_anchor" in modules:
                modules["actor_anchor"].load_state_dict(modules["actor"].state_dict())
    return int(payload.get("episode", -1)) + 1


def load_full_training_checkpoint(
    path, modules, optimizers, replay, rng, device: torch.device,
) -> dict:
    """Restore network and continuation state, with legacy fallback."""
    start_episode = load_training_checkpoint(path, modules, device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("training_state")
    if state is None:
        return {
            "start_episode": start_episode,
            "update_step": 0,
            "total_env_steps": 0,
            "full_state_restored": False,
            "reward_config": payload.get("reward_config"),
        }
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    replay.load_state_dict(state["replay"])
    rng.bit_generator.state = copy.deepcopy(state["rng_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_state = state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
    return {
        "start_episode": start_episode,
        "update_step": int(state["update_step"]),
        "total_env_steps": int(state["total_env_steps"]),
        "full_state_restored": True,
        "reward_config": payload.get("reward_config"),
    }


def _load_reward_config_from_checkpoint(path: str) -> Optional[dict]:
    """Peek at the reward configuration stored in a checkpoint without loading networks."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return payload.get("reward_config")


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_training_args(args)
    if args.resume_from is not None:
        saved_reward_config = _load_reward_config_from_checkpoint(args.resume_from)
        if saved_reward_config is not None:
            for key, value in saved_reward_config.items():
                if hasattr(args, key):
                    setattr(args, key, float(value))
            print(f"[resume] restored reward_config: {saved_reward_config}")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))

    env = StageCoalitionEnv(
        length=args.map_size, num_obstacle=args.num_obstacles,
        max_pursuers=6, max_evaders=3, d_cap=args.d_cap,
        hold_steps=args.hold_steps, v_max=args.pursuer_speed, seed=args.seed,
        reward_config=reward_config_from_args(args),
    )
    env.v_max_e = args.evader_speed
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=args.num_obstacles,
        map_size=args.map_size, hidden_dim=args.hidden_dim,
        future_steps=args.future_steps, seq_len=args.seq_len, node_dim=9,
        device=device, use_ring_slot_features=True,
        output_layer_norm=bool(args.encoder_output_norm),
        output_gain=float(args.encoder_output_gain),
    ).to(device)
    if args.init_encoder_from is not None:
        load_encoder_checkpoint(args.init_encoder_from, encoder, device)
    if args.freeze_encoder:
        # Matched-pipeline control: freeze after warm-start so the
        # only difference from the joint-update arm is whether ST-HGAT receives
        # gradients. The forward pass stays differentiable so the graph shape and
        # the Actor/Critic gradient paths are unchanged.
        for parameter in encoder.auxiliary_parameters():
            parameter.requires_grad_(False)
        print("[freeze] ST-HGAT/relation head frozen; only actor/critic learn")
    elif args.freeze_backbone:
        for parameter in encoder.st_hgat.parameters():
            parameter.requires_grad_(False)
        print("[freeze-backbone] ST-HGAT frozen; relation/demand heads + actor/critic learn")
    else:
        print("[joint] ST-HGAT/relation head updated jointly with actor/critic")
    actor = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    encoder_target = copy.deepcopy(encoder).to(device) if args.target_encoder else None
    actor_target = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    actor_anchor = copy.deepcopy(actor).to(device)
    encoder_anchor = copy.deepcopy(encoder).to(device)
    critic1 = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic2 = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic1_target = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic2_target = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    modules = {
        "encoder": encoder,
        "actor": actor, "actor_target": actor_target,
        "actor_anchor": actor_anchor, "encoder_anchor": encoder_anchor,
        "critic1": critic1, "critic2": critic2,
        "critic1_target": critic1_target, "critic2_target": critic2_target,
    }
    if encoder_target is not None:
        modules["encoder_target"] = encoder_target
    for anchor_module in (actor_anchor, encoder_anchor, encoder_target):
        if anchor_module is None:
            continue
        anchor_module.eval()
        for parameter in anchor_module.parameters():
            parameter.requires_grad_(False)
    optimizers = {
        "encoder": torch.optim.Adam(encoder.auxiliary_parameters(), lr=args.lr_encoder),
        "actor": torch.optim.Adam(actor.parameters(), lr=args.lr_actor),
        "critics": torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), lr=args.lr_critic,
        ),
    }
    replay = RawTrajectoryBuffer(
        args.max_replay_episodes, args.max_elite_episodes,
    )
    start_episode = 0
    update_step = 0
    total_env_steps = 0
    if args.resume_from is not None:
        restored = load_full_training_checkpoint(
            args.resume_from, modules, optimizers, replay, rng, device,
        )
        start_episode = restored["start_episode"]
        update_step = restored["update_step"]
        total_env_steps = restored["total_env_steps"]
        state_kind = "full training state" if restored["full_state_restored"] else "legacy network stack"
        print(
            f"[resume] restored {state_kind} from {args.resume_from}; "
            f"next episode={start_episode} update_step={update_step} "
            f"replay_episodes={len(replay.episodes)}",
        )
    config = EndToEndUpdateConfig(
        gamma=args.gamma, tau=args.tau, freeze_encoder=bool(args.freeze_encoder),
        logit_penalty=bool(args.logit_action_penalty),
        actor_action_norm_weight=float(args.action_norm_weight),
        policy_delay=max(1, int(args.policy_delay)),
        policy_anchor_weight=max(0.0, float(args.policy_anchor_weight)),
    )
    allocator = StageCoalitionAllocator(
        d_cap=args.d_cap, device=str(device), lock_on_close3=True,
    )
    last_metrics = {}
    best_key = None
    started = time.time()

    for local_episode in range(args.episodes):
        episode = start_episode + local_episode
        env.reset(6, 2, scenario_seed=args.seed + episode)
        encoder.reset()
        allocator.reset()
        transitions = []
        prev_pred = prev_unc = prev_rel = None
        allocation = None
        done = False
        episode_reward = 0.0
        episode_reward_components = {}
        max_hold_counter = 0
        max_geometry_score = 0.0
        step = 0
        while not done and step < args.max_steps:
            before_assignment = env.snapshot()
            if allocation is None or step % 5 == 0:
                with torch.no_grad():
                    allocation_features = encoder.encode(
                        before_assignment, differentiable=False,
                    )
                allocation = allocator.allocate(
                    env.pursuer_positions(), env.pursuer_velocities(),
                    env.evader_positions(), env.evader_velocities(),
                    allocation_features["predictions"][:env.num_evaders],
                    allocation_features["uncertainty"][:env.num_evaders],
                    [(obstacle.position, obstacle.radius) for obstacle in env.obstacles],
                    before_assignment["task_mask"][:env.num_evaders],
                    env.pursuer_mask[:env.num_pursuers],
                    allocation_features["relation_cost"][:env.num_pursuers, :env.num_evaders],
                    step,
                    soft_task_plan=allocation_features["soft_assignment"][
                        :env.num_pursuers, :env.num_evaders
                    ],
                )
            env.apply_roles(
                allocation["core_assignment"], allocation["sup_assignment"],
                allocation["roles"], allocation["stages"],
            )
            snapshot = env.snapshot()
            features = encoder.encode(snapshot)
            prev_pred = features["predictions"][:env.num_evaders]
            prev_unc = features["uncertainty"][:env.num_evaders]
            prev_rel = features["relation_cost"]
            with torch.no_grad():
                action = actor(features["obs"]).cpu().numpy()
            exploration_noise = exploration_noise_at_episode(
                episode, args.exploration_decay_episodes,
                args.exploration_noise, args.final_exploration_noise,
            )
            action += rng.normal(0.0, exploration_noise, size=action.shape)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            reward, done, info = env.step(action, max_steps=args.max_steps)
            for name, value in info.get("reward_components", {}).items():
                episode_reward_components[name] = (
                    episode_reward_components.get(name, 0.0) + float(value)
                )
            max_hold_counter = max(
                max_hold_counter,
                int(np.max(env.hold_counters[: env.num_evaders], initial=0)),
            )
            max_geometry_score = max(
                max_geometry_score,
                float(np.max(
                    env.prev_geometry_scores[: env.num_evaders], initial=0.0,
                )),
            )
            transitions.append({
                "snapshot": snapshot,
                "action": action,
                "reward": scale_reward_for_critic(
                    reward, args.critic_reward_scale, args.critic_reward_clip,
                ),
                "done": done,
                "next_snapshot": env.snapshot(),
            })
            episode_reward += reward
            step += 1
        total_env_steps += step
        max_hold_fraction = min(
            1.0, max_hold_counter / max(1, env.hold_steps),
        )
        elite_priority = max(max_hold_fraction, max_geometry_score)
        replay.add_episode(
            transitions, successful=env.all_captured(), priority=elite_priority,
        )

        if local_episode + 1 >= args.warmup_episodes:
            for _ in range(args.updates_per_episode):
                # Sample one extra transition: encode_ordered_segment uses
                # segment[i+1]["snapshot"] as the bootstrap state for every i,
                # including the last, so the organization context is always the
                # post-role-assignment state of the next decision.
                segment = replay.sample_segment(
                    args.segment_length + 1, rng,
                    success_fraction=args.success_replay_fraction,
                    elite_fraction=args.elite_replay_fraction,
                )
                last_metrics = end_to_end_update(
                    modules, optimizers, segment, config,
                    update_index=update_step + 1,
                )
                update_step += 1
                if writer is not None:
                    for name, value in last_metrics.items():
                        writer.add_scalar(name, value, update_step)
        if writer is not None:
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/captured_targets", int(env.captured[:2].sum()), episode)
            # Episodes can end early, so the realised env-step budget can differ
            # between arms even at the same episode count; surface it directly.
            writer.add_scalar("episode/env_steps", step, episode)
            writer.add_scalar("episode/total_env_steps", total_env_steps, episode)
            writer.add_scalar("episode/terminated_early", int(step < args.max_steps), episode)
            writer.add_scalar("episode/max_hold_fraction", max_hold_fraction, episode)
            writer.add_scalar("episode/max_geometry_score", max_geometry_score, episode)
            writer.add_scalar("exploration/noise_std", exploration_noise, episode)
            for name, value in episode_reward_components.items():
                writer.add_scalar(f"reward_component/{name}", value, episode)
                writer.add_scalar(
                    f"reward_component_per_step/{name}",
                    value / max(1, step), episode,
                )

        if episode % max(1, args.eval_interval) == 0 or local_episode == args.episodes - 1:
            metrics = evaluate(
                env, encoder, allocator, actor, args.eval_episodes,
                args.max_steps, device, args.seed + 50000, scale=(6, 2),
            )
            capture = float(metrics["capture_rate"])
            # Capture rate is the headline comparison metric against the
            # encoder-frozen arm, so it must be on the event file, not only stdout.
            if writer is not None:
                writer.add_scalar("eval/capture_rate", capture, episode)
                writer.add_scalar("eval/six_v_two_capture_rate", capture, episode)
                writer.add_scalar("eval_scale/6v2_capture_rate", capture, episode)
                writer.add_scalar(
                    "eval/target_capture_rate",
                    float(metrics["target_capture_rate"]), episode,
                )
                writer.add_scalar(
                    "eval/partial_capture_rate",
                    float(metrics["partial_capture_rate"]), episode,
                )
                writer.add_scalar(
                    "eval/near_capture_rate",
                    float(metrics["near_capture_rate"]), episode,
                )
                writer.add_scalar(
                    "eval/mean_max_hold_fraction",
                    float(metrics["mean_max_hold_fraction"]), episode,
                )
                writer.add_scalar(
                    "eval/mean_reward", float(metrics["mean_reward"]), episode,
                )
                writer.add_scalar(
                    "eval/timeout_rate", float(metrics["timeout_rate"]), episode,
                )
                writer.add_scalar(
                    "eval/evader_goal_rate",
                    float(metrics["evader_goal_rate"]), episode,
                )
            write_eval_summary(args.output, episode, total_env_steps, metrics)
            selection_key = model_selection_key(metrics)
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                _save(
                    output / "best_model.pth", modules, args, episode, metrics,
                    reward_config=env.reward_config, optimizers=optimizers,
                    replay=replay, update_step=update_step,
                    total_env_steps=total_env_steps, rng=rng,
                )
            print(
                f"[e2e] ep={episode}/{start_episode + args.episodes - 1} reward={episode_reward:.2f} "
                f"eval_full={capture:.3f} updates={update_step} "
                f"grad_st={last_metrics.get('grad_norm/st_hgat', float('nan')):.4g} "
                f"grad_rel={last_metrics.get('grad_norm/relation_head', float('nan')):.4g} "
                f"grad_dem={last_metrics.get('grad_norm/demand_head', float('nan')):.4g} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
            _save(
                output / f"checkpoint_{episode:06d}.pth", modules, args,
                episode, metrics, reward_config=env.reward_config,
                optimizers=optimizers, replay=replay,
                update_step=update_step, total_env_steps=total_env_steps,
                rng=rng,
            )
    _save(
        output / "final_model.pth", modules, args,
        start_episode + args.episodes - 1, last_metrics,
        reward_config=env.reward_config, optimizers=optimizers,
        replay=replay, update_step=update_step,
        total_env_steps=total_env_steps, rng=rng,
    )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
