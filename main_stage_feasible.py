"""Training entry for the paper method:
Stage-Feasible Dynamic Graph Coalition Reconfiguration (SDGCR).

Pipeline per step:
    allocator.allocate(geometry, predictions) -> (X, stages)     # stage-capacity-member
    env.apply_assignment(X, stages)                              # apply coalition
    encoder.encode(snapshot) -> obs                              # padded features
    shared actor -> planar velocity                              # cmdVelocityWorld semantics
    env.step(velocity) -> normalized team reward

Training:
* TD3 update with twin set-invariant graph critics and one shared actor
* optional delayed relation-cost supervision for the graph encoder
* fixed mixed-scale sampling: 50% 6V2, remaining mass across 4V2/5V2/6V3
"""

import argparse
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
torch.set_num_threads(4)  # CPU perf: avoid thread oversubscription on tiny nets
import torch.nn.functional as F

from .stage_env import StageCoalitionEnv, MODE_NAMES, StageRewardConfig
from .scenario_protocol import derive_scenario_seed
from .stage_coalitions import StageCoalitionAllocator
from .stage_networks import SharedActor, SetInvariantCritic, StageFeatureEncoder
from .encoder_pretraining import load_encoder_checkpoint
from .ablation_config import (
    AblationSpec,
    canonical_ablation_names,
    resolve_ablation,
)
from .training_protocol import (
    DEFAULT_D_CAP, DEFAULT_HOLD_STEPS, DEFAULT_SCALES, FixedScaleSampler,
    checkpoint_score, parse_scale_weights, stratified_seeds,
)

METHOD_NAME = "stage_feasible_dgcr"
CHECKPOINT_VERSION = 9  # v9: predictor-free deterministic CV lookahead
REWARD_VERSION = 2  # target-specific progress + causal contacts + enclosure potential


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------
class StageReplayBuffer:
    def __init__(self, max_size: int = 100000, obs_dim: int = 159, n_pursuers: int = 6):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.n_pursuers = n_pursuers
        self.data: deque = deque(maxlen=max_size)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        masks: np.ndarray,
        next_masks: np.ndarray,
        episode_id=None,
    ) -> None:
        self.data.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "done": float(done),
                "masks": np.asarray(masks, dtype=np.float32),
                "next_masks": np.asarray(next_masks, dtype=np.float32),
                "episode_id": episode_id,
                "success": False,
            }
        )

    def __len__(self) -> int:
        return len(self.data)

    def mark_episode_success(self, episode_id) -> None:
        for item in self.data:
            if item.get("episode_id") == episode_id:
                item["success"] = True

    def sample(self, batch_size: int, success_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        success_indices = [i for i, item in enumerate(self.data) if item.get("success", False)]
        requested = int(round(batch_size * max(0.0, min(1.0, success_fraction)))) if success_indices else 0
        selected = []
        if requested:
            selected.extend(np.random.choice(success_indices, size=requested, replace=requested > len(success_indices)).tolist())
        selected_set = set(selected)
        remaining = [i for i in range(len(self.data)) if i not in selected_set] or list(range(len(self.data)))
        count = batch_size - len(selected)
        selected.extend(np.random.choice(remaining, size=count, replace=count > len(remaining)).tolist())
        indices = np.asarray(selected, dtype=np.int64)
        items = [self.data[i] for i in indices]
        return {
            "obs": torch.as_tensor(np.stack([it["obs"] for it in items]), dtype=torch.float32),
            "action": torch.as_tensor(np.stack([it["action"] for it in items]), dtype=torch.float32),
            "reward": torch.as_tensor(np.array([it["reward"] for it in items]), dtype=torch.float32),
            "next_obs": torch.as_tensor(np.stack([it["next_obs"] for it in items]), dtype=torch.float32),
            "done": torch.as_tensor(np.array([it["done"] for it in items]), dtype=torch.float32),
            "masks": torch.as_tensor(np.stack([it["masks"] for it in items]), dtype=torch.float32),
            "next_masks": torch.as_tensor(np.stack([it["next_masks"] for it in items]), dtype=torch.float32),
            "success": torch.as_tensor(np.array([it.get("success", False) for it in items]), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def exploration_noise(
    episode: int, initial: float, decay: float, minimum: float,
) -> float:
    """Episode-based exploration schedule that survives complete trajectories."""
    return max(float(initial) * (float(decay) ** int(episode)), float(minimum))


def training_events_due(
    episode: int, total_episodes: int, eval_interval: int, save_interval: int,
) -> Dict[str, bool]:
    """Return independent evaluation and recovery-checkpoint decisions."""
    return {
        "evaluate": episode % max(1, eval_interval) == 0 or episode == total_episodes - 1,
        "save": episode > 0 and episode % max(1, save_interval) == 0,
    }


def should_log_step(total_steps: int, interval: int) -> bool:
    """Throttle high-volume step summaries while keeping episode/eval metrics."""
    return int(total_steps) % max(1, int(interval)) == 0


def log_allocation_feasibility(writer, allocation, step: int) -> None:
    """Write the decomposed soft stage-feasibility diagnostics to TensorBoard.

    ``allocation`` is the dict returned by ``StageCoalitionAllocator.allocate``;
    missing keys (older allocations) are logged as 0.0.
    """
    names = (
        "feasibility_cost",
        "capacity_deficit",
        "distance_deficit",
        "geometry_deficit",
        "hard_capture_feasible_ratio",
    )
    for name in names:
        writer.add_scalar(
            f"coalition/{name}",
            float((allocation or {}).get(name, 0.0)),
            int(step),
        )


def scale_reward_for_critic(reward: float, scale: float, clip: float) -> float:
    """Bound the critic target scale without changing reported environment reward."""
    bound = abs(float(clip))
    return float(np.clip(float(reward) * float(scale), -bound, bound))


def actor_update_due(critic_update: int, policy_delay: int) -> bool:
    """TD3-style delayed policy update schedule for the deterministic actor."""
    delay = max(1, int(policy_delay))
    return int(critic_update) % delay == 0


def clipped_double_q(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Conservative TD3 bootstrap value from two independent critics."""
    return torch.minimum(q1, q2)


def smooth_target_actions(
    actions: torch.Tensor,
    policy_noise: float,
    noise_clip: float,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply TD3 target-policy smoothing in normalized action space."""
    if noise is None:
        noise = torch.randn_like(actions) * max(0.0, float(policy_noise))
    clipped_noise = noise.clamp(-abs(float(noise_clip)), abs(float(noise_clip)))
    return (actions + clipped_noise).clamp(-1.0, 1.0)


def excess_action_norm_penalty(actions: torch.Tensor) -> torch.Tensor:
    """Penalize commands only where the environment would project them."""
    excess = torch.relu(torch.linalg.vector_norm(actions, dim=-1) - 1.0)
    return torch.mean(excess.square())


def allocation_event_metrics(allocation: Dict, updated: bool) -> Tuple[int, int]:
    """Count switch/invalid metrics only when a fresh plan was computed."""
    if not updated:
        return 0, 0
    switches = int(allocation["target_switch"]) + int(allocation["role_switch"])
    return switches, int(allocation["invalid"])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_eval_summary(
    output: str, episode: int, total_steps: int, metrics: Dict,
) -> Path:
    """Atomically expose the latest small evaluation payload for monitoring."""
    payload = {"episode": int(episode), "total_steps": int(total_steps)}
    payload.update(_json_safe(metrics))
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "latest_eval.json"
    temporary = output_dir / "latest_eval.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def checkpoint_ablation_metadata(spec: AblationSpec) -> Dict[str, object]:
    return {"ablation": spec.name, "architecture": spec.architecture()}


def validate_checkpoint_ablation(
    payload: Dict,
    requested: AblationSpec,
    allow_mismatch: bool = False,
) -> None:
    saved_name = payload.get("ablation")
    if saved_name is None:
        saved_name = payload.get("args", {}).get("ablation", "none")
    saved_name = "none" if saved_name in (None, "") else str(saved_name)
    saved_architecture = payload.get("architecture")
    mismatch = saved_name != requested.name
    if saved_architecture is not None:
        mismatch = mismatch or saved_architecture != requested.architecture()
    if mismatch and not allow_mismatch:
        raise ValueError(
            f"checkpoint ablation mismatch: saved={saved_name!r}, "
            f"requested={requested.name!r}",
        )

# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    encoder: StageFeatureEncoder,
    actor: SharedActor,
    actor_target: SharedActor,
    critic: SetInvariantCritic,
    critic_target: SetInvariantCritic,
    critic2: SetInvariantCritic,
    critic2_target: SetInvariantCritic,
    args: argparse.Namespace,
    metrics: Optional[Dict] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "reward_version": REWARD_VERSION,
        "method": METHOD_NAME,
        "args": vars(args),
        "obs_dim": encoder.obs_dim,
        "use_ring_slot_features": encoder.use_ring_slot_features,
        "use_stage_embedding": encoder.use_stage_embedding,
        "use_graph_embedding": encoder.use_graph_embedding,
        "ring_slot_normalizer": encoder.ring_slot_normalizer,
        "ring_slot_phase_hysteresis": encoder.ring_slot_phase_hysteresis,
        "hidden_dim": args.hidden_dim,
        "future_steps": args.future_steps,
        "seq_len": args.seq_len,
        "map_size": args.map_size,
        "num_obstacles": args.num_obstacles,
        "d_cap": args.d_cap,
        "hold_steps": args.hold_steps,
        "reward_config": reward_config_from_args(args).to_dict(),
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "critic2": critic2.state_dict(),
        "critic2_target": critic2_target.state_dict(),
        "st_hgat": encoder.st_hgat.state_dict(),
        "relation_head": encoder.relation_head.state_dict(),
        "metrics": metrics or {},
    }
    if encoder.fixed_concat_projector is not None:
        payload["fixed_concat_projector"] = encoder.fixed_concat_projector.state_dict()
    payload.update(checkpoint_ablation_metadata(
        resolve_ablation(getattr(args, "ablation", "none")),
    ))
    torch.save(payload, path)


def reward_config_from_args(args: argparse.Namespace) -> StageRewardConfig:
    """Build the exact shaping configuration persisted and used by training."""
    return StageRewardConfig(
        closure_weight=float(getattr(args, "closure_weight", 0.0)),
        closure_band=float(getattr(args, "closure_band", 3.0)),
        angular_closure_weight=float(getattr(args, "angular_closure_weight", 0.0)),
        angular_closure_band=float(getattr(args, "angular_closure_band", 2.5)),
        ring_weight=float(getattr(args, "ring_weight", 0.0)),
        ring_radius=float(getattr(args, "ring_radius", 1.5)),
        ring_lookahead=float(getattr(args, "ring_lookahead", 0.4)),
    )


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


def validate_initialization_args(args: argparse.Namespace) -> None:
    if getattr(args, "init_actor_from", None) and getattr(args, "init_encoder_from", None):
        raise ValueError("--init-encoder-from cannot be combined with --init-actor-from")


def initialize_encoder_from_checkpoint(
    path, encoder: StageFeatureEncoder, device: torch.device,
) -> Dict:
    """Load only the learned representation, leaving Actor/Critics untouched."""
    return load_encoder_checkpoint(path, encoder, device)


def configure_encoder_training(
    encoder: StageFeatureEncoder, train_encoder: bool,
) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad_(bool(train_encoder))
    encoder.train(bool(train_encoder))




def validate_predictor_free_checkpoint(payload: Dict) -> None:
    """Reject checkpoints whose Actor learned a different predictor input."""
    if "predictor" in payload:
        raise ValueError(
            "legacy predictor checkpoint cannot be loaded by the predictor-free "
            "SDGCR pipeline; retrain and use a v9 predictor-free checkpoint"
        )


def load_policy_stack(
    payload: Dict, actor: SharedActor, encoder: StageFeatureEncoder,
) -> None:
    """Restore the actor and the exact learned representation it consumes."""
    validate_predictor_free_checkpoint(payload)
    required = ("actor", "st_hgat", "relation_head")
    missing = [name for name in required if name not in payload]
    if missing:
        raise KeyError(f"policy-stack checkpoint missing keys: {missing}")
    actor.load_state_dict(payload["actor"])
    encoder.st_hgat.load_state_dict(payload["st_hgat"])
    encoder.relation_head.load_state_dict(payload["relation_head"])


def load_twin_critic_stack(
    payload: Dict,
    critic: SetInvariantCritic,
    critic_target: SetInvariantCritic,
    critic2: SetInvariantCritic,
    critic2_target: SetInvariantCritic,
) -> bool:
    """Load v5 twin critics, or expand a legacy single-critic checkpoint.

    A legacy critic initializes both estimators with a tiny symmetry-breaking
    perturbation, retaining its value scale without producing identical twins.
    """
    if "critic" not in payload:
        return False
    critic.load_state_dict(payload["critic"])
    critic_target.load_state_dict(payload.get("critic_target", payload["critic"]))
    if "critic2" in payload:
        critic2.load_state_dict(payload["critic2"])
        critic2_target.load_state_dict(payload.get("critic2_target", payload["critic2"]))
    else:
        critic2.load_state_dict(payload["critic"])
        critic2_target.load_state_dict(payload.get("critic_target", payload["critic"]))
        # An exact clone trained on the same minibatches would remain identical
        # forever and provide no clipped-double-Q protection.  Keep the legacy
        # value estimate while breaking that symmetry by a very small, matched
        # online/target perturbation.
        with torch.no_grad():
            for online_param, target_param in zip(
                critic2.parameters(), critic2_target.parameters(),
            ):
                perturbation = torch.randn_like(online_param) * 1e-3
                online_param.add_(perturbation)
                target_param.add_(perturbation)
    return True


@contextmanager
def evaluation_mode(*modules):
    """Temporarily evaluate modules, then restore each previous mode."""
    previous_modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in zip(modules, previous_modes):
            module.train(was_training)

# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _evaluate_impl(
    env: StageCoalitionEnv,
    encoder: StageFeatureEncoder,
    allocator: StageCoalitionAllocator,
    actor: SharedActor,
    episodes: int,
    max_steps: int,
    device: torch.device,
    seed: int = 0,
    scale: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    capture_rates = []
    capture_times = []
    target_capture_rates = []
    partial_capture_rates = []
    switch_counts = []
    invalid_counts = []
    rewards = []
    outcomes = {"all_captured": 0, "evader_goal_success": 0, "timeout": 0}
    stage_count = {name: 0 for name in MODE_NAMES.values()}
    for ep in range(episodes):
        episode_scale = scale if scale is not None else env.sample_scale()
        scenario_seed = derive_scenario_seed(seed, "validation", ep, episode_scale)
        env.reset(
            num_pursuers=episode_scale[0], num_evaders=episode_scale[1],
            scenario_seed=scenario_seed,
        )
        encoder.reset()
        allocator.reset()
        done = False
        step = 0
        switches = 0
        invalid = 0
        total_reward = 0.0
        capture_time = None
        prev_pred: Optional[np.ndarray] = None
        prev_unc: Optional[np.ndarray] = None
        prev_rel: Optional[np.ndarray] = None
        alloc = None
        while not done and step < max_steps:
            snap = env.snapshot()
            # allocation (uses predictions from the previous encode)
            allocation_updated = alloc is None or step % 5 == 0
            if allocation_updated:
                alloc = allocator.allocate(
                    env.pursuer_positions(), env.pursuer_velocities(),
                    env.evader_positions(), env.evader_velocities(),
                    prev_pred, prev_unc,
                    [(o.position, o.radius) for o in env.obstacles],
                    snap["task_mask"][: env.num_evaders],
                    env.pursuer_mask[: env.num_pursuers],
                    prev_rel, step,
                )
            env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
                            alloc["roles"], alloc["stages"])
            step_switches, step_invalid = allocation_event_metrics(
                alloc, allocation_updated,
            )
            switches += step_switches
            invalid += step_invalid
            snap = env.snapshot()
            features = encoder.encode(snap)
            prev_pred = features["predictions"][: env.num_evaders]
            prev_unc = features["uncertainty"][: env.num_evaders]
            prev_rel = features["relation_cost"]
            with torch.no_grad():
                actions = actor(features["obs"]).cpu().numpy()
            reward, done, info = env.step(actions, max_steps=max_steps)
            total_reward += reward
            step += 1
            for j in range(env.num_evaders):
                stage_count[MODE_NAMES.get(int(snap["stages"][j]), "?")] += 1
            if capture_time is None and env.all_captured():
                capture_time = step * env.time_step
        capture_rates.append(1.0 if env.all_captured() else 0.0)
        capture_times.append(capture_time if capture_time is not None else float("inf"))
        captured_count = int(np.count_nonzero(env.captured[: env.num_evaders]))
        target_capture_rates.append(captured_count / max(1, env.num_evaders))
        partial_capture_rates.append(
            1.0 if 0 < captured_count < env.num_evaders else 0.0
        )
        switch_counts.append(switches)
        invalid_counts.append(invalid)
        rewards.append(total_reward)
        outcomes[env.outcome] += 1
    finite_times = [t for t in capture_times if np.isfinite(t)]
    return {
        "capture_rate": float(np.mean(capture_rates)),
        "target_capture_rate": float(np.mean(target_capture_rates)),
        "partial_capture_rate": float(np.mean(partial_capture_rates)),
        "mean_capture_time": float(np.mean(finite_times)) if finite_times else float("inf"),
        "median_switches": float(np.median(switch_counts)),
        "invalid_rate": float(np.mean([1.0 if i > 0 else 0.0 for i in invalid_counts])),
        "mean_reward": float(np.mean(rewards)),
        "evader_goal_rate": outcomes["evader_goal_success"] / max(1, episodes),
        "timeout_rate": outcomes["timeout"] / max(1, episodes),
        "stage_histogram": dict(stage_count),
    }


def evaluate(
    env: StageCoalitionEnv,
    encoder: StageFeatureEncoder,
    allocator: StageCoalitionAllocator,
    actor: SharedActor,
    episodes: int,
    max_steps: int,
    device: torch.device,
    seed: int = 0,
    scale: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """Run deterministic evaluation without leaking eval mode into training.

    Evaluation always runs at protocol evader speed (scale 1.0) even when the
    training loop is inside an evader-speed curriculum.
    """
    prev_scale = getattr(env, "evader_speed_scale", 1.0)
    env.evader_speed_scale = 1.0
    try:
        with evaluation_mode(actor, encoder):
            return _evaluate_impl(
                env, encoder, allocator, actor, episodes, max_steps, device, seed, scale,
            )
    finally:
        env.evader_speed_scale = prev_scale
def resolve_eval_seed(train_seed: int, eval_seed: Optional[int] = None) -> int:
    """Base seed for the *fixed* validation scenario list.

    Every evaluation point (ep 0/500/1000/...) must reuse the SAME validation
    scenarios so checkpoint metrics are directly comparable; otherwise the
    reported capture rates are not comparable across training time.
    ``train_seed + EVAL_SEED_OFFSET`` keeps the validation space separate from
    the training seed space.
    """
    EVAL_SEED_OFFSET = 977
    return train_seed + EVAL_SEED_OFFSET if eval_seed is None else int(eval_seed)


def resolve_training_scenario_seed(
    train_seed: int,
    episode: int,
    scale: Tuple[int, int],
    curated_six_v_two_seeds: Sequence[int] = (),
) -> int:
    """Return a deterministic training scenario, optionally cycling a 6v2 pool.

    The curated pool is intentionally restricted to the paper's primary 6v2
    setting.  Auxiliary scales retain their normal seeded sampling so this
    option improves learning from known feasible cases without changing their
    distribution.
    """
    if tuple(scale) == (6, 2) and curated_six_v_two_seeds:
        return int(curated_six_v_two_seeds[int(episode) % len(curated_six_v_two_seeds)])
    return derive_scenario_seed(train_seed, "train", episode, scale)


def _weighted_rate(per_scale: Dict, field: str) -> float:
    """0.7 x 6v2 + 0.3 x macro-average of a per-scale metric field."""
    vals = [per_scale[f"{s[0]}v{s[1]}"][field] for s in DEFAULT_SCALES]
    return 0.7 * float(vals[0]) + 0.3 * float(np.mean(vals))


def model_selection_key(
    metrics: Dict, selection_scale: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Hierarchical best-model selection key (larger is better).

    Order: full capture rate (paper primary) > target capture rate >
    partial capture rate > higher mean reward.  Without a hierarchical key,
    best_model.pth only ever captured the ep0 -1->0 transition because every
    eval had full capture rate 0.
    """
    ps = metrics["per_scale"]
    if selection_scale is not None:
        selected = ps[selection_scale]
        return tuple(float(selected[field]) for field in (
            "capture_rate", "target_capture_rate", "partial_capture_rate", "mean_reward",
        ))
    return (
        _weighted_rate(ps, "capture_rate"),
        _weighted_rate(ps, "target_capture_rate"),
        _weighted_rate(ps, "partial_capture_rate"),
        _weighted_rate(ps, "mean_reward"),
    )


def evaluate_stratified(
    env, encoder, allocator, actor, episodes_per_scale, max_steps, device,
    seed=0, scales=DEFAULT_SCALES,
):
    rng_state = np.random.get_state()
    per_scale = {}
    try:
        scenarios = stratified_seeds(scales, episodes_per_scale, seed)
        for scale, seeds in scenarios.items():
            runs = [evaluate(env, encoder, allocator, actor, 1, max_steps, device, s, scale) for s in seeds]
            key = f"{scale[0]}v{scale[1]}"
            per_scale[key] = {
                "capture_rate": float(np.mean([r["capture_rate"] for r in runs])),
                "target_capture_rate": float(np.mean([r["target_capture_rate"] for r in runs])),
                "partial_capture_rate": float(np.mean([r["partial_capture_rate"] for r in runs])),
                "mean_reward": float(np.mean([r["mean_reward"] for r in runs])),
                "evader_goal_rate": float(np.mean([r["evader_goal_rate"] for r in runs])),
                "timeout_rate": float(np.mean([r["timeout_rate"] for r in runs])),
                "invalid_rate": float(np.mean([r["invalid_rate"] for r in runs])),
                "median_switches": float(np.median([r["median_switches"] for r in runs])),
                "mean_capture_time": float(np.mean([r["mean_capture_time"] for r in runs]))
                if all(np.isfinite(r["mean_capture_time"]) for r in runs) else float("inf"),
            }
    finally:
        np.random.set_state(rng_state)
    rates = {scale: per_scale[f"{scale[0]}v{scale[1]}"]["capture_rate"] for scale in scales}
    six_v_two_rate = rates.get((6, 2), float("nan"))
    return {
        "per_scale": per_scale,
        "six_v_two_capture_rate": six_v_two_rate,
        "macro_capture_rate": float(np.mean(list(rates.values()))),
        "checkpoint_score": (
            six_v_two_rate if tuple(scales) == ((6, 2),) else checkpoint_score(rates)
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--success-replay-fraction", type=float, default=0.25,
                        help="minimum successful-transition share per replay batch when available")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=None,
                        help="fixed base seed for the validation scenario list; "
                        "all eval points share the same scenarios (default: seed + 977)")
    parser.add_argument("--train-scenario-seeds", type=int, nargs="*", default=(),
                        help="optional curated scenario pool cycled only for 6v2 training; "
                             "other scales preserve their normal seeded sampling")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda (auto picks cuda when available)")
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--d-cap", type=float, default=DEFAULT_D_CAP)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument(
        "--evader-speed-curriculum-start", type=float, default=None,
        help="training-time evader max-speed start (m/s); linearly ramps to "
             "--evader-speed-curriculum-end over training; eval ALWAYS uses the "
             "protocol speed 0.35. None = protocol speed throughout.",
    )
    parser.add_argument("--evader-speed-curriculum-end", type=float, default=0.35)
    parser.add_argument(
        "--closure-weight", type=float, default=0.0,
        help="dense reward weight for contracting an assigned coalition's max "
             "distance inside the closure band (capture rule d_cap/hold unchanged); "
             "0.0 = off",
    )
    parser.add_argument("--closure-band", type=float, default=3.0,
                        help="meters inside which closure contraction is rewarded")
    parser.add_argument(
        "--angular-closure-weight", type=float, default=0.0,
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
    parser.add_argument(
        "--feasibility-capacity-weight", type=float, default=1.0,
        help="soft stage-feasibility cost: weight of the core-capacity deficit",
    )
    parser.add_argument(
        "--feasibility-distance-weight", type=float, default=1.0,
        help="soft stage-feasibility cost: weight of the d_cap distance deficit",
    )
    parser.add_argument(
        "--feasibility-geometry-weight", type=float, default=1.0,
        help="soft stage-feasibility cost: weight of the enclosure-geometry deficit",
    )
    parser.add_argument(
        "--feasibility-distance-band", type=float, default=1.0,
        help="normalizer (m) for the soft distance deficit",
    )
    parser.add_argument("--lock-radius", type=float, default=None,
                        help="override d_cap as the close3 lock trigger radius")
    parser.add_argument("--lock-preassign-on-capture", action="store_true",
                        help="after a locked target is captured, immediately dedicate the "
                             "nearest three pursuers to a remaining target")
    parser.add_argument(
        "--lock-on-close3", action="store_true",
        help="lock nearest-3 core while any evader has >=3 pursuers within d_cap "
             "(cross-layer dwell; capture/hard-fail/stall release)",
    )
    parser.add_argument(
        "--role-set",
        choices=("core_support_reserve", "core_reserve"),
        default="core_support_reserve",
        help="operational role set; core_reserve removes support without "
             "relabeling support members as core",
    )
    parser.add_argument(
        "--demand-mode",
        choices=("stage", "fixed", "direct"),
        default="stage",
        help="target-level core demand: stage=heuristic TRK/BLK/CAP, "
             "fixed=always CAP (records capacity deficit), "
             "direct=geometry-derived kappa mapped to TRK/BLK/CAP",
    )
    parser.add_argument(
        "--solver-mode",
        choices=("sinkhorn", "greedy", "hungarian"),
        default="sinkhorn",
        help="coalition assignment solver: sinkhorn=masked OT + projection, "
             "greedy=capacity-aware greedy, hungarian=linear sum assignment",
    )
    parser.add_argument("--flight-height", type=float, default=1.0,
                        help="cruise altitude of the 3-D state (z fluctuates via the height controller)")
    parser.add_argument("--v-max", type=float, default=0.35,
                        help="max pursuer speed m/s; must match SixVTwoConfig.max_pursuer_speed "
                             "(deployment) so training dynamics equal deployment dynamics")
    parser.add_argument("--evader-speed", type=float, default=0.35,
                        help="evader maximum speed in m/s for both training and periodic evaluation")
    parser.add_argument("--perception-radius", type=float, default=None,
                        help="partial observability: each pursuer perceives entities within "
                             "this radius (meters). None = full observability (paper protocol). "
                             "Recommended for this 8 m arena: 3.0 (covers the 1.5-3.0 m spawn "
                             "engagement envelope while keeping real occlusion).")
    parser.add_argument(
        "--ring-slot-interface", action="store_true",
        help="append a unique predicted ring-slot displacement and validity flag "
             "to each core/support actor observation; the actor still learns vx/vy",
    )
    parser.add_argument(
        "--ring-slot-normalizer", type=float, default=4.0,
        help="meters used to normalize the actor's ring-slot displacement",
    )
    parser.add_argument(
        "--ring-slot-phase-hysteresis", type=float, default=0.25,
        help="per-member squared-distance improvement required before changing "
             "the coalition's cyclic slot phase",
    )
    parser.add_argument(
        "--fresh-critic", action="store_true",
        help="with --init-actor-from / --init-encoder-from: keep the loaded policy "
             "but reinitialise the twin critics and their targets instead of "
             "restoring them; use when a previous run's value function diverged",
    )
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="[legacy no-op] encoder is frozen by default; see --train-encoder")
    parser.add_argument("--train-encoder", action="store_true",
                        help="keep training the ST-HGAT/relation encoder via auxiliary "
                             "losses during RL. Default (off) FREEZES the encoder after "
                             "warm-start so replay observations stay semantically consistent "
                             "(aux-trained encoder + encoded replay is the least stable combo)")
    parser.add_argument("--min-pursuers", type=int, default=4)
    parser.add_argument("--max-pursuers", type=int, default=6)
    parser.add_argument("--min-evaders", type=int, default=1)
    parser.add_argument("--max-evaders", type=int, default=3)
    parser.add_argument("--scale-weights", type=parse_scale_weights,
                        default=parse_scale_weights("0.7,0.1,0.1,0.1"),
                        help="fixed weights for 6v2,4v2,5v2,6v3")
    parser.add_argument(
        "--ablation",
        choices=("none",) + canonical_ablation_names(),
        default="none",
        help="one canonical formal ablation; combinations are intentionally rejected",
    )
    parser.add_argument(
        "--supplementary-ablation",
        default="",
        help="comma-separated legacy role variants: merge_core_support,reserve_hover,"
             "no_role_embedding,no_promotion,no_stage_embedding,no_graph_embedding",
    )
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--lr-relation", type=float, default=1e-3)
    parser.add_argument("--critic-reward-scale", type=float, default=0.1)
    parser.add_argument("--critic-reward-clip", type=float, default=10.0)
    parser.add_argument("--policy-delay", type=int, default=2,
                        help="update actor and target networks every N critic updates")
    parser.add_argument("--target-policy-noise", type=float, default=0.2,
                        help="TD3 Gaussian smoothing std in normalized action space")
    parser.add_argument("--target-noise-clip", type=float, default=0.5,
                        help="absolute clip applied to TD3 target-action noise")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="discount per 50 Hz step; 0.999 keeps ~70% terminal "
                             "credit over 350 steps (7 s) of approach, vs 0.995 "
                             "which leaves only ~5.6% at the mean episode length")
    parser.add_argument("--actor-action-norm-weight", type=float, default=0.1,
                        help="penalize only policy action norms above the feasible unit disk")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.20)
    parser.add_argument("--min-exploration-noise", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--actor-grad-clip", type=float, default=1.0)
    parser.add_argument("--critic-grad-clip", type=float, default=5.0)
    parser.add_argument("--noise-decay", type=float, default=0.9995)
    parser.add_argument("--train-interval", type=int, default=2)
    parser.add_argument("--step-log-interval", type=int, default=100,
                        help="write high-volume TensorBoard step metrics every N environment steps")
    parser.add_argument("--assign-interval", type=int, default=5,
                        help="recompute the coalition every K steps (paper: low-frequency reconfiguration)")
    parser.add_argument("--relation-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--validation-scales", choices=("all", "six_v_two"), default="all",
        help="periodic validation/model selection scales; use six_v_two when other "
             "team sizes are reserved exclusively for final generalization tests",
    )
    parser.add_argument("--init-actor-from", default=None,
                        help="pretrained BC actor checkpoint to warm-start the policy")
    parser.add_argument(
        "--init-encoder-from", default=None,
        help="self-supervised encoder-only checkpoint; loads ST-HGAT and relation "
             "head while leaving Actor/Critics fresh (cannot combine with "
             "--init-actor-from)",
    )
    parser.add_argument("--output", default="tmp/stage_feasible")
    parser.add_argument("--tensorboard-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    validate_initialization_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    ablation_spec = resolve_ablation(args.ablation)
    supplementary = set(
        x.strip() for x in args.supplementary_ablation.split(",") if x.strip()
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    run_name = args.run_name or f"SDGCR_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_dir, run_name))
        except Exception as exc:  # pragma: no cover
            print(f"[warning] TensorBoard unavailable: {exc}")

    env = StageCoalitionEnv(
        length=args.map_size,
        num_obstacle=args.num_obstacles,
        max_pursuers=args.max_pursuers,
        max_evaders=args.max_evaders,
        d_cap=args.d_cap,
        hold_steps=args.hold_steps,
        v_max=args.v_max,
        flight_height=args.flight_height,
        ablation_reserve_hover=("reserve_hover" in supplementary),
        perception_radius=args.perception_radius,
        reward_config=reward_config_from_args(args),
    )
    env.v_max_e = float(args.evader_speed)
    encoder = StageFeatureEncoder(
        max_pursuers=args.max_pursuers,
        max_evaders=args.max_evaders,
        num_obstacles=args.num_obstacles,
        map_size=args.map_size,
        hidden_dim=args.hidden_dim,
        future_steps=args.future_steps,
        seq_len=args.seq_len,
        device=device,
        use_role_embedding=("no_role_embedding" not in supplementary),
        use_stage_embedding=("no_stage_embedding" not in supplementary),
        use_graph_embedding=("no_graph_embedding" not in supplementary),
        perception_radius=args.perception_radius,
        node_encoder=ablation_spec.node_encoder,
        coalition_pool=ablation_spec.coalition_pool,
        use_ring_slot_features=args.ring_slot_interface,
        ring_slot_normalizer=args.ring_slot_normalizer,
        ring_slot_phase_hysteresis=args.ring_slot_phase_hysteresis,
    ).to(device)
    actor = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    actor_target = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    critic = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic_target = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic2 = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    critic2_target = SetInvariantCritic(encoder.obs_dim, hidden_dim=64).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    if args.init_actor_from:
        bc = load_checkpoint(args.init_actor_from, device)
        validate_checkpoint_ablation(bc, ablation_spec)
        if bc.get("obs_dim", encoder.obs_dim) != encoder.obs_dim:
            raise ValueError("pretrained policy-stack obs_dim mismatch")
        load_policy_stack(bc, actor, encoder)
        actor_target.load_state_dict(bc.get("actor_target", bc["actor"]))
        if getattr(args, "fresh_critic", False):
            # Keep the policy but throw the value function away. A warm-started
            # critic carries whatever bias its earlier run accumulated, and with a
            # slow Polyak target that bias decays at rate tau*(1-gamma) per update
            # -- about 5e-5 with gamma=0.99 and tau=0.005, i.e. tens of thousands of
            # updates per e-folding. Relearning the critic from a fresh replay is
            # far cheaper than unwinding it.
            critic_message = "FRESH twin critics (loaded critic deliberately discarded)"
        else:
            critics_loaded = load_twin_critic_stack(
                bc, critic, critic_target, critic2, critic2_target,
            )
            critic_message = "twin/legacy critic stack" if critics_loaded else "fresh twin critics"
        print(
            f"[init] actor + encoder + {critic_message} warm-started from "
            f"{args.init_actor_from}"
        )
    elif args.init_encoder_from:
        initialize_encoder_from_checkpoint(args.init_encoder_from, encoder, device)
        print(
            f"[init] ST-HGAT + relation encoder warm-started from "
            f"{args.init_encoder_from}; actor and twin critics are fresh"
        )

    allocator = StageCoalitionAllocator(
        d_cap=args.d_cap,
        device=args.device,
        role_set=args.role_set,
        demand_mode=args.demand_mode,
        merge_core_support=("merge_core_support" in supplementary),
        use_stage_feasibility=ablation_spec.use_stage_feasibility,
        allow_promotion=("no_promotion" not in supplementary),
        coalition_mode=ablation_spec.coalition_mode,
        solver_mode=args.solver_mode,
        lock_on_close3=bool(getattr(args, "lock_on_close3", False)),
        lock_radius=getattr(args, "lock_radius", None),
        lock_preassign_on_capture=bool(getattr(args, "lock_preassign_on_capture", False)),
        feasibility_capacity_weight=args.feasibility_capacity_weight,
        feasibility_distance_weight=args.feasibility_distance_weight,
        feasibility_geometry_weight=args.feasibility_geometry_weight,
        feasibility_distance_band=args.feasibility_distance_band,
    )
    scale_sampler = FixedScaleSampler(DEFAULT_SCALES, args.scale_weights, args.seed + 101)
    buffer = StageReplayBuffer(args.replay_size, encoder.obs_dim, args.max_pursuers)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_parameters = list(critic.parameters()) + list(critic2.parameters())
    opt_critic = torch.optim.Adam(critic_parameters, lr=args.lr_critic)
    opt_relation = torch.optim.Adam(encoder.auxiliary_parameters(), lr=args.lr_relation)
    configure_encoder_training(encoder, args.train_encoder)
    if not args.train_encoder:
        # DEFAULT: freeze after any warm-start load (see --train-encoder): replay obs now come from a FIXED
        # encoder, so buffered features are semantically consistent across the
        # whole run (fixes the stale-embedding drift of a moving encoder)
        print("[freeze] ST-HGAT/relation encoder frozen; only actor/critic learn")
    actor.train()
    critic.train()
    critic2.train()

    total_steps = 0
    critic_updates = 0
    last_actor_loss = float("nan")
    last_actor_grad_norm = 0.0
    last_actor_norm_penalty = 0.0
    best_capture_rate = -1.0
    best_selection_key: Optional[Tuple[float, float, float, float]] = None
    metric_history = []

    for episode in range(args.episodes):
        if args.evader_speed_curriculum_start is not None:
            frac = min(1.0, episode / max(1, args.episodes - 1))
            speed = (
                args.evader_speed_curriculum_start
                + (args.evader_speed_curriculum_end - args.evader_speed_curriculum_start) * frac
            )
            # Scale relative to this run's configured evaluation speed;
            # evaluate() forces scale 1.0 without overwriting env.v_max_e.
            env.evader_speed_scale = float(np.clip(speed / args.evader_speed, 0.1, 1.0))
        else:
            env.evader_speed_scale = 1.0
        n_p, n_e = scale_sampler.sample()
        scenario_seed = resolve_training_scenario_seed(
            args.seed, episode, (n_p, n_e), args.train_scenario_seeds,
        )
        env.reset(num_pursuers=n_p, num_evaders=n_e, scenario_seed=scenario_seed)
        encoder.reset(step=0)
        allocator.reset()

        episode_reward = 0.0
        episode_reward_components: Dict[str, float] = {}
        episode_switches = 0
        episode_invalid = 0
        done = False
        step = 0
        prev_pred: Optional[np.ndarray] = None
        prev_unc: Optional[np.ndarray] = None
        prev_rel: Optional[np.ndarray] = None
        alloc = None

        while not done and step < args.max_steps:
            snap = env.snapshot()
            allocation_updated = (
                alloc is None or total_steps % args.assign_interval == 0
            )
            if allocation_updated:
                alloc = allocator.allocate(
                    env.pursuer_positions(), env.pursuer_velocities(),
                    env.evader_positions(), env.evader_velocities(),
                    prev_pred, prev_unc,
                    [(o.position, o.radius) for o in env.obstacles],
                    snap["task_mask"][: env.num_evaders],
                    env.pursuer_mask[: env.num_pursuers],
                    prev_rel, total_steps,
                )
            env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
                            alloc["roles"], alloc["stages"])
            step_switches, step_invalid = allocation_event_metrics(
                alloc, allocation_updated,
            )
            episode_switches += step_switches
            episode_invalid += step_invalid

            snap = env.snapshot()
            features = encoder.encode(snap)
            prev_pred = features["predictions"][: env.num_evaders]
            prev_unc = features["uncertainty"][: env.num_evaders]
            prev_rel = features["relation_cost"]

            noise = exploration_noise(episode, args.exploration_noise, args.noise_decay, args.min_exploration_noise)
            with torch.no_grad():
                actions = actor(features["obs"]).cpu().numpy()
            actions += np.random.randn(*actions.shape).astype(np.float32) * noise
            actions = np.clip(actions, -1.0, 1.0)

            reward, done, info = env.step(actions, max_steps=args.max_steps)

            snap_next = env.snapshot()
            features_next = encoder.encode(snap_next)
            buffer.store(
                features["obs"].cpu().numpy(),
                actions,
                scale_reward_for_critic(
                    reward, args.critic_reward_scale, args.critic_reward_clip,
                ),
                features_next["obs"].cpu().numpy(),
                done,
                features["masks"].cpu().numpy(),
                features_next["masks"].cpu().numpy(),
                episode_id=episode,
            )
            episode_reward += reward
            for name, value in info.get("reward_components", {}).items():
                episode_reward_components[name] = (
                    episode_reward_components.get(name, 0.0) + float(value)
                )
            total_steps += 1
            step += 1

            if total_steps % args.train_interval == 0 and len(buffer) >= max(args.batch_size, args.warmup_steps):
                batch = buffer.sample(args.batch_size, args.success_replay_fraction)
                batch = {k: v.to(device) for k, v in batch.items()}

                # ---- critic
                with torch.no_grad():
                    next_actions = smooth_target_actions(
                        actor_target(batch["next_obs"]),
                        args.target_policy_noise,
                        args.target_noise_clip,
                    )
                    q1_next = critic_target(
                        batch["next_obs"], next_actions, batch["next_masks"],
                    )
                    q2_next = critic2_target(
                        batch["next_obs"], next_actions, batch["next_masks"],
                    )
                    q_next = clipped_double_q(q1_next, q2_next)
                    y = batch["reward"] + args.gamma * (1.0 - batch["done"]) * q_next
                q1_pred = critic(batch["obs"], batch["action"], batch["masks"])
                q2_pred = critic2(batch["obs"], batch["action"], batch["masks"])
                critic1_loss = F.smooth_l1_loss(q1_pred, y)
                critic2_loss = F.smooth_l1_loss(q2_pred, y)
                # Keep the legacy aggregate-loss scale comparable in TensorBoard.
                critic_loss = 0.5 * (critic1_loss + critic2_loss)
                opt_critic.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, args.critic_grad_clip,
                )
                opt_critic.step()

                critic_updates += 1
                actor_loss = None
                actor_grad_norm = 0.0
                actor_updated = actor_update_due(critic_updates, args.policy_delay)
                if actor_updated:
                    # ---- delayed actor
                    policy_actions = actor(batch["obs"])
                    q_policy = critic(
                        batch["obs"], policy_actions, batch["masks"],
                    ).mean()
                    actor_norm_penalty = excess_action_norm_penalty(policy_actions)
                    actor_loss = (
                        -q_policy
                        + args.actor_action_norm_weight * actor_norm_penalty
                    )
                    opt_actor.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), args.actor_grad_clip,
                    )
                    opt_actor.step()
                    last_actor_loss = float(actor_loss.item())
                    last_actor_grad_norm = float(actor_grad_norm)
                    last_actor_norm_penalty = float(actor_norm_penalty.item())

                    # ---- delayed soft targets
                    for target, source in (
                        (actor_target, actor),
                        (critic_target, critic),
                        (critic2_target, critic2),
                    ):
                        for tp, sp in zip(target.parameters(), source.parameters()):
                            tp.data.copy_(
                                args.tau * sp.data + (1.0 - args.tau) * tp.data
                            )

                if writer is not None and should_log_step(total_steps, args.step_log_interval):
                    writer.add_scalar("train/critic_loss", critic_loss.item(), total_steps)
                    if np.isfinite(last_actor_loss):
                        writer.add_scalar("train/actor_loss", last_actor_loss, total_steps)
                        writer.add_scalar("train/actor_grad_norm", last_actor_grad_norm, total_steps)
                        writer.add_scalar(
                            "train/actor_norm_penalty", last_actor_norm_penalty, total_steps,
                        )
                    td_abs = (clipped_double_q(q1_pred, q2_pred) - y).abs()
                    writer.add_scalar("train/td_abs_mean", td_abs.mean().item(), total_steps)
                    writer.add_scalar("train/td_abs_p95", torch.quantile(td_abs, 0.95).item(), total_steps)
                    writer.add_scalar("train/q_pred_mean", q1_pred.mean().item(), total_steps)
                    writer.add_scalar("train/q1_pred_mean", q1_pred.mean().item(), total_steps)
                    writer.add_scalar("train/q2_pred_mean", q2_pred.mean().item(), total_steps)
                    writer.add_scalar(
                        "train/q_min_pred_mean",
                        clipped_double_q(q1_pred, q2_pred).mean().item(),
                        total_steps,
                    )
                    writer.add_scalar(
                        "train/q_disagreement_mean",
                        (q1_pred - q2_pred).abs().mean().item(),
                        total_steps,
                    )
                    writer.add_scalar("train/critic1_loss", critic1_loss.item(), total_steps)
                    writer.add_scalar("train/critic2_loss", critic2_loss.item(), total_steps)
                    writer.add_scalar("train/q_target_mean", y.mean().item(), total_steps)
                    writer.add_scalar("train/replay_reward_mean", batch["reward"].mean().item(), total_steps)
                    writer.add_scalar(
                        "train/actor_updates",
                        critic_updates // max(1, args.policy_delay),
                        total_steps,
                    )
                    writer.add_scalar("train/critic_grad_norm", float(critic_grad_norm), total_steps)
                    writer.add_scalar(
                        "train/critic_grad_clipped",
                        float(critic_grad_norm > args.critic_grad_clip),
                        total_steps,
                    )
                    writer.add_scalar("train/exploration_noise", noise, total_steps)

                # ---- delayed graph/relation auxiliary supervision
                # (skipped entirely when the encoder is frozen: obs semantics fixed)
                if args.train_encoder and total_steps % args.relation_interval == 0:
                    aux_losses = encoder.auxiliary_loss(encoder._step, batch_size=64)
                    if aux_losses is not None:
                        opt_relation.zero_grad()
                        aux_losses["total"].backward()
                        aux_grad_norm = torch.nn.utils.clip_grad_norm_(
                            encoder.auxiliary_parameters(), 10.0,
                        )
                        opt_relation.step()
                        if writer is not None and should_log_step(total_steps, args.step_log_interval):
                            writer.add_scalar("train/aux_total", aux_losses["total"].item(), total_steps)
                            writer.add_scalar("train/relation_loss", aux_losses["relation_loss"].item(), total_steps)
                            writer.add_scalar("train/aux_grad_norm", float(aux_grad_norm), total_steps)

            if writer is not None and should_log_step(total_steps, args.step_log_interval):
                # latest allocation diagnostics (alloc is refreshed at most
                # assign_interval steps before this log point)
                log_allocation_feasibility(writer, alloc, total_steps)
                writer.add_scalar("train/team_reward", reward, total_steps)
                writer.add_scalar("train/switch_count", step_switches, total_steps)
                writer.add_scalar(
                    "train/action_saturation",
                    float(np.mean(np.abs(actions[:env.num_pursuers]) >= 0.98)),
                    total_steps,
                )
                writer.add_scalar(
                    "train/action_norm",
                    float(np.linalg.norm(actions[:env.num_pursuers], axis=1).mean()),
                    total_steps,
                )
                for name, value in info.get("reward_components", {}).items():
                    writer.add_scalar(
                        f"reward_component/{name}", float(value), total_steps,
                    )
                writer.add_scalar(
                    "train/target_switch",
                    int(alloc["target_switch"]) if allocation_updated else 0,
                    total_steps,
                )
                writer.add_scalar(
                    "train/role_switch",
                    int(alloc["role_switch"]) if allocation_updated else 0,
                    total_steps,
                )
                writer.add_scalar("train/invalid", step_invalid, total_steps)
                writer.add_scalar("train/scale_p", env.num_pursuers, total_steps)
                writer.add_scalar("train/scale_e", env.num_evaders, total_steps)

        if env.all_captured():
            buffer.mark_episode_success(episode)
        metrics = {
            "reward": episode_reward,
            "switches": episode_switches,
            "invalid": episode_invalid,
            "captured_all": 1.0 if env.all_captured() else 0.0,
            "steps": step,
            "scale": f"{env.num_pursuers}v{env.num_evaders}",
            "outcome": env.outcome,
            "scenario_seed": env.scenario_seed,
            "reward_components": dict(episode_reward_components),
        }
        metric_history.append(metrics)

        if writer is not None:
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/switches", episode_switches, episode)
            writer.add_scalar("episode/captured_all", metrics["captured_all"], episode)
            # 部分捕获计数：captured_all 是布尔，把"抓到 1 个"和"一个没抓到"抹成同一个值。
            # 逐目标计数（0..num_evaders）保留部分捕获信息，且与 main_stage_end_to_end 的
            # episode/captured_targets 同口径，可跨线比较"平均捕获目标数/回合"。
            writer.add_scalar(
                "episode/captured_targets",
                int(np.count_nonzero(env.captured[: env.num_evaders])),
                episode,
            )
            writer.add_scalar("episode/invalid", episode_invalid, episode)
            writer.add_scalar("episode/evader_goal_success", float(env.outcome == "evader_goal_success"), episode)
            writer.add_scalar("episode/timeout", float(env.outcome == "timeout"), episode)
            for name, value in episode_reward_components.items():
                writer.add_scalar(
                    f"episode_reward_component/{name}", float(value), episode,
                )

        events_due = training_events_due(episode, args.episodes, args.eval_interval, args.save_interval)
        if events_due["evaluate"]:
            eval_metrics = evaluate_stratified(
                env, encoder, allocator, actor,
                args.eval_episodes, args.max_steps, device,
                seed=resolve_eval_seed(args.seed, args.eval_seed),
                scales=((6, 2),) if args.validation_scales == "six_v_two" else DEFAULT_SCALES,
            )
            write_eval_summary(args.output, episode, total_steps, eval_metrics)
            if writer is not None:
                writer.add_scalar("eval/capture_rate", eval_metrics["checkpoint_score"], episode)
                writer.add_scalar("eval/six_v_two_capture_rate", eval_metrics["six_v_two_capture_rate"], episode)
                writer.add_scalar("eval/macro_capture_rate", eval_metrics["macro_capture_rate"], episode)
                for scale_name, scale_metrics in eval_metrics["per_scale"].items():
                    writer.add_scalar(f"eval_scale/{scale_name}_capture_rate", scale_metrics["capture_rate"], episode)
            print(
                f"[ep {episode}] scale={n_p}v{n_e} outcome={metrics['outcome']} "
                f"reward={episode_reward:.2f} eval_cap={eval_metrics['checkpoint_score']:.3f} "
                f"sw={episode_switches} inv={episode_invalid}"
            )
            if eval_metrics["checkpoint_score"] > best_capture_rate:
                best_capture_rate = eval_metrics["checkpoint_score"]
            # hierarchical best-model selection: full > target > partial > reward
            sel_key = model_selection_key(
                eval_metrics,
                selection_scale="6v2" if args.validation_scales == "six_v_two" else None,
            )
            if best_selection_key is None or sel_key > best_selection_key:
                best_selection_key = sel_key
                save_checkpoint(
                    os.path.join(args.output, "best_model.pth"), encoder, actor,
                    actor_target, critic, critic_target, critic2, critic2_target,
                    args, eval_metrics,
                )
                print(
                    f"  -> saved best (full={sel_key[0]:.3f} target={sel_key[1]:.3f} "
                    f"partial={sel_key[2]:.3f})"
                )

        if events_due["save"]:
            save_checkpoint(
                os.path.join(args.output, f"checkpoint_{episode:06d}.pth"), encoder,
                actor, actor_target, critic, critic_target, critic2, critic2_target,
                args,
            )

    save_checkpoint(
        os.path.join(args.output, "final_model.pth"), encoder, actor,
        actor_target, critic, critic_target, critic2, critic2_target, args,
        {"best_capture_rate": best_capture_rate},
    )
    Path(os.path.join(args.output, "metrics.json")).write_text(
        json.dumps(metric_history[-500:], indent=2), encoding="utf-8"
    )
    print(f"[done] best eval capture_rate={best_capture_rate:.3f} -> {args.output}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
