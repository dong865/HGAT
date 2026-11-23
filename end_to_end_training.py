"""Raw-sequence replay and gradient updates for experimental SDGCR training."""

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .stage_networks import StageFeatureEncoder
from .encoder_pretraining import (
    module_grad_norm,
    module_parameter_delta,
    snapshot_parameters,
)


@dataclass(frozen=True)
class EndToEndUpdateConfig:
    gamma: float = 0.999
    tau: float = 0.005
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    actor_action_norm_weight: float = 0.1
    encoder_grad_clip: float = 10.0
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 5.0
    policy_delay: int = 1
    policy_anchor_weight: float = 0.0
    demand_loss_weight: float = 0.1
    # Matched-pipeline control: keep the ST-HGAT/relation encoder frozen.  The
    # forward pass, segment sampling and the actor/critic/relation phases are
    # otherwise identical to the joint-update arm, so the only difference is
    # whether encoder parameters receive gradients.  This isolates the effect of
    # end-to-end joint updating from any difference in the training pipeline.
    freeze_encoder: bool = False
    logit_penalty: bool = False
    # Apply the action-magnitude penalty to the PRE-tanh logits instead of the
    # tanh output. At |a|=0.98 the tanh derivative is 0.04, so a post-tanh
    # penalty has ~25x less gradient exactly where it must act; the observed
    # action norm grew 0.975 -> 1.108 despite the penalty being active.


class RawTrajectoryBuffer:
    """Episode-preserving replay; sampled segments never cross a reset."""

    def __init__(self, max_episodes: int = 200, max_elite_episodes: int = 40):
        self.episodes = deque(maxlen=int(max_episodes))
        self.elite_episodes = deque(maxlen=max(0, int(max_elite_episodes)))

    def __len__(self) -> int:
        return sum(len(episode) for episode in self.episodes)

    def add_episode(
        self, transitions: Sequence[Dict], successful: bool = False,
        priority: float = 0.0,
    ) -> None:
        if not transitions:
            return
        episode = copy.deepcopy(list(transitions))
        for transition in episode:
            transition["_successful_episode"] = bool(successful)
            transition["_episode_priority"] = float(priority)
        self.episodes.append(episode)
        if self.elite_episodes.maxlen and (successful or float(priority) > 0.0):
            candidate = copy.deepcopy(episode)
            elites = list(self.elite_episodes)
            elites.append(candidate)
            elites.sort(
                key=lambda item: float(item[0].get("_episode_priority", 0.0)),
                reverse=True,
            )
            self.elite_episodes = deque(
                elites[: self.elite_episodes.maxlen],
                maxlen=self.elite_episodes.maxlen,
            )

    def state_dict(self) -> Dict:
        return {
            "max_episodes": int(self.episodes.maxlen),
            "episodes": copy.deepcopy(list(self.episodes)),
            "max_elite_episodes": int(self.elite_episodes.maxlen or 0),
            "elite_episodes": copy.deepcopy(list(self.elite_episodes)),
        }

    def load_state_dict(self, state: Dict) -> None:
        max_episodes = int(state["max_episodes"])
        if max_episodes <= 0:
            raise ValueError("replay max_episodes must be positive")
        self.episodes = deque(
            copy.deepcopy(list(state["episodes"])), maxlen=max_episodes,
        )
        max_elite = int(state.get("max_elite_episodes", 40))
        self.elite_episodes = deque(
            copy.deepcopy(list(state.get("elite_episodes", []))),
            maxlen=max(0, max_elite),
        )

    def sample_segment(
        self, length: int, rng: np.random.Generator, success_fraction: float = 0.0,
        elite_fraction: float = 0.0,
    ) -> List[Dict]:
        length = int(length)
        eligible = [episode for episode in self.episodes if len(episode) >= length]
        if not eligible:
            raise ValueError(f"no eligible episode for segment length {length}")
        successful = [episode for episode in eligible if episode[0].get("_successful_episode", False)]
        elite = [episode for episode in self.elite_episodes if len(episode) >= length]
        success_fraction = max(0.0, min(1.0, float(success_fraction)))
        elite_fraction = max(0.0, min(1.0 - success_fraction, float(elite_fraction)))
        draw = float(rng.random())
        if successful and draw < success_fraction:
            pool = successful
        elif elite and draw < success_fraction + elite_fraction:
            pool = elite
        else:
            pool = eligible
        episode = pool[int(rng.integers(0, len(pool)))]
        start = int(rng.integers(0, len(episode) - length + 1))
        return copy.deepcopy(episode[start:start + length])


def encode_ordered_segment(
    encoder: StageFeatureEncoder,
    segment: Sequence[Dict],
    differentiable: bool = True,
    plan_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Encode a contiguous trajectory segment.

    ``segment`` must contain ``plan_length + 1`` transitions.  Transition
    ``i`` contributes its (obs, action, reward, done); the bootstrap state is
    ALWAYS ``segment[i + 1]["snapshot"]`` — the post-role-assignment snapshot
    that the next actor decision is made from.  This removes the segment-
    boundary exception where a fresh reallocation could pair an action with
    stale organization context.  When ``plan_length`` is None it defaults to
    ``len(segment) - 1``; pass ``plan_length=len(segment)`` only for legacy
    callers that cannot provide the extra transition.
    """
    if not segment:
        raise ValueError("cannot encode an empty trajectory segment")
    if plan_length is None:
        plan_length = len(segment) - 1
    if plan_length <= 0:
        raise ValueError("plan_length must be positive")
    if len(segment) < plan_length + 1:
        raise ValueError(
            f"segment has {len(segment)} transitions but plan_length={plan_length} "
            "requires at least plan_length + 1"
        )
    first_step = int(segment[0]["snapshot"].get("episode_step", 0))
    encoder.reset(step=first_step)
    current = encoder.encode(
        segment[0]["snapshot"], differentiable=differentiable,
    )
    obs, next_obs, masks, next_masks = [], [], [], []
    demand_logits, demand_hat, soft_assignments, stages, task_masks = [], [], [], [], []
    plan_segment = segment[:plan_length]
    for i, transition in enumerate(plan_segment):
        following = encoder.encode(
            segment[i + 1]["snapshot"], differentiable=differentiable,
        )
        obs.append(current["obs"])
        masks.append(current["masks"])
        demand_logits.append(current["demand_logits"][: len(current["stages"])])
        demand_hat.append(current["demand_hat"][: len(current["stages"])])
        soft_assignments.append(
            current["soft_assignment"][:, : len(current["stages"])]
        )
        stages.append(torch.as_tensor(current["stages"], device=current["obs"].device))
        task_masks.append(torch.as_tensor(current["task_mask"], device=current["obs"].device))
        next_obs.append(following["obs"])
        next_masks.append(following["masks"])
        current = following
    device = obs[0].device
    return {
        "obs": torch.stack(obs),
        "next_obs": torch.stack(next_obs),
        "masks": torch.stack(masks),
        "next_masks": torch.stack(next_masks),
        "demand_logits": torch.stack(demand_logits),
        "demand_hat": torch.stack(demand_hat),
        "soft_assignment": torch.stack(soft_assignments),
        "stages": torch.stack(stages).long(),
        "task_masks": torch.stack(task_masks).bool(),
        "action": torch.as_tensor(
            np.stack([item["action"] for item in plan_segment]),
            dtype=torch.float32, device=device,
        ),
        "reward": torch.as_tensor(
            [item["reward"] for item in plan_segment],
            dtype=torch.float32, device=device,
        ),
        "done": torch.as_tensor(
            [item["done"] for item in plan_segment],
            dtype=torch.float32, device=device,
        ),
    }


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(),
        ):
            target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


def _clip(module: torch.nn.Module, maximum: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(module.parameters(), maximum))


def end_to_end_update(
    modules: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    segment: Sequence[Dict],
    config: EndToEndUpdateConfig,
    update_index: int = 1,
) -> Dict[str, float]:
    """Run critic, actor and relation phases with fresh sequence encodings."""
    encoder = modules["encoder"]
    encoder_target = modules.get("encoder_target")
    actor = modules["actor"]
    actor_anchor = modules.get("actor_anchor")
    encoder_anchor = modules.get("encoder_anchor")
    actor_target = modules["actor_target"]
    critic1 = modules["critic1"]
    critic2 = modules["critic2"]
    critic1_target = modules["critic1_target"]
    critic2_target = modules["critic2_target"]
    encoder.train()
    actor.train()
    critic1.train()
    critic2.train()

    tracked = {
        "st_hgat": encoder.st_hgat,
        "relation_head": encoder.relation_head,
        "demand_head": encoder.demand_head,
        "actor": actor,
        "critic1": critic1,
        "critic2": critic2,
    }
    before = {name: snapshot_parameters(module) for name, module in tracked.items()}
    gradient_norms = {name: 0.0 for name in tracked}

    # Critic phase: current observations retain the ST-HGAT graph, bootstrap
    # values are detached and therefore never update target modules.
    batch = encode_ordered_segment(encoder, segment, differentiable=True)
    with torch.no_grad():
        if encoder_target is None:
            target_next_obs = batch["next_obs"].detach()
        else:
            target_next_obs = encode_ordered_segment(
                encoder_target, segment, differentiable=False,
            )["next_obs"]
        next_actions = actor_target(target_next_obs)
        noise = torch.randn_like(next_actions) * max(0.0, config.target_policy_noise)
        noise = noise.clamp(-config.target_noise_clip, config.target_noise_clip)
        next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        next_q = torch.minimum(
            critic1_target(target_next_obs, next_actions, batch["next_masks"]),
            critic2_target(target_next_obs, next_actions, batch["next_masks"]),
        )
        target = batch["reward"] + config.gamma * (1.0 - batch["done"]) * next_q
    q1 = critic1(batch["obs"], batch["action"], batch["masks"])
    q2 = critic2(batch["obs"], batch["action"], batch["masks"])
    critic1_loss = torch.nn.functional.smooth_l1_loss(q1, target)
    critic2_loss = torch.nn.functional.smooth_l1_loss(q2, target)
    critic_loss = 0.5 * (critic1_loss + critic2_loss)
    optimizers["encoder"].zero_grad(set_to_none=True)
    optimizers["critics"].zero_grad(set_to_none=True)
    critic_loss.backward()
    gradient_norms["st_hgat"] = max(
        gradient_norms["st_hgat"], module_grad_norm(encoder.st_hgat),
    )
    gradient_norms["critic1"] = module_grad_norm(critic1)
    gradient_norms["critic2"] = module_grad_norm(critic2)
    if not config.freeze_encoder:
        _clip(encoder.st_hgat, config.encoder_grad_clip)
    _clip(critic1, config.critic_grad_clip)
    _clip(critic2, config.critic_grad_clip)
    optimizers["critics"].step()
    if not config.freeze_encoder:
        optimizers["encoder"].step()

    policy_delay = max(1, int(config.policy_delay))
    actor_applied = int(update_index) % policy_delay == 0

    # Actor phase: TD3 delays policy and target updates relative to the critic.
    # Re-encoding avoids reusing an autograd graph after a parameter step.
    actor_batch = encode_ordered_segment(encoder, segment, differentiable=True)
    for parameter in list(critic1.parameters()) + list(critic2.parameters()):
        parameter.requires_grad_(False)
    policy_logits = actor.forward_logits(actor_batch["obs"])
    policy_actions = torch.tanh(policy_logits)
    actor_q = critic1(actor_batch["obs"], policy_actions, actor_batch["masks"]).mean()
    if config.policy_anchor_weight > 0.0:
        if actor_anchor is None:
            raise KeyError("policy_anchor_weight requires modules['actor_anchor']")
        with torch.no_grad():
            if encoder_anchor is None:
                # Legacy behavior: constrain only the Actor head on the current
                # (possibly drifting) representation.
                anchor_obs = actor_batch["obs"].detach()
            else:
                # Functional policy anchor: evaluate the same raw trajectory
                # through the frozen reference encoder as well as the frozen
                # Actor, so representation drift cannot bypass the constraint.
                anchor_obs = encode_ordered_segment(
                    encoder_anchor, segment, differentiable=False,
                )["obs"]
            anchor_actions = actor_anchor(anchor_obs)
        policy_anchor_loss = torch.nn.functional.mse_loss(
            policy_actions, anchor_actions,
        )
    else:
        policy_anchor_loss = policy_actions.new_zeros(())
    if config.logit_penalty:
        # Pre-tanh (full gradient in the saturated region) AND norm-based.
        # Per-component thresholds were tried and failed: components below the
        # threshold are then unconstrained, so overall magnitude grew instead
        # (norm 1.14 -> 1.27). The limit is the logit norm that corresponds to a
        # 2-D action norm of 0.90, i.e. atanh(0.90)*sqrt(2) ~ 2.08.
        logit_limit = math.atanh(0.90) * math.sqrt(2.0)
        excess = torch.relu(torch.linalg.vector_norm(policy_logits, dim=-1) - logit_limit)
    else:
        excess = torch.relu(torch.linalg.vector_norm(policy_actions, dim=-1) - 1.0)
    actor_loss = (
        -actor_q
        + config.actor_action_norm_weight * excess.square().mean()
        + config.policy_anchor_weight * policy_anchor_loss
    )
    if actor_applied:
        optimizers["encoder"].zero_grad(set_to_none=True)
        optimizers["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        gradient_norms["st_hgat"] = max(
            gradient_norms["st_hgat"], module_grad_norm(encoder.st_hgat),
        )
        gradient_norms["relation_head"] = max(
            gradient_norms["relation_head"], module_grad_norm(encoder.relation_head),
        )
        gradient_norms["demand_head"] = max(
            gradient_norms["demand_head"], module_grad_norm(encoder.demand_head),
        )
        gradient_norms["actor"] = module_grad_norm(actor)
        if not config.freeze_encoder:
            _clip(encoder.st_hgat, config.encoder_grad_clip)
        _clip(actor, config.actor_grad_clip)
        optimizers["actor"].step()
        if not config.freeze_encoder:
            optimizers["encoder"].step()
    for parameter in list(critic1.parameters()) + list(critic2.parameters()):
        parameter.requires_grad_(True)

    # Relation phase supplies the legitimate learning path for relation_head;
    # the discrete allocator itself remains intentionally outside autograd.
    demand_batch = encode_ordered_segment(encoder, segment, differentiable=True)
    active_demands = demand_batch["task_masks"]
    demand_targets = demand_batch["stages"].clamp(0, 2)
    demand_loss = torch.nn.functional.cross_entropy(
        demand_batch["demand_logits"][active_demands],
        demand_targets[active_demands],
    )
    auxiliary = encoder.auxiliary_loss(
        encoder._step, batch_size=max(1, len(segment) - 1),
    )
    if auxiliary is None:
        raise RuntimeError("trajectory segment has no eligible future-relation labels")
    relation_loss = auxiliary["total"]
    auxiliary_loss = relation_loss + config.demand_loss_weight * demand_loss
    if not config.freeze_encoder:
        optimizers["encoder"].zero_grad(set_to_none=True)
        auxiliary_loss.backward()
        gradient_norms["st_hgat"] = max(
            gradient_norms["st_hgat"], module_grad_norm(encoder.st_hgat),
        )
        gradient_norms["relation_head"] = module_grad_norm(encoder.relation_head)
        gradient_norms["demand_head"] = max(
            gradient_norms["demand_head"], module_grad_norm(encoder.demand_head),
        )
        torch.nn.utils.clip_grad_norm_(
            encoder.auxiliary_parameters(), config.encoder_grad_clip,
        )
        optimizers["encoder"].step()
    # When frozen, the relation loss above is still reported for monitoring but
    # deliberately never optimised, so ST-HGAT and the relation head stay fixed.

    if actor_applied:
        if encoder_target is not None:
            _soft_update(encoder_target, encoder, config.tau)
        _soft_update(actor_target, actor, config.tau)
        _soft_update(critic1_target, critic1, config.tau)
        _soft_update(critic2_target, critic2, config.tau)

    scalars = [
        critic_loss, actor_loss, relation_loss, demand_loss, policy_anchor_loss,
        q1.mean(), q2.mean(),
    ]
    if not all(bool(torch.isfinite(value)) for value in scalars):
        raise FloatingPointError("non-finite end-to-end update")
    metrics = {
        "loss/critic": float(critic_loss.detach()),
        "loss/critic1": float(critic1_loss.detach()),
        "loss/critic2": float(critic2_loss.detach()),
        "loss/actor": float(actor_loss.detach()),
        "loss/relation": float(relation_loss.detach()),
        "loss/demand": float(demand_loss.detach()),
        "loss/policy_anchor": float(policy_anchor_loss.detach()),
        "q/q1_mean": float(q1.detach().mean()),
        "q/q2_mean": float(q2.detach().mean()),
        "action/saturation": float((policy_actions.detach().abs() >= 0.98).float().mean()),
        # Diagnostics for the frozen-vs-joint action-scale question: a frozen
        # encoder cannot rescale its own features, so if its observations are
        # larger the actor saturates more. These numbers make that visible
        # instead of assumed.
        "action/abs_mean": float(policy_actions.detach().abs().mean()),
        "action/logit_abs_mean": float(policy_logits.detach().abs().mean()),
        "action/penalty_excess": float(excess.detach().mean()),
        "action/norm_mean": float(
            torch.linalg.vector_norm(policy_actions.detach(), dim=-1).mean()
        ),
        "action/anchor_mse": float(policy_anchor_loss.detach()),
        "update/actor_applied": float(actor_applied),
        "obs/scale": float(actor_batch["obs"].detach().abs().mean()),
        "obs/std": float(actor_batch["obs"].detach().std()),
    }
    active_plan = actor_batch["soft_assignment"] * actor_batch["task_masks"].unsqueeze(1)
    probabilities = active_plan / active_plan.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    metrics["assignment/entropy"] = float(
        (-(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1)).detach().mean()
    )
    metrics["assignment/soft_capacity_error"] = float(
        (
            (active_plan.sum(dim=1) - actor_batch["demand_hat"])
            .abs()[actor_batch["task_masks"]]
            .detach().mean()
        )
    )
    for name, module in tracked.items():
        metrics[f"grad_norm/{name}"] = float(gradient_norms[name])
        metrics[f"parameter_delta/{name}"] = module_parameter_delta(before[name], module)
    return metrics
