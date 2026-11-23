"""continuous_membership_v2 训练模块（阶段 D）。

- 可微 V2 观测构建（soft 上下文用于训练，同一聚合函数）；
- raw-transition 回放（L 条 transition + L+1 对齐决策状态）；
- 三阶段顺序更新：Critic（soft 在线 / 可配 hard+no_grad 目标）
  -> 延迟 Actor（Q1 + logit 超阈值惩罚）-> 辅助（仅关系 Smooth L1）。
  需求头只经 P 路径获得策略/价值梯度；无阶段 CE、无人数一致性伪监督。
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .continuous_membership import (
    ContinuousDemandHead,
    build_membership_context,
    build_soft_plan,
)
from .stage_networks import StageFeatureEncoder, unique_ring_slot_features
from .v2_controller import (
    ALLOCATION_SOURCE,
    ARCHITECTURE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    v2_obs_dim,
)


# ---------------------------------------------------------------------------
# 可微 V2 观测（训练用；与 v2_controller._build_v2_obs 语义一致：
# 同一聚合函数 build_membership_context，同一观测 schema）
# ---------------------------------------------------------------------------


def build_v2_obs_torch(
    enc_out: Dict,
    K: torch.Tensor,                       # [Np_max, Nt_max]：P（soft）或 X（hard）
    snap: Dict,
    max_pursuers: int,
    max_evaders: int,
    epsilon_c: float = 1e-6,
    stop_plan_gradient: bool = False,
) -> torch.Tensor:
    """构造 V2 观测张量，图块保持 autograd 连接。

    K 为任务关联矩阵（真实任务列）。stop_plan_gradient=True 时在上下文
    消费前 detach K（消融臂：前向不变、P 梯度断开）。
    """
    if stop_plan_gradient:
        K = K.detach()

    device = enc_out["pursuer_embeddings"].device
    H = enc_out["pursuer_embeddings"].shape[1]
    n_p = int(snap["num_pursuers"])
    n_ev = int(snap["num_evaders"])
    positions = np.asarray(snap["positions"])
    velocities = np.asarray(snap["velocities"])
    length = float(snap["length"])
    future_steps = enc_out["predictions"].shape[1]

    pursuer_emb = enc_out["pursuer_embeddings"]              # [Np_max, H]
    task_emb = enc_out["task_embeddings"]                    # [Nt_max, H]
    t_ctx, c_ctx = build_membership_context(
        K.to(device), pursuer_emb, task_emb, epsilon_c=epsilon_c)

    task_mask = np.asarray(snap["task_mask"])
    pursuer_mask = np.asarray(snap["pursuer_mask"])
    assignment = np.asarray(snap["assignment"])
    obstacles = snap["obstacles"]
    pred = enc_out["predictions"]
    unc = enc_out["uncertainty"]

    obs_rows: List[torch.Tensor] = []
    zero_row = torch.zeros(v2_obs_dim(max_pursuers, H, future_steps),
                           dtype=torch.float32, device=device)
    for i in range(n_p):
        if not pursuer_mask[i]:
            obs_rows.append(zero_row)
            continue
        p_pos = positions[i]
        base = [
            p_pos[0] / length, p_pos[1] / length,
            velocities[i][0] / 0.35, velocities[i][1] / 0.35,
            p_pos[0] / length, (length - p_pos[0]) / length,
            p_pos[1] / length, (length - p_pos[1]) / length,
            p_pos[2] / 2.5, velocities[i][2] / 0.35,
        ]
        team = []
        others = sorted(
            [(float(np.linalg.norm(positions[k] - p_pos)), k)
             for k in range(n_p) if k != i and pursuer_mask[k]])
        for _, k in others[: max_pursuers - 1]:
            team += [positions[k][0] / length, positions[k][1] / length,
                     positions[k][2] / 2.5]
        while len(team) < 3 * (max_pursuers - 1):
            team += [0.0, 0.0, 0.0]

        a = int(assignment[i]) if i < len(assignment) else -1
        if 0 <= a < n_ev and task_mask[a]:
            t_pos = [positions[n_p + a][0] / length, positions[n_p + a][1] / length,
                     positions[n_p + a][2] / 2.5]
            pred_feat = np.asarray(pred[a]).reshape(-1) / length
            u = [float(unc[a])]
        else:
            weights = []
            for j in range(n_ev):
                if not task_mask[j]:
                    weights.append(0.0)
                    continue
                d = float(np.linalg.norm(positions[n_p + j] - p_pos))
                weights.append(1.0 / (1.0 + d))
            wsum = float(np.sum(weights)) or 1.0
            weights = [w / wsum for w in weights]
            t_pos = [0.0, 0.0, 0.0]
            for j in range(n_ev):
                if weights[j] <= 0:
                    continue
                t_pos[0] += weights[j] * positions[n_p + j][0] / length
                t_pos[1] += weights[j] * positions[n_p + j][1] / length
                t_pos[2] += weights[j] * positions[n_p + j][2] / 2.5
            pred_feat = np.zeros(2 * future_steps, dtype=np.float32)
            u = [0.0]

        same = 1.0 if (0 <= a < task_mask.shape[0] and task_mask[a]) else 0.0
        same_count = 0.0
        if same > 0.5:
            # Match execution semantics: controller._build_v2_obs reports the
            # number of agents assigned to this agent's accepted target.  The
            # old code used K[i].sum(), which is the row mass (~1) rather than
            # the target's coalition size (~3), creating a train/eval shift.
            same_count = float(K[:, a].sum().detach().cpu())
        n_avail = max(1, int(pursuer_mask[:n_p].sum()))
        membership_feat = [same, same_count / n_avail]

        obst = []
        for o in sorted(obstacles, key=lambda ob: float(np.linalg.norm(np.asarray(ob[0])[:2] - p_pos[:2])))[:2]:
            obst += [np.asarray(o[0])[0] / length, np.asarray(o[0])[1] / length]
        while len(obst) < 4:
            obst += [0.0, 0.0]

        prefix = np.concatenate([
            np.asarray(base, dtype=np.float32),
            np.asarray(team, dtype=np.float32),
            np.asarray(t_pos, dtype=np.float32),
            np.asarray(membership_feat, dtype=np.float32),
            np.asarray(obst, dtype=np.float32),
        ])
        suffix = np.concatenate([pred_feat.astype(np.float32),
                                 np.asarray(u, dtype=np.float32)])
        # 图块保持 autograd 连接（训练上下文梯度经 t_ctx/c_ctx 回传）
        obs_rows.append(torch.cat([
            torch.as_tensor(prefix, dtype=torch.float32, device=device),
            pursuer_emb[i],
            t_ctx[i],
            c_ctx[i],
            torch.as_tensor(suffix, dtype=torch.float32, device=device),
        ]))
    while len(obs_rows) < max_pursuers:
        obs_rows.append(zero_row)
    return torch.stack(obs_rows)


def compute_demand_and_plan(encoder: StageFeatureEncoder,
                            demand_head: Optional[ContinuousDemandHead],
                            enc_out: Dict,
                            snap: Dict,
                            max_pursuers: int,
                            max_evaders: int,
                            demand_source: str,
                            fixed_demand: float,
                            epsilon_ot: float,
                            sinkhorn_iters: int):
    """从编码器输出计算连续需求与软计划（训练期重算用）。"""
    device = enc_out["pursuer_embeddings"].device
    n_ev = int(snap["num_evaders"])
    n_p = int(snap["num_pursuers"])
    task_mask_t = torch.as_tensor(
        np.pad(np.asarray(snap["task_mask"])[:n_ev].astype(np.float32),
               (0, max_evaders - n_ev)), device=device)
    pursuer_mask_t = torch.as_tensor(
        np.pad(np.asarray(snap["pursuer_mask"])[:n_p].astype(np.float32),
               (0, max_pursuers - n_p)), device=device)
    if demand_source == "learned_continuous":
        if demand_head is None:
            raise RuntimeError("demand_source=learned_continuous 但 demand_head 缺失")
        demand = demand_head(enc_out["task_embeddings"], task_mask_t)
    else:
        demand = torch.full((max_evaders,), float(fixed_demand),
                            device=device) * task_mask_t
    costs = enc_out["relation_costs"]
    if costs.shape[1] < max_evaders:
        costs = F.pad(costs, (0, max_evaders - costs.shape[1]))
    plan = build_soft_plan(costs, demand, pursuer_mask_t, task_mask_t,
                           epsilon=epsilon_ot, iters=sinkhorn_iters)
    return demand, plan


def hard_K_from_assignment(assignment, task_mask, max_pursuers, max_evaders, device):
    K = torch.zeros((max_pursuers, max_evaders), dtype=torch.float32, device=device)
    n_p = min(len(assignment), max_pursuers)
    for i in range(n_p):
        a = int(assignment[i])
        if 0 <= a < max_evaders and a < len(task_mask) and task_mask[a]:
            K[i, a] = 1.0
    return K


# ---------------------------------------------------------------------------
# 回放
# ---------------------------------------------------------------------------


class V2Replay:
    """episode 保持的回放；采样 L+1 条 transition 实现 L 步对齐契约。"""

    def __init__(self, max_episodes: int = 200):
        self.episodes: deque = deque(maxlen=int(max_episodes))

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def add_episode(self, transitions: Sequence[Dict]):
        if transitions:
            self.episodes.append(copy.deepcopy(list(transitions)))

    def sample_segment(self, plan_length: int, rng: np.random.Generator) -> List[Dict]:
        need = plan_length + 1
        eligible = [ep for ep in self.episodes if len(ep) >= need]
        if not eligible:
            raise ValueError(f"没有长度 >= {need} 的 episode")
        ep = eligible[int(rng.integers(0, len(eligible)))]
        start = int(rng.integers(0, len(ep) - need + 1))
        return copy.deepcopy(ep[start:start + need])


# ---------------------------------------------------------------------------
# 片段编码：L 条 transition + L+1 对齐决策状态
# ---------------------------------------------------------------------------


def encode_segment_v2(encoder: StageFeatureEncoder,
                      demand_head: Optional[ContinuousDemandHead],
                      segment: Sequence[Dict],
                      plan_length: int,
                      context_mode: str,          # "soft" | "hard"
                      grad_enabled: bool,
                      demand_source: str,
                      fixed_demand: float,
                      epsilon_ot: float,
                      sinkhorn_iters: int,
                      max_pursuers: int,
                      max_evaders: int,
                      stop_plan_gradient: bool = False):
    """返回 dict：obs/next_obs [L,...]、K/next_K [L,Np,Nt]、action/reward/done。

    t+1 状态取下一条 transition 的 snapshot（= 下一次决策时刻的 post-assignment
    状态），满足交接文档 12.1 的对齐契约，片段末尾无例外。
    """
    if len(segment) != plan_length + 1:
        raise ValueError("片段必须是 plan_length+1 条 transition")
    device = next(encoder.parameters()).device

    def _encode(transition):
        snap = transition["snapshot"]
        with torch.set_grad_enabled(grad_enabled):
            enc_out = encoder.encode(snap, differentiable=grad_enabled)
        demand, plan = compute_demand_and_plan(
            encoder, demand_head, enc_out, snap, max_pursuers, max_evaders,
            demand_source, fixed_demand, epsilon_ot, sinkhorn_iters)
        if context_mode == "soft":
            K = plan.task_plan
        elif context_mode == "hard":
            K = hard_K_from_assignment(
                transition.get("assignment", snap.get("assignment", [])),
                snap["task_mask"], max_pursuers, max_evaders, device)
        else:
            raise ValueError(f"未知 context_mode {context_mode!r}")
        obs = build_v2_obs_torch(enc_out, K, snap, max_pursuers, max_evaders,
                                 stop_plan_gradient=stop_plan_gradient)
        return enc_out, obs, K, demand, plan

    first = int(segment[0]["snapshot"].get("episode_step", 0))
    encoder.reset(step=first)
    current = _encode(segment[0])
    obs_l, next_obs_l, k_l, mask_l = [], [], [], []
    actions, rewards, dones = [], [], []
    for i in range(plan_length):
        nxt = _encode(segment[i + 1])
        _, obs, K, _, _ = current
        _, next_obs, next_K, _, _ = nxt
        obs_l.append(obs)
        next_obs_l.append(next_obs)
        k_l.append(K)
        pm = np.asarray(segment[i]["snapshot"]["pursuer_mask"], dtype=np.float32)
        mask_l.append(np.pad(pm, (0, max_pursuers - len(pm))))
        actions.append(segment[i]["action"])
        rewards.append(segment[i]["reward"])
        dones.append(segment[i]["done"])
        current = nxt

    stack = lambda xs: torch.stack(xs)
    return {
        "obs": stack(obs_l),
        "next_obs": stack(next_obs_l),
        "K": stack(k_l),
        "mask": torch.as_tensor(np.stack(mask_l), dtype=torch.float32, device=device),
        "action": torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device),
        "reward": torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device),
        "done": torch.as_tensor(np.asarray(dones, dtype=np.float32), device=device),
    }


# ---------------------------------------------------------------------------
# 三阶段更新
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V2UpdateConfig:
    gamma: float = 0.999
    tau: float = 0.005
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    actor_action_norm_weight: float = 0.5
    actor_los_weight: float = 0.0
    actor_ring_weight: float = 0.0
    d_cap: float = 2.0
    logit_penalty: bool = True
    encoder_grad_clip: float = 10.0
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 5.0
    critic_reward_scale: float = 0.1
    critic_reward_clip: float = 10.0
    target_context: str = "hard"      # hard + no_grad | soft + no_grad（显式对照）
    freeze_encoder: bool = False      # 仅明确对照
    stop_plan_gradient: bool = False  # 仅明确消融
    critic_stop_plan_gradient: bool = False   # S2：critic 阶段不更新需求/关系头（值函数不碰分配）
    demand_scale_anchor: float = 0.0          # S3：λ(Σd_active - N)² 尺度锚定，0=关闭


def actor_los_auxiliary_loss(obs: torch.Tensor,
                             policy_actions: torch.Tensor,
                             mask: torch.Tensor,
                             max_pursuers: int) -> torch.Tensor:
    """Supervise the basic assigned-target line-of-sight direction.

    The V2 schema stores own ``xy/L`` at columns 0:2 and assigned target
    ``xy/L`` after the 10-value base block plus the padded teammate block.
    This auxiliary term is deliberately training-only: capture geometry and
    deployment actions remain controlled by the learned policy.  Rows without
    an accepted assignment are excluded via the membership-valid feature.
    """
    target_offset = 10 + 3 * (int(max_pursuers) - 1)
    own_xy = obs[..., 0:2]
    target_xy = obs[..., target_offset:target_offset + 2]
    assignment_valid = obs[..., target_offset + 3] > 0.5
    active = (mask > 0.5) & assignment_valid
    if not bool(active.any()):
        return policy_actions.sum() * 0.0
    delta = target_xy - own_xy
    desired = F.normalize(delta, p=2, dim=-1, eps=1e-6)
    return F.smooth_l1_loss(policy_actions[active], desired[active])


def actor_ring_auxiliary_targets(snapshots: Sequence[Dict],
                                 max_pursuers: int,
                                 d_cap: float,
                                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build distinct moving ring-slot directions for accepted coalition members."""
    targets = []
    valids = []
    for snap in snapshots:
        n_p = int(snap["num_pursuers"])
        n_e = int(snap["num_evaders"])
        positions = np.asarray(snap["positions"], dtype=np.float64)
        velocities = np.asarray(snap["velocities"], dtype=np.float64)
        assignment = np.asarray(snap["assignment"], dtype=np.int64)[:n_p]
        # V2 intentionally has no learned/legacy role tiers.  Mark every
        # accepted member as core so the geometric helper consumes exactly the
        # same coalition used by the capture rule.
        roles = np.zeros(n_p, dtype=np.int64)
        task_mask = np.asarray(snap["task_mask"], dtype=bool)[:n_e]
        feat = unique_ring_slot_features(
            positions[:n_p], positions[n_p:n_p + n_e],
            np.zeros((n_e, 1, 2), dtype=np.float32),
            assignment, roles, task_mask, float(d_cap), float(snap["length"]),
            evader_velocities=velocities[n_p:n_p + n_e], normalizer=1.0,
        )
        target = np.zeros((max_pursuers, 2), dtype=np.float32)
        valid = np.zeros(max_pursuers, dtype=bool)
        count = min(n_p, max_pursuers)
        delta = torch.as_tensor(feat[:count, :2], dtype=torch.float32)
        target[:count] = F.normalize(delta, p=2, dim=-1, eps=1e-6).numpy()
        valid[:count] = feat[:count, 2] > 0.5
        targets.append(target)
        valids.append(valid)
    return (
        torch.as_tensor(np.stack(targets), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(valids), dtype=torch.bool, device=device),
    )


def actor_ring_auxiliary_loss(snapshots: Sequence[Dict],
                              policy_actions: torch.Tensor,
                              mask: torch.Tensor,
                              max_pursuers: int,
                              d_cap: float) -> torch.Tensor:
    target, valid = actor_ring_auxiliary_targets(
        snapshots, max_pursuers, d_cap, policy_actions.device)
    active = (mask > 0.5) & valid
    if not bool(active.any()):
        return policy_actions.sum() * 0.0
    return F.smooth_l1_loss(policy_actions[active], target[active])


def _soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def v2_update(modules: Dict[str, torch.nn.Module],
              optimizers: Dict[str, torch.optim.Optimizer],
              segment: Sequence[Dict],
              config: V2UpdateConfig,
              plan_length: int,
              demand_source: str,
              fixed_demand: float,
              epsilon_ot: float,
              sinkhorn_iters: int,
              max_pursuers: int,
              max_evaders: int,
              update_index: int = 1) -> Dict[str, float]:
    encoder = modules["encoder"]
    actor = modules["actor"]
    critic1, critic2 = modules["critic1"], modules["critic2"]
    actor_t = modules["actor_target"]
    critic1_t, critic2_t = modules["critic1_target"], modules["critic2_target"]
    encoder_t = modules.get("encoder_target")
    demand_head = modules.get("demand_head")

    encoder.train(not config.freeze_encoder)
    actor.train()
    critic1.train()
    critic2.train()

    logs: Dict[str, float] = {"aux_skipped": 0.0}

    # ---- Critic 阶段：在线 soft+grad；bootstrap 目标 target_context + no_grad ----
    batch = encode_segment_v2(
        encoder, demand_head, segment, plan_length,
        context_mode="soft", grad_enabled=not config.freeze_encoder,
        demand_source=demand_source, fixed_demand=fixed_demand,
        epsilon_ot=epsilon_ot, sinkhorn_iters=sinkhorn_iters,
        max_pursuers=max_pursuers, max_evaders=max_evaders,
        stop_plan_gradient=config.stop_plan_gradient or config.critic_stop_plan_gradient)
    with torch.no_grad():
        target_encoder = encoder_t if encoder_t is not None else encoder
        tgt = encode_segment_v2(
            target_encoder, demand_head, segment, plan_length,
            context_mode=config.target_context, grad_enabled=False,
            demand_source=demand_source, fixed_demand=fixed_demand,
            epsilon_ot=epsilon_ot, sinkhorn_iters=sinkhorn_iters,
            max_pursuers=max_pursuers, max_evaders=max_evaders)
        next_actions = actor_t(tgt["next_obs"])
        noise = (torch.randn_like(next_actions) * config.target_policy_noise
                 ).clamp(-config.target_noise_clip, config.target_noise_clip)
        next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        q_next = torch.minimum(critic1_t(tgt["next_obs"], next_actions, tgt["mask"]),
                               critic2_t(tgt["next_obs"], next_actions, tgt["mask"]))
        r = (batch["reward"] * config.critic_reward_scale
             ).clamp(-config.critic_reward_clip, config.critic_reward_clip)
        y = r + config.gamma * (1.0 - batch["done"]) * q_next
    q1 = critic1(batch["obs"], batch["action"], batch["mask"])
    q2 = critic2(batch["obs"], batch["action"], batch["mask"])
    c1 = F.smooth_l1_loss(q1, y)
    c2 = F.smooth_l1_loss(q2, y)
    critic_loss = 0.5 * (c1 + c2)
    optimizers["encoder"].zero_grad(set_to_none=True)
    optimizers["critics"].zero_grad(set_to_none=True)
    critic_loss.backward()
    logs["loss/critic"] = float(critic_loss.detach())
    logs["q/mean"] = float(q1.detach().mean())
    if not config.freeze_encoder:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.encoder_grad_clip)
    torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()),
                                   config.critic_grad_clip)
    optimizers["critics"].step()
    if not config.freeze_encoder:
        optimizers["encoder"].step()

    # ---- Actor 阶段（延迟，Q1）----
    applied = int(update_index) % max(1, int(config.policy_delay)) == 0
    logs["actor_applied"] = float(applied)
    if applied:
        fresh = encode_segment_v2(
            encoder, demand_head, segment, plan_length,
            context_mode="soft", grad_enabled=not config.freeze_encoder,
            demand_source=demand_source, fixed_demand=fixed_demand,
            epsilon_ot=epsilon_ot, sinkhorn_iters=sinkhorn_iters,
            max_pursuers=max_pursuers, max_evaders=max_evaders,
            stop_plan_gradient=config.stop_plan_gradient)
        for p in list(critic1.parameters()) + list(critic2.parameters()):
            p.requires_grad_(False)
        logits = actor.forward_logits(fresh["obs"])
        policy_actions = torch.tanh(logits)
        actor_q = critic1(fresh["obs"], policy_actions, fresh["mask"]).mean()
        if config.logit_penalty:
            limit = math.atanh(0.90) * math.sqrt(2.0)
            excess = torch.relu(torch.linalg.vector_norm(logits, dim=-1) - limit)
        else:
            excess = torch.relu(torch.linalg.vector_norm(policy_actions, dim=-1) - 1.0)
        actor_loss = (-actor_q
                      + config.actor_action_norm_weight * excess.square().mean())
        if config.actor_los_weight > 0.0:
            los_loss = actor_los_auxiliary_loss(
                fresh["obs"], policy_actions, fresh["mask"], max_pursuers)
            actor_loss = actor_loss + config.actor_los_weight * los_loss
            logs["loss/actor_los"] = float(los_loss.detach())
        if config.actor_ring_weight > 0.0:
            ring_loss = actor_ring_auxiliary_loss(
                [tr["snapshot"] for tr in segment[:plan_length]],
                policy_actions, fresh["mask"], max_pursuers, config.d_cap)
            actor_loss = actor_loss + config.actor_ring_weight * ring_loss
            logs["loss/actor_ring"] = float(ring_loss.detach())
        optimizers["encoder"].zero_grad(set_to_none=True)
        optimizers["actor"].zero_grad(set_to_none=True)
        if "demand" in optimizers:
            optimizers["demand"].zero_grad(set_to_none=True)
        actor_loss.backward()
        # S3 尺度锚定：总需求锚向总供给（仅作用于 demand 参数）
        if "demand" in optimizers and config.demand_scale_anchor > 0.0:
            with torch.enable_grad():
                anchor_logs = []
                anchor_total = None
                for tr in segment[:plan_length]:
                    snap_ = tr["snapshot"]
                    enc_ = encoder.encode(snap_, differentiable=True)
                    _, plan_ = compute_demand_and_plan(
                        encoder, demand_head, enc_, snap_, max_pursuers, max_evaders,
                        demand_source, fixed_demand, epsilon_ot, sinkhorn_iters)
                    d_active = plan_.demand_raw.to(torch.float32)
                    n_avail = float(plan_.available_agent_mask.to(torch.float32).sum())
                    anchor = (d_active.sum() - n_avail) ** 2
                    anchor_logs.append(anchor)
                anchor_loss = config.demand_scale_anchor * torch.stack(anchor_logs).mean()
                anchor_loss.backward()
                logs["loss/demand_anchor"] = float(anchor_loss.detach())
        logs["loss/actor"] = float(actor_loss.detach())
        logs["loss/actor_value"] = float(-actor_q.detach())
        logs["action/saturation"] = float(
            (policy_actions.detach().abs() >= 0.98).float().mean())
        if not config.freeze_encoder:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.encoder_grad_clip)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.actor_grad_clip)
        optimizers["actor"].step()
        if "demand" in optimizers:
            optimizers["demand"].step()
        if not config.freeze_encoder:
            optimizers["encoder"].step()
        for p in list(critic1.parameters()) + list(critic2.parameters()):
            p.requires_grad_(True)

        # ---- Polyak：Actor、双 Critic、整个 eta（含 demand/relation 头）----
        _soft_update(actor_t, actor, config.tau)
        _soft_update(critic1_t, critic1, config.tau)
        _soft_update(critic2_t, critic2, config.tau)
        if encoder_t is not None:
            _soft_update(encoder_t, encoder, config.tau)

    # ---- 辅助阶段：仅关系 Smooth L1（无阶段 CE / 无人数一致性项）----
    aux = encoder.auxiliary_loss(encoder._step, batch_size=plan_length)
    if aux is None:
        logs["aux_skipped"] = 1.0
        logs["loss/relation"] = float("nan")
    else:
        relation_loss = aux["total"]
        if config.freeze_encoder:
            logs["loss/relation"] = float(relation_loss.detach())
        else:
            optimizers["encoder"].zero_grad(set_to_none=True)
            relation_loss.backward()
            logs["loss/relation"] = float(relation_loss.detach())
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.encoder_grad_clip)
            optimizers["encoder"].step()

    scalars = [v for v in logs.values() if isinstance(v, float) and math.isfinite(v)]
    if not all(math.isfinite(v) for v in scalars):
        raise FloatingPointError("non-finite V2 update metric")
    return logs
