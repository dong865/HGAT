"""v3cf2 集成：81 维价值头（含前视特征）在线策略包装。

不修改任何共享文件：ScaleAwareCoalitionValueHead 的首层扩为 81 输入，
allocate 在调用价值头前给候选特征拼接 4 个前视量（闭合秒数/逃逸时间窗/时间余量/gap）。
其余分配、门控、可行性惩罚、切换逻辑与 V3HybridAllocator 完全一致。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .v3_demand_distillation import (
    GEOMETRY_DIM, SCALE_FEATURE_DIM, ScaleAwareCoalitionValueHead,
    build_demand_geometry, build_scale_aware_candidate_features,
)
from .v3_hybrid_controller import V3HybridAllocator
from .train_v3_hybrid import HybridPolicy


class ValueHead81(ScaleAwareCoalitionValueHead):
    """与 77 维原版同构，仅首层输入扩到 81（原 77 + 4 前视量）。"""

    def __init__(self, hidden_dim: int, width: int = 64):
        super().__init__(hidden_dim, width)
        feature_dim = 2 * int(hidden_dim) + GEOMETRY_DIM + SCALE_FEATURE_DIM + 4
        self.task_encoder[0] = nn.Linear(feature_dim, width)


def augment_forward_features(base: torch.Tensor, candidate: np.ndarray,
                             snapshot: dict, evader_goals: np.ndarray,
                             capture_size: int = 3) -> torch.Tensor:
    """[M,77] -> [M,81]：为某候选的每个任务追加前视特征。"""
    positions = np.asarray(snapshot["positions"])
    num_agents = int(len(candidate))
    num_tasks = int(base.shape[0])
    pursuer_pos = positions[:num_agents, :2]
    task_pos = positions[num_agents:num_agents + num_tasks, :2]
    v_max = float(snapshot.get("v_max", 0.35))
    v_max_e = float(snapshot.get("v_max_e", 0.35))
    rows = []
    for j in range(num_tasks):
        members = pursuer_pos[np.asarray(candidate) == j]
        if len(members):
            rel = members - task_pos[j]
            dists = np.linalg.norm(rel, axis=1)
            dmax = float(dists.max())
            ang = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
            gap = float(np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max()) if len(ang) >= 2 else 2 * np.pi
        else:
            dmax, gap = 0.0, 2 * np.pi
        closure_s = dmax / max(v_max, 1e-6)
        goal_d = float(np.linalg.norm(evader_goals[j][:2] - task_pos[j]))
        window_s = goal_d / max(v_max_e, 1e-6)
        rows.append([closure_s, window_s, window_s - closure_s, gap / (2 * np.pi)])
    aug = torch.as_tensor(np.asarray(rows, dtype=np.float32), device=base.device, dtype=base.dtype)
    return torch.cat([base, aug], dim=-1)


class V3cf2Allocator(V3HybridAllocator):
    """在价值头调用处插入前视特征增广；其余流程与父类逐行一致。"""

    def allocate(self, encoded, snapshot, alive_mask, step, force=False,
                 hold_counters=None, remaining_fraction=1.0, capture_urgent=False):
        # 借用父类流程：先完整调用父类 allocate 得到 candidates 等全部中间量不可行
        # （父类内部直接用 value_head），因此这里复制父类主体，仅在两处价值头调用前增广。
        pursuer_embeddings = encoded["pursuer_embeddings"]
        task_embeddings = encoded["task_embeddings"]
        relation_costs = encoded["relation_costs"]
        num_agents = int(encoded.get("num_pursuers", pursuer_embeddings.shape[0]))
        num_tasks = int(encoded.get("num_evaders", len(alive_mask)))
        alive = torch.as_tensor(alive_mask, dtype=torch.bool,
                                device=task_embeddings.device)[:num_tasks]
        positions = np.asarray(snapshot["positions"])
        pursuer_positions = positions[:num_agents, :2]
        task_positions = positions[num_agents:num_agents + num_tasks, :2]
        geometry = build_demand_geometry(
            pursuer_positions, task_positions, self.map_size).to(task_embeddings)
        demand = self.demand_head(task_embeddings[:num_tasks], alive, num_agents,
                                  self.capture_size, geometry)
        from .v3_sparse_demand import capacities_from_sparse_demand, neighbor_capacity_modes
        capacity = capacities_from_sparse_demand(demand, alive, num_agents)
        previous = self.gate.assignment
        invalid_previous = (previous is not None and (
            previous.shape != (num_agents,) or np.any(previous < 0)
            or np.any(previous >= num_tasks)
            or any(not bool(alive[t]) for t in previous)))
        usable_previous = None if invalid_previous else previous
        current_mode = (None if usable_previous is None else tuple(
            np.bincount(usable_previous, minlength=num_tasks).tolist()))
        modes = neighbor_capacity_modes(capacity, alive, num_agents, self.mode_limit,
                                        current_mode=current_mode)
        if hasattr(self.demand_head, "score_modes"):
            mode_logits, mode_probabilities = self.demand_head.score_modes(
                task_embeddings[:num_tasks], alive, num_agents, modes,
                self.capture_size, geometry)
        else:
            mode_logits = task_embeddings.new_zeros(len(modes))
            mode_probabilities = torch.softmax(mode_logits, dim=0)
        if self.freeze_guard is not None and usable_previous is not None:
            holds = (np.zeros(num_tasks, dtype=np.int64) if hold_counters is None
                     else np.asarray(hold_counters)[:num_tasks])
            frozen_tasks = self.freeze_guard.update(
                usable_previous, pursuer_positions, task_positions,
                alive.detach().cpu().numpy(), holds)
        else:
            frozen_tasks = np.zeros(num_tasks, dtype=bool)
        from .v3_candidate_generator import generate_bounded_candidates
        candidates = generate_bounded_candidates(
            relation_costs[:num_agents, :num_tasks], modes, alive,
            current_assignment=usable_previous, max_candidates=self.max_candidates)
        if (frozen_tasks.any() and usable_previous is not None
                and not np.any(np.all(candidates == usable_previous, axis=1))):
            if len(candidates) >= self.max_candidates:
                candidates = np.vstack([usable_previous, candidates[:-1]])
            else:
                candidates = np.vstack([usable_previous, candidates])
        # 状态分级：远离捕获区时价值头离线判别不可靠，改用几何赶路分；
        # 接近捕获区（任一目标最近3机均值<1.5m）才启用 81 维价值头。
        near_capture = False
        for j in range(num_tasks):
            if not bool(alive[j]):
                continue
            dists = np.sort(np.linalg.norm(pursuer_positions - task_positions[j], axis=1))[:3]
            if float(dists.mean()) < 1.5:
                near_capture = True
                break
        if near_capture:
            features = torch.stack([
                augment_forward_features(
                    build_scale_aware_candidate_features(
                        pursuer_embeddings[:num_agents], task_embeddings[:num_tasks],
                        pursuer_positions, task_positions, candidate, self.capture_size),
                    candidate, snapshot, self.evader_goals, self.capture_size)
                for candidate in candidates
            ])
            raw_values = self.value_head(features, alive).reshape(-1)
        else:
            rows = []
            for candidate in candidates:
                per_task = []
                for j in range(num_tasks):
                    if not bool(alive[j]):
                        per_task.append(0.0)
                        continue
                    members = pursuer_positions[np.asarray(candidate) == j]
                    if len(members):
                        d = np.linalg.norm(members - task_positions[j], axis=1)
                        per_task.append(-float(d.mean()))
                    else:
                        per_task.append(-self.map_size)
                rows.append(float(np.sum(per_task)))
            raw_values = torch.as_tensor(rows, dtype=torch.float32,
                                         device=task_embeddings.device)
        from .v3_hybrid_controller import score_candidates, capture_feasibility_penalty
        unified_scores = score_candidates(
            raw_values, candidates, capacity, usable_previous,
            self.demand_weight, self.switch_weight)
        feasibility_penalty = capture_feasibility_penalty(
            candidates, alive.detach().cpu().numpy(), self.capture_size,
            self.infeasible_mode_penalty, remaining_fraction,
            self.infeasible_mode_warmup_fraction, capture_urgent,
        ).to(device=unified_scores.device, dtype=unified_scores.dtype)
        unified_scores = unified_scores - feasibility_penalty
        mode_to_probability = {tuple(mode): mode_probabilities[i] for i, mode in enumerate(modes)}
        candidate_mode_probabilities = torch.stack([
            mode_to_probability[tuple(np.bincount(c, minlength=num_tasks).tolist())]
            for c in candidates])
        if self.mode_prior_weight != 0.0:
            log_prior = candidate_mode_probabilities.clamp_min(1e-8).log()
            log_prior = log_prior.clamp(min=-self.mode_prior_clip, max=0.0)
            unified_scores = unified_scores + self.mode_prior_weight * log_prior
        incumbent_score = None
        if usable_previous is not None and not np.any(
                np.all(candidates == usable_previous, axis=1)):
            if near_capture:
                incumbent_features = augment_forward_features(
                    build_scale_aware_candidate_features(
                        pursuer_embeddings[:num_agents], task_embeddings[:num_tasks],
                        pursuer_positions, task_positions, usable_previous, self.capture_size),
                    usable_previous, snapshot, self.evader_goals, self.capture_size)
                incumbent_value = self.value_head(incumbent_features.unsqueeze(0), alive).reshape(-1)
            else:
                vals = []
                for j in range(num_tasks):
                    if not bool(alive[j]):
                        vals.append(0.0)
                        continue
                    members = pursuer_positions[np.asarray(usable_previous) == j]
                    if len(members):
                        vals.append(-float(np.linalg.norm(members - task_positions[j], axis=1).mean()))
                    else:
                        vals.append(-self.map_size)
                incumbent_value = torch.as_tensor([float(np.sum(vals))], dtype=torch.float32,
                                                  device=task_embeddings.device)
            incumbent_unified = score_candidates(
                incumbent_value, usable_previous.reshape(1, -1), capacity,
                usable_previous, self.demand_weight, self.switch_weight)
            incumbent_score = float(incumbent_unified[0].item())
        assignment, info = self.gate.select(
            candidates, unified_scores, step, force=force or invalid_previous,
            incumbent_score=incumbent_score, frozen_tasks=frozen_tasks)
        return {
            "assignment": assignment, "demand": demand, "capacity": capacity,
            "modes": modes, "mode_logits": mode_logits,
            "mode_probabilities": mode_probabilities,
            "candidate_mode_probabilities": candidate_mode_probabilities,
            "candidates": candidates, "candidate_count": len(candidates),
            "raw_values": raw_values, "unified_scores": unified_scores,
            "feasibility_penalty": feasibility_penalty,
            "switched": info["switched"], "reason": info["reason"],
            "frozen_tasks": frozen_tasks,
        }


class V3cf2Policy(HybridPolicy):
    """把默认分配器替换为带前视特征的 81 维价值头分配器。"""

    def __init__(self, modules, config):
        super().__init__(modules, config)
        self.allocator = V3cf2Allocator(
            modules["demand_head"], modules["value_head"], config["map_size"],
            capture_size=3, mode_limit=config["max_capacity_modes"],
            max_candidates=config["max_alliance_candidates"],
            demand_weight=config["demand_weight"], switch_weight=config["switch_weight"],
            freeze_distance=(config.get("freeze_distance_ratio", 0.8)
                             * config.get("d_cap", 2.0)),
            freeze_release_ratio=config.get("freeze_release_ratio", 1.15),
            mode_prior_weight=config.get("mode_prior_weight", 0.0),
            infeasible_mode_penalty=config.get("infeasible_mode_penalty", 0.0),
            infeasible_mode_warmup_fraction=config.get(
                "infeasible_mode_warmup_fraction", 0.5),
            **config["gate_config"],
        )
        self.reset()

    def decision(self, env):
        self.allocator.evader_goals = env.evader_goals
        return super().decision(env)
