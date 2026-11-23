"""Same-state d_cap=0.8 forks with near-capture potential supervision.

The executed alliance and forced 2+2/3+1/1+3 layouts are rolled out from an
identical decision state.  Labels prioritize the first five-step near-capture
(at radius 1.0) over real captures and shaping return.  Four lookahead values
are appended to the original 77 per-task candidate features.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .train_v3_hybrid import HybridPolicy, make_environment
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment
from .v3_demand_distillation import build_scale_aware_candidate_features
from .v3_dynamic_counterfactual import fork_environment
from .v3cf_dataset import (
    ANCHOR_SEED,
    ANCHOR_STEP,
    SHAPING_COMPONENTS,
    _PriorityReservoir,
    _event_priority,
    _jsonable,
    _load_policy,
    save_dataset,
    select_coverage_candidates,
)

DATASET_SCHEMA = "v3_same_state_near_capture_potential_value_v2"
BASE_FEATURE_DIM = 77
LOOKAHEAD_FEATURE_NAMES = (
    "closure_seconds", "escape_window_seconds", "time_margin_seconds", "max_gap_over_pi",
)
LOOKAHEAD_FEATURE_DIM = len(LOOKAHEAD_FEATURE_NAMES)
FEATURE_DIM = BASE_FEATURE_DIM + LOOKAHEAD_FEATURE_DIM
NEAR_CAPTURE_RADIUS = 1.0
NEAR_CAPTURE_HOLD = 5


def _max_angular_gap(relative_positions):
    relative = np.asarray(relative_positions, dtype=np.float64).reshape(-1, 2)
    if len(relative) < 2:
        return 2.0 * math.pi
    angles = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
    return float(np.diff(np.r_[angles, angles[0] + 2.0 * math.pi]).max())


def coalition_geometry(pursuer_positions, task_position, member_ids):
    """Return all-member maximum planar distance and angular gap in radians."""
    members = np.asarray(pursuer_positions, dtype=np.float64)[list(member_ids), :2]
    if not len(members):
        return None, 2.0 * math.pi
    relative = members - np.asarray(task_position, dtype=np.float64)[:2]
    return float(np.linalg.norm(relative, axis=1).max()), _max_angular_gap(relative)


class NearCaptureTracker:
    """Track a sustained assigned-coalition enclosure independently per task."""

    def __init__(self, radius=NEAR_CAPTURE_RADIUS, capture_size=3,
                 required_hold=NEAR_CAPTURE_HOLD):
        if radius <= 0 or capture_size < 1 or required_hold < 1:
            raise ValueError("radius, capture_size and required_hold must be positive")
        self.radius = float(radius)
        self.capture_size = int(capture_size)
        self.required_hold = int(required_hold)
        self.streaks = {}
        self.near_capture_step = None
        self.min_d_max = None

    def update(self, snapshot, assignment, step):
        """Consume one branch state; return the step only when first achieved."""
        positions = np.asarray(snapshot["positions"], dtype=np.float64)
        n = int(snapshot["num_pursuers"])
        m = int(snapshot["num_evaders"])
        assignment = np.asarray(assignment, dtype=np.int64)
        alive = np.asarray(snapshot["task_mask"][:m], dtype=bool)
        pursuer_mask = np.asarray(
            snapshot.get("pursuer_mask", np.ones(n, dtype=bool))[:n], dtype=bool,
        )
        for task_id in range(m):
            if not alive[task_id]:
                self.streaks[task_id] = 0
                continue
            member_ids = np.flatnonzero((assignment == task_id) & pursuer_mask)
            if len(member_ids) < self.capture_size:
                self.streaks[task_id] = 0
                continue
            d_max, gap = coalition_geometry(
                positions[:n], positions[n + task_id], member_ids,
            )
            self.min_d_max = (d_max if self.min_d_max is None
                              else min(self.min_d_max, d_max))
            enclosed = d_max <= self.radius and gap < math.pi
            self.streaks[task_id] = self.streaks.get(task_id, 0) + 1 if enclosed else 0
            if (self.near_capture_step is None
                    and self.streaks[task_id] >= self.required_hold):
                self.near_capture_step = int(step)
                return self.near_capture_step
        return None


def build_lookahead_features(pursuer_positions, task_positions, evader_goals,
                             assignment, v_max, v_max_e):
    """Return [closure s, escape window s, window-closure s, gap/pi] per task."""
    if not math.isfinite(v_max) or not math.isfinite(v_max_e) or v_max <= 0 or v_max_e <= 0:
        raise ValueError("v_max and v_max_e must be finite and positive")
    pursuers = np.asarray(pursuer_positions, dtype=np.float64)
    tasks = np.asarray(task_positions, dtype=np.float64)
    goals = np.asarray(evader_goals, dtype=np.float64)
    assignment = np.asarray(assignment, dtype=np.int64)
    if len(tasks) != len(goals) or len(pursuers) != len(assignment):
        raise ValueError("positions, goals and assignment must be aligned")
    rows = []
    for task_id, task in enumerate(tasks):
        member_ids = np.flatnonzero(assignment == task_id)
        d_max, gap = coalition_geometry(pursuers, task, member_ids)
        closure_seconds = 0.0 if d_max is None else d_max / float(v_max)
        remaining = float(np.linalg.norm(goals[task_id, :2] - task[:2]))
        escape_window = remaining / float(v_max_e)
        rows.append([
            closure_seconds, escape_window, escape_window - closure_seconds,
            gap / math.pi,
        ])
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite lookahead feature")
    return result


def append_lookahead_features(base_features, pursuer_positions, task_positions,
                              evader_goals, assignment, v_max, v_max_e):
    """Append the four lookahead columns after the frozen 77-column layout."""
    base = torch.as_tensor(base_features)
    if base.ndim != 2 or base.shape[1] != BASE_FEATURE_DIM:
        raise ValueError(f"expected [num_tasks, {BASE_FEATURE_DIM}] base features")
    lookahead = build_lookahead_features(
        pursuer_positions, task_positions, evader_goals, assignment, v_max, v_max_e,
    )
    extra = torch.as_tensor(lookahead, dtype=base.dtype, device=base.device)
    return torch.cat([base, extra], dim=-1)


def near_capture_potential_score(summary, horizon=300, capture_reward=10.0,
                                 full_capture_bonus=50.0):
    """Lexicographic score: achieved > missing, then earlier > later, then captures."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    step = summary.get("near_capture_step")
    new_captures = int(summary.get("delta_captured", summary.get("new_captures", 0)))
    real_capture = (float(capture_reward) * new_captures
                    + float(full_capture_bonus) * bool(summary.get("full_capture", False)))
    if step is None:
        score = real_capture
    else:
        step = int(step)
        if not 1 <= step <= horizon:
            raise ValueError("near_capture_step must lie within the rollout horizon")
        # One step is worth more than the largest possible two-evader reward
        # difference, so capture rewards can never reverse early > late.
        score = 1_000_000.0 + 1_000.0 * (horizon + 1 - step) + real_capture
    if not math.isfinite(score):
        raise ValueError("non-finite near-capture G_H")
    return float(score)


@torch.no_grad()
def rollout_near_capture_label(env, encoder, actor, first_encoded, assignment,
                               horizon=300, max_steps=1000, gamma=0.99):
    """Roll out one locked alliance and measure near and actual capture events."""
    branch = fork_environment(env)
    branch_encoder = copy.deepcopy(encoder)
    initial = np.asarray(assignment, dtype=np.int64).copy()
    captured_before = int(branch.captured[:branch.num_evaders].sum())
    rewards, first_capture_step = [], None
    tracker = NearCaptureTracker()
    for offset in range(int(horizon)):
        active = np.flatnonzero(branch.task_mask[:branch.num_evaders])
        if not len(active) or branch.done:
            break
        repaired = initial.copy()
        repaired[~np.isin(repaired, active)] = int(active[0])
        apply_membership(branch, repaired)
        snapshot = branch.snapshot()
        # Observe before stepping: the fifth near-capture observation can
        # coincide with a real 0.8 capture that removes the task in step().
        tracker.update(snapshot, repaired, offset + 1)
        encoded = (first_encoded if offset == 0 else branch_encoder.encode(
            snapshot, differentiable=False,
        ))
        membership = hard_K_from_assignment(
            repaired, branch.task_mask, 6, 3,
            encoded["pursuer_embeddings"].device,
        )
        obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
        mask = torch.as_tensor(
            branch.pursuer_mask, dtype=torch.float32, device=obs.device,
        )
        actions = (actor(obs) * mask.unsqueeze(-1)).cpu().numpy()
        reward, done, info = branch.step(actions, max_steps=max_steps)
        shaping = sum(float(info["reward_components"].get(name, 0.0))
                      for name in SHAPING_COMPONENTS)
        rewards.append(float(reward) - shaping)
        if first_capture_step is None and info.get("capture_events"):
            first_capture_step = offset + 1
        if done:
            break
    raw_return = float(sum((float(gamma) ** index) * value
                           for index, value in enumerate(rewards)))
    captured_after = int(branch.captured[:branch.num_evaders].sum())
    result = {
        "near_capture_step": tracker.near_capture_step,
        "near_capture_radius": tracker.radius,
        "near_capture_required_hold": tracker.required_hold,
        "min_d_max": tracker.min_d_max,
        "delta_captured": max(0, captured_after - captured_before),
        "new_captures": max(0, captured_after - captured_before),
        "captured_before": captured_before,
        "captured_after": captured_after,
        "full_capture": bool(captured_after == branch.num_evaders),
        "first_capture_step": first_capture_step,
        "raw_return": raw_return,
        "discounted_return": raw_return,
        "steps": len(rewards),
        "outcome": str(branch.outcome),
    }
    result["G_H"] = near_capture_potential_score(result, horizon=horizon)
    return result


def _make_record(context, modules, config, horizon, include_extreme=False,
                 state_id=None):
    env = context["env"]
    snapshot = context["decision"]["snapshot"]
    encoded = context["decision"]["encoded"]
    n, m = snapshot["num_pursuers"], snapshot["num_evaders"]
    alive = np.asarray(snapshot["task_mask"][:m], dtype=bool)
    candidates, modes = select_coverage_candidates(
        encoded["relation_costs"][:n, :m], context["decision"]["assignment"],
        alive, include_extreme=include_extreme,
    )
    positions = np.asarray(snapshot["positions"])
    features, labels, diagnostics = [], [], []
    for candidate in candidates:
        base = build_scale_aware_candidate_features(
            encoded["pursuer_embeddings"][:n], encoded["task_embeddings"][:m],
            positions[:n], positions[n:n + m], candidate,
            int(config["capture_size"]),
        ).detach().cpu()
        features.append(append_lookahead_features(
            base, positions[:n], positions[n:n + m], env.evader_goals[:m],
            candidate, float(env.v_max), float(env.v_max_e),
        ))
        diagnostics.append(rollout_near_capture_label(
            env, modules["encoder"], modules["actor"], encoded, candidate,
            horizon=horizon, max_steps=int(config["max_steps"]),
            gamma=float(config["gamma"]),
        ))
        labels.append(diagnostics[-1]["G_H"])
    return {
        "schema": DATASET_SCHEMA,
        "state_id": state_id or f"{env.scenario_seed}:{env.episode_step_count}",
        "scale": f"{n}v{m}", "seed": int(env.scenario_seed),
        "episode_step": int(env.episode_step_count),
        "feature_dim": FEATURE_DIM,
        "base_feature_dim": BASE_FEATURE_DIM,
        "lookahead_feature_names": list(LOOKAHEAD_FEATURE_NAMES),
        "features": torch.stack(features).numpy().tolist(),
        "near_capture_step": [row["near_capture_step"] for row in diagnostics],
        "min_d_max": [row["min_d_max"] for row in diagnostics],
        "capture_labels": diagnostics,
        "G_H": labels,
        "alive": np.repeat(alive[None, :], len(candidates), axis=0).tolist(),
        "assignments": candidates.tolist(),
        "modes": [list(mode) for mode in modes],
        "meta": {
            "sampling_reason": context["reason"],
            "sampling_priority": float(context["priority"]),
            "scenario": context["scenario"],
            "d_cap": 0.8,
            "horizon": int(horizon),
            "state_snapshot": _jsonable(snapshot),
            "evader_goals": _jsonable(env.evader_goals[:m]),
            "encoder_state": "omitted; use recorded features and labels",
        },
    }


def collect_episode_records(modules, config, seed, horizon=300, max_states=2,
                            forced_step=None):
    env = make_environment(config, d_cap=0.8)
    env.scenario_sampler = Deploy5x5Sampler()
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=int(seed))
    try:
        policy = HybridPolicy(modules, config)
        reservoir = _PriorityReservoir(max_states, int(seed) + 991)
        decision = policy.decision(env)
        last_progress_step, last_capture_count = 0, 0
        for _ in range(int(config["max_steps"])):
            if decision["allocated"] and int(np.sum(env.task_mask[:2])) == 2:
                priority, reason = _event_priority(
                    decision, env, last_progress_step, last_capture_count,
                )
                if forced_step is not None:
                    if env.episode_step_count == int(forced_step):
                        priority, reason = 100.0, "anchor_3plus1_switch"
                    else:
                        priority = -1.0
                if forced_step is None or priority >= 0:
                    reservoir.add(priority, {
                        "env": fork_environment(env), "decision": decision,
                        "priority": priority, "reason": reason,
                        "scenario": "deploy5x5",
                    })
            captured = int(env.captured[:2].sum())
            if captured > last_capture_count:
                last_capture_count = captured
                last_progress_step = env.episode_step_count
            _, done, _ = env.step(decision["actions"], max_steps=int(config["max_steps"]))
            if done:
                break
            decision = policy.decision(env)
        contexts = reservoir.contexts()
        if forced_step is not None and len(contexts) != 1:
            raise RuntimeError(f"anchor seed {seed} did not reach allocation step {forced_step}")
        records = []
        for index, context in enumerate(contexts):
            include_extreme = (int(seed) + index) % 10 == 0
            records.append(_make_record(
                context, modules, config, horizon, include_extreme,
                state_id=f"{seed}:{context['env'].episode_step_count}:deploy5x5",
            ))
        return records
    finally:
        env.close()


def iter_dataset(path) -> Iterator[dict]:
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row.get("schema") != DATASET_SCHEMA:
                        raise ValueError(f"unsupported dataset schema in {file}")
                    yield row


def load_dataset(path):
    return list(iter_dataset(path))


def generate_dataset(args):
    shard_index, shard_count = args.shard
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard requires 0 <= i < n")
    modules, config = _load_policy(args.checkpoint, torch.device(args.device))
    records = []
    if shard_index == 0:
        records.extend(collect_episode_records(
            modules, config, ANCHOR_SEED, args.horizon,
            max_states=1, forced_step=ANCHOR_STEP,
        ))
    episode = 0
    while len(records) < int(args.states):
        seed = int(args.seed) + shard_index * 1_000_000 + episode * shard_count
        records.extend(collect_episode_records(
            modules, config, seed, args.horizon,
        ))
        episode += 1
    records = records[:int(args.states)]
    if shard_index == 0 and not any(row["seed"] == ANCHOR_SEED for row in records):
        raise RuntimeError("fixed anchor was truncated from shard zero")
    path = save_dataset(records, args.out, f"{shard_index}/{shard_count}")
    print(json.dumps({"states": len(records), "output": str(path)}))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=400)
    parser.add_argument("--out", default="checkpoints/v3cf2/dataset")
    parser.add_argument("--workers", type=int, default=1,
                        help="launch shards externally; retained for CLI compatibility")
    parser.add_argument("--shard", type=lambda text: tuple(map(int, text.split("/"))),
                        default=(0, 1), metavar="I/N")
    parser.add_argument("--seed", type=int, default=920000)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_08_parallel/promoted/actor/winner.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    if args.states < 1 or args.workers < 1 or args.horizon < 1:
        raise ValueError("states, workers and horizon must be positive")
    generate_dataset(args)


if __name__ == "__main__":
    main()
