"""Same-state alliance-fork dataset for the V3 d_cap=0.8 value head.

Only decision states are recorded.  Every state includes the executed alliance
and minimum-relation-cost representatives of 2+2, 3+1 and 1+3, independent of
what the current value head prefers.  Labels come from locked-alliance forks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch

from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .train_v3_hybrid import HybridPolicy, build_system, make_environment
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment
from .v3_candidate_generator import generate_bounded_candidates
from .v3_demand_distillation import (
    build_demand_geometry,
    build_scale_aware_candidate_features,
)
from .v3_dynamic_counterfactual import fork_environment

DATASET_SCHEMA = "v3_same_state_capture_priority_value_v1"
ANCHOR_SEED = 70086
ANCHOR_STEP = 170
SHAPING_COMPONENTS = ("closure", "angular_closure", "ring")


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def alliance_mode(assignment, num_tasks=2):
    return tuple(np.bincount(
        np.asarray(assignment, dtype=np.int64), minlength=int(num_tasks),
    )[:num_tasks].tolist())


def select_coverage_candidates(relation_costs, current_assignment, alive_mask,
                               include_extreme=False):
    """Return current plus one minimum-cost member layout per required mode.

    Deliberately no value scores enter this function.  Duplicate assignments
    are retained when the incumbent already is a mode's optimum, making the
    mode coverage explicit and keeping branch diagnostics aligned.
    """
    if isinstance(relation_costs, torch.Tensor):
        relation_costs = relation_costs.detach().cpu().numpy()
    costs = np.asarray(relation_costs, dtype=np.float64)
    current = np.asarray(current_assignment, dtype=np.int64)
    alive = np.asarray(alive_mask, dtype=bool)
    if costs.shape != (len(current), len(alive)) or alive.sum() != 2:
        raise ValueError("coverage candidates require a two-live-task 4v2 state")
    if len(current) != 4:
        raise ValueError("2+2/3+1/1+3 coverage requires four pursuers")
    modes = [(2, 2), (3, 1), (1, 3)]
    if include_extreme:
        modes.extend([(4, 0), (0, 4)])
    rows = [current.copy()]
    for mode in modes:
        best = generate_bounded_candidates(
            costs, [mode], alive, max_candidates=1,
        )[0]
        rows.append(best.copy())
    return np.stack(rows).astype(np.int64), [alliance_mode(row, 2) for row in rows]


def capture_priority_score(summary, horizon=100, r_cap=10.0,
                           full_capture_bonus=50.0, delay_weight=0.02,
                           progress_weight=0.1):
    """Capture-priority G_H; raw progress is discounted capture-free reward."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    new_captures = int(summary["new_captures"])
    first = summary.get("first_capture_step")
    delay = horizon if first is None else int(first)
    raw_progress = float(summary.get("discounted_return", 0.0))
    score = (float(r_cap) * new_captures
             + float(full_capture_bonus) * float(summary.get("full_capture", False))
             - float(delay_weight) * delay
             + float(progress_weight) * raw_progress)
    if not math.isfinite(score):
        raise ValueError("non-finite G_H")
    return float(score)


@torch.no_grad()
def rollout_capture_label(env, encoder, actor, first_encoded, assignment,
                          horizon=100, max_steps=1000, gamma=0.99):
    """Self-contained locked-alliance rollout with raw-reward diagnostics."""
    branch = fork_environment(env)
    branch_encoder = copy.deepcopy(encoder)
    initial = np.asarray(assignment, dtype=np.int64).copy()
    captured_before = int(branch.captured[:branch.num_evaders].sum())
    rewards, first_capture_step = [], None
    for offset in range(int(horizon)):
        active = np.flatnonzero(branch.task_mask[:branch.num_evaders])
        if not len(active) or branch.done:
            break
        repaired = initial.copy()
        repaired[~np.isin(repaired, active)] = int(active[0])
        apply_membership(branch, repaired)
        snapshot = branch.snapshot()
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
        "delta_captured": max(0, captured_after - captured_before),
        "new_captures": max(0, captured_after - captured_before),
        "captured_before": captured_before,
        "captured_after": captured_after,
        "full_capture": bool(captured_after == branch.num_evaders),
        "first_capture_step": first_capture_step,
        "first_capture_delay": (None if first_capture_step is None
                                else float(first_capture_step)),
        "raw_return": raw_return,
        "discounted_return": raw_return,
        "steps": len(rewards),
        "outcome": str(branch.outcome),
    }
    result["G_H"] = capture_priority_score(result, horizon=horizon)
    return result


def _load_policy(checkpoint, device):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config.update({
        "device": str(device), "d_cap": 0.8,
        "init_v3_from": str(checkpoint),
        "init_policy_from": None, "init_value_from": None,
        "torch_threads": 1,
    })
    modules, _, _ = build_system(config)
    for module in modules.values():
        module.eval()
    return modules, config


def _event_priority(decision, env, last_progress_step, last_capture_count):
    """Return (priority, reason) for an allocation decision state."""
    snapshot = decision["snapshot"]
    n, m = snapshot["num_pursuers"], snapshot["num_evaders"]
    positions = np.asarray(snapshot["positions"])
    alive = np.asarray(snapshot["task_mask"][:m], dtype=bool)
    geometry = build_demand_geometry(
        positions[:n], positions[n:n + m], snapshot["length"],
    ).numpy()
    nearest_three_means = []
    for task_id in np.flatnonzero(alive):
        distances = np.linalg.norm(
            positions[:n, :2] - positions[n + task_id, :2], axis=1,
        )
        nearest_three_means.append(float(np.sort(distances)[:3].mean()))
    near_distance = bool(min(nearest_three_means) < 1.5)
    near_closure = bool(np.any((geometry[alive, 4] < 0.55)
                               & (geometry[alive, 3] < 2.0 / snapshot["length"])))
    scores = decision["allocation"]["unified_scores"].detach().cpu().numpy()
    top_gap = float(np.sort(scores)[-1] - np.sort(scores)[-2]) if len(scores) > 1 else 1.0
    reason = decision["allocation"].get("reason", "")
    switched = bool(decision["allocation"].get("switched", False))
    stale = (alliance_mode(decision["assignment"], m) == (2, 2)
             and env.episode_step_count - last_progress_step >= 200)
    if near_distance or near_closure:
        return 4.0 - min(nearest_three_means) / 1.5, "capture_critical"
    if reason not in ("", "initial", "improved") or switched:
        return 3.0, "gate_switch" if switched else "gate_rejected"
    if top_gap < 0.15:
        return 2.0 - min(top_gap, 0.15), "value_uncertain"
    if stale:
        return 1.0, "stalemate_2plus2"
    return 0.0, "ordinary_fallback"


class _PriorityReservoir:
    """Bounded random tie-breaking reservoir, ordered by event priority."""
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.rows = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, priority, context):
        self.seen += 1
        tie = float(self.rng.random())
        self.rows.append((float(priority), tie, context))
        self.rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        del self.rows[self.capacity:]

    def contexts(self):
        return [row[2] for row in self.rows]


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
        features.append(build_scale_aware_candidate_features(
            encoded["pursuer_embeddings"][:n], encoded["task_embeddings"][:m],
            positions[:n], positions[n:n + m], candidate,
            int(config["capture_size"]),
        ).detach().cpu())
        diagnostics.append(rollout_capture_label(
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
        "features": torch.stack(features).numpy().tolist(),
        "G_H": labels,
        "alive": np.repeat(alive[None, :], len(candidates), axis=0).tolist(),
        "assignments": candidates.tolist(),
        "modes": [list(mode) for mode in modes],
        "meta": {
            "sampling_reason": context["reason"],
            "sampling_priority": float(context["priority"]),
            "scenario": context["scenario"],
            "capture_labels": diagnostics,
            "state_snapshot": _jsonable(snapshot),
            "encoder_state": "omitted; use recorded features and labels",
        },
    }


def collect_episode_records(modules, config, seed, scenario="deploy5x5",
                            horizon=100, max_states=2, forced_step=None):
    env = make_environment(config, d_cap=0.8)
    if scenario == "deploy5x5":
        env.scenario_sampler = Deploy5x5Sampler()
    elif scenario != "v4_dispersed":
        raise ValueError(f"unknown scenario {scenario}")
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=int(seed))
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
                    "priority": priority, "reason": reason, "scenario": scenario,
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
            state_id=f"{seed}:{context['env'].episode_step_count}:{scenario}",
        ))
    env.close()
    return records


def collect_anchor_record(checkpoint, horizon=100, device="cpu"):
    modules, config = _load_policy(checkpoint, torch.device(device))
    return collect_episode_records(
        modules, config, ANCHOR_SEED, scenario="deploy5x5", horizon=horizon,
        max_states=1, forced_step=ANCHOR_STEP,
    )[0]


def save_dataset(records: Iterable[Mapping], output, shard="0/1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shard_index, shard_count = (int(value) for value in str(shard).split("/"))
    name = ("data.jsonl" if shard_count == 1 else
            f"shard_{shard_index:03d}_of_{shard_count:03d}.jsonl")
    path = output / name
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
    return path


def iter_dataset(path) -> Iterator[dict]:
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in files:
        with file.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


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
            modules, config, ANCHOR_SEED, "deploy5x5", args.horizon,
            max_states=1, forced_step=ANCHOR_STEP,
        ))
    episode = 0
    while len(records) < int(args.states):
        seed = int(args.seed) + shard_index * 1_000_000 + episode * shard_count
        scenario = "deploy5x5" if episode % 5 else "v4_dispersed"
        records.extend(collect_episode_records(
            modules, config, seed, scenario, args.horizon,
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
    parser.add_argument("--out", default="checkpoints/v3cf_value/dataset")
    parser.add_argument("--workers", type=int, default=1,
                        help="launch shards externally; retained for CLI compatibility")
    parser.add_argument("--shard", type=lambda text: tuple(map(int, text.split("/"))),
                        default=(0, 1), metavar="I/N")
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_hybrid_dcap08c/best_4v2.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    if args.states < 1 or args.workers < 1 or args.horizon < 1:
        raise ValueError("states, workers and horizon must be positive")
    generate_dataset(args)


if __name__ == "__main__":
    main()
