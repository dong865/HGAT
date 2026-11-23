"""Frozen 4v2 deploy5x5/d_cap=0.8 acceptance evaluation.

``oracle`` is the retrospective best of all 16 initial 4v2 assignments on
the same seed. It is an Actor ceiling, not an online deployable policy.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Mapping

import numpy as np


SEED_RANGES = {"train": (72000, 73000), "validation": (73000, 73200),
               "test": (74000, 74200)}
MODES = {"online", "oracle", "both", "sealed_final"}


def validate_protocol(config: Mapping) -> dict:
    """Fail closed when a checkpoint would change the acceptance physics."""
    if config.get("scenario_protocol") != "deploy5x5":
        raise ValueError("scenario_protocol must be deploy5x5")
    if (config.get("d_cap") != 0.8
            or config.get("d_cap_final") not in (None, 0.8)):
        raise ValueError("d_cap must be fixed at 0.8")
    if config.get("map_size") != 8.0 or config.get("num_obstacles") != 10:
        raise ValueError("deploy5x5 requires an 8 m frame and 10 obstacle slots")
    if config.get("hold_steps") != 5:
        raise ValueError("capture hold_steps must be 5")
    if not isinstance(config.get("max_steps"), int) or config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    return {"scenario_protocol": "deploy5x5", "d_cap": 0.8,
            "map_size": 8.0, "num_obstacles": 10, "hold_steps": 5,
            "max_steps": config["max_steps"], "v_max": 0.35,
            "v_max_e": 0.35}


def _seed_entry(entry):
    if isinstance(entry, Mapping):
        seed = entry.get("seed")
        ancestry = [entry.get(key) for key in
                    ("parent_seed", "base_seed", "source_seed", "derived_from")]
    else:
        seed, ancestry = entry, []
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("scenario seed must be an integer")
    sealed = int(seed) == 70086 or any("70086" in str(value) for value in ancestry)
    return int(seed), sealed


def validate_evaluation_seeds(seeds, allocation_mode, split=None) -> list[int]:
    if allocation_mode not in MODES:
        raise ValueError(f"unknown allocation_mode {allocation_mode!r}")
    if split is not None and split not in SEED_RANGES:
        raise ValueError(f"unknown seed split {split!r}")
    if allocation_mode == "sealed_final" and split is not None:
        raise ValueError("sealed final mode cannot be a train/validation/test split")
    entries = [_seed_entry(entry) for entry in seeds]
    if not entries:
        raise ValueError("at least one seed is required")
    if any(sealed for _, sealed in entries) and allocation_mode != "sealed_final":
        raise ValueError("sealed seed 70086 or derived scene is final-only")
    values = [seed for seed, _ in entries]
    if len(values) != len(set(values)):
        raise ValueError("duplicate scenario seeds")
    if split is not None:
        lo, hi = SEED_RANGES[split]
        if any(not lo <= seed < hi for seed in values):
            raise ValueError(f"seed outside {split} range [{lo}, {hi})")
    elif allocation_mode != "sealed_final":
        if any(not any(lo <= seed < hi for lo, hi in SEED_RANGES.values())
               for seed in values):
            raise ValueError("seed outside reserved train/validation/test ranges")
    return sorted(values)


def validate_seed_splits(splits: Mapping) -> dict:
    if set(splits) != set(SEED_RANGES):
        raise ValueError("train, validation and test seed splits are required")
    result = {split: validate_evaluation_seeds(seeds, "online", split)
              for split, seeds in splits.items()}
    flattened = [seed for seeds in result.values() for seed in seeds]
    if len(flattened) != len(set(flattened)):
        raise ValueError("seed split overlap")
    return result


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_records(records) -> dict:
    rows = list(records)
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        group = [row for row in rows if row["mode"] == mode]
        result[mode] = {
            "episodes": len(group),
            "any_capture_rate": float(np.mean([row["captured"] >= 1 for row in group])),
            "full_capture_rate": float(np.mean([row["captured"] >= 2 for row in group])),
            "mean_captured": _mean(group, "captured"),
            "mean_switches": _mean(group, "switches"),
            "mean_min_max_assigned_member_distance":
                _mean(group, "min_max_assigned_member_distance"),
            "mean_min_angular_gap": _mean(group, "min_angular_gap"),
            "mean_first_capture_time_success_only": _mean(group, "first_capture_time"),
            "mean_full_capture_time_success_only": _mean(group, "full_capture_time"),
        }
    return result


def _load_checkpoint(path):
    from .train_v3_hybrid import load_v3_hybrid_checkpoint
    return load_v3_hybrid_checkpoint(Path(path), map_location="cpu")


def _load_modules(path, payload):
    from .train_v3_hybrid import build_system, load_v3_hybrid_checkpoint
    config = dict(payload["config"])
    config.update(device="cpu", init_policy_from=None, init_value_from=None,
                  init_v3_from=None)
    modules, _, _ = build_system(config)
    load_v3_hybrid_checkpoint(path, modules={key: modules[key] for key in
                              ("encoder", "demand_head", "value_head", "actor")})
    for module in modules.values():
        module.eval()
    return modules


def _geometry(env, assignment):
    from .scenario_protocol import _angular_gap
    positions = env.positions
    distances, gaps = [], []
    for task in range(env.num_evaders):
        if not env.task_mask[task]:
            continue
        members = positions[:env.num_pursuers][assignment == task, :2]
        if not len(members):
            continue
        target = positions[env.num_pursuers + task, :2]
        distances.append(float(np.linalg.norm(members - target, axis=1).max()))
        if len(members) >= 3:
            gaps.append(_angular_gap(members, target))
    return (min(distances) if distances else None,
            min(gaps) if gaps else None)


def _run_episode(modules, config, seed, mode, assignment=None):
    import torch
    from .train_v3_hybrid import HybridPolicy, make_environment
    from .v2_env_adapter import apply_membership
    from .v2_training import build_v2_obs_torch, hard_K_from_assignment

    env = make_environment(config, d_cap=0.8)
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=seed)
    if env.d_cap != 0.8 or env.v_max != 0.35 or env.v_max_e != 0.35:
        raise ValueError("environment violates frozen d_cap/speed protocol")
    policy = HybridPolicy(modules, config) if mode == "online" else None
    if policy is None:
        modules["encoder"].reset()
    first_capture_time = full_capture_time = None
    closest_max_distance = closest_angular_gap = None
    previous_assignment = None
    switches = 0
    reward_total = 0.0
    for step in range(config["max_steps"]):
        if policy is not None:
            decision = policy.decision(env)
            current = np.asarray(decision["assignment"][:4], dtype=np.int64)
            actions = decision["actions"]
        else:
            active = np.flatnonzero(env.task_mask[:2])
            if not len(active):
                break
            current = np.asarray(assignment, dtype=np.int64).copy()
            current[~np.isin(current, active)] = int(active[0])
            apply_membership(env, current)
            snapshot = env.snapshot()
            with torch.no_grad():
                encoded = modules["encoder"].encode(snapshot, differentiable=False)
                membership = hard_K_from_assignment(
                    current, env.task_mask, 6, 3,
                    encoded["pursuer_embeddings"].device,
                )
                obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
                mask = torch.as_tensor(env.pursuer_mask, dtype=torch.float32,
                                       device=obs.device)
                actions = (modules["actor"](obs) * mask.unsqueeze(-1)).cpu().numpy()
        if previous_assignment is not None and not np.array_equal(current, previous_assignment):
            switches += 1
        previous_assignment = current.copy()
        distance, gap = _geometry(env, current)
        if distance is not None:
            closest_max_distance = (distance if closest_max_distance is None
                                    else min(closest_max_distance, distance))
        if gap is not None:
            closest_angular_gap = (gap if closest_angular_gap is None
                                   else min(closest_angular_gap, gap))
        if not np.isfinite(actions).all():
            raise FloatingPointError("nonfinite Actor action")
        reward, done, _ = env.step(actions, max_steps=config["max_steps"])
        reward_total += float(reward)
        captured = int(np.count_nonzero(env.captured[:2]))
        if captured >= 1 and first_capture_time is None:
            first_capture_time = (step + 1) * env.time_step
        if captured == 2 and full_capture_time is None:
            full_capture_time = (step + 1) * env.time_step
        if done:
            break
    return {"seed": seed, "mode": mode,
            "captured": int(np.count_nonzero(env.captured[:2])),
            "first_capture_time": first_capture_time,
            "full_capture_time": full_capture_time,
            "min_max_assigned_member_distance": closest_max_distance,
            "min_angular_gap": closest_angular_gap, "switches": switches,
            "reward": reward_total, "outcome": env.outcome,
            "steps": env.episode_step_count}


def evaluate_checkpoint(path, seeds, allocation_mode) -> dict:
    """Evaluate online, all-alliance oracle, both, or sealed final mode."""
    resolved_seeds = validate_evaluation_seeds(seeds, allocation_mode)
    payload = _load_checkpoint(path)
    protocol = validate_protocol(payload["config"])
    modules = _load_modules(path, payload)
    config = dict(payload["config"])
    config["device"] = "cpu"
    modes = (("online", "oracle") if allocation_mode in ("both", "sealed_final")
             else (allocation_mode,))
    records = []
    for seed in resolved_seeds:
        if "online" in modes:
            records.append(_run_episode(modules, config, seed, "online"))
        if "oracle" in modes:
            candidates = []
            for assignment in itertools.product((0, 1), repeat=4):
                row = _run_episode(modules, config, seed, "oracle", assignment)
                row["initial_assignment"] = list(assignment)
                candidates.append(row)
            best = max(candidates, key=lambda row: (
                row["captured"],
                -row["first_capture_time"] if row["first_capture_time"] is not None else -float("inf"),
                row["reward"],
            ))
            records.append(best)
    return {"schema": "v3_08_acceptance_v1", "checkpoint": str(Path(path).resolve()),
            "protocol": protocol, "allocation_mode": allocation_mode,
            "seeds": resolved_seeds, "records": records,
            "summary": summarize_records(records)}
