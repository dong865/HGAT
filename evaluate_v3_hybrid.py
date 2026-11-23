"""Paired V3/V2 evaluation for the sparse-demand hybrid controller.

Records one fixed-seed episode at a time for the V3 hybrid allocator and the
frozen V2 continuous-membership baseline, then aggregates pairs with the
lexicographic capture hierarchy: full capture > any capture > mean captured >
earlier first capture (None is worst) > raw environment return.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv, StageRewardConfig
from .stage_networks import SharedActor, StageFeatureEncoder
from .train_v3_hybrid import HybridPolicy, build_system, load_v3_hybrid_checkpoint
from .v2_controller import V2Controller, v2_obs_dim


SCALES = ("4v2", "6v2", "6v3")
V2_SCHEMA = "continuous_membership_v2"
EVAL_SEED_BASE = 2_000_000_000


def parse_scale(text) -> tuple[int, int]:
    """Parse '4v2'-style scale names into (pursuers, evaders)."""
    text = str(text).strip().lower()
    for separator in ("v", "x", ":"):
        if separator in text:
            left, right = text.split(separator, 1)
            try:
                scale = (int(left), int(right))
            except ValueError:
                break
            if scale not in {(4, 2), (6, 2), (6, 3)}:
                raise ValueError(f"unsupported scale {text!r}; expected one of {SCALES}")
            return scale
    raise ValueError(f"unsupported scale {text!r}; expected one of {SCALES}")


def parse_seeds(text, scales, base=EVAL_SEED_BASE) -> dict[tuple[int, int], list[int]]:
    """Resolve ``--seeds`` as a comma list or a per-scale count.

    A comma-separated list applies to every scale.  A single non-negative
    integer ``count`` derives deterministic per-scale seeds
    ``base + scale_index * 1_000_000 + j`` with the same construction as the
    trainer's held-out evaluation seeds, so evaluator seeds stay far away from
    any training-seed interval.
    """
    scales = [parse_scale(scale) if isinstance(scale, str) else tuple(scale) for scale in scales]
    text = str(text).strip()
    if "," in text:
        values = [int(item) for item in text.split(",") if item.strip()]
        if not values or any(value < 0 for value in values):
            raise ValueError("seed lists must contain nonnegative integers")
        return {scale: list(values) for scale in scales}
    try:
        count = int(text)
    except ValueError:
        raise ValueError(f"cannot parse seeds {text!r}")
    if count <= 0 or base < 0:
        raise ValueError("seed counts must be positive and seed base nonnegative")
    return {scale: [int(base) + i * 1_000_000 + j for j in range(count)]
            for i, scale in enumerate(scales)}


def selection_key(record) -> tuple:
    """Lexicographic hierarchy: full > any > mean captured > first capture > raw."""
    first = record.get("first_capture_time")
    return (float(record["full_capture"]), float(record["any_capture"]),
            float(record["mean_captured"]),
            -float(first) if first is not None else -math.inf,
            float(record["raw_return"]))


def _index_records(records, label):
    grouped = {}
    for record in records:
        key = (str(record["scale"]), int(record["seed"]))
        if key in grouped:
            raise ValueError(f"duplicate {label} record for (scale, seed) {key}")
        grouped[key] = record
    return grouped


def paired_summary(baseline_records, v3_records) -> dict:
    """Pair records by (scale, seed) and count deterministic per-scale wins/ties/losses."""
    baseline = _index_records(baseline_records, "baseline")
    v3 = _index_records(v3_records, "v3")
    if set(baseline) != set(v3):
        raise ValueError(
            "paired records must cover identical (scale, seed) keys; "
            f"missing baseline={sorted(set(v3) - set(baseline))}, "
            f"missing v3={sorted(set(baseline) - set(v3))}")
    per_scale = {}
    totals = {"episodes": 0, "wins": 0, "ties": 0, "losses": 0}
    pairs = []
    for key in sorted(baseline):
        scale, _ = key
        v3_key = selection_key(v3[key])
        baseline_key = selection_key(baseline[key])
        result = "win" if v3_key > baseline_key else ("loss" if v3_key < baseline_key else "tie")
        row = per_scale.setdefault(scale, {"episodes": 0, "wins": 0, "ties": 0, "losses": 0})
        row["episodes"] += 1
        row[{"win": "wins", "tie": "ties", "loss": "losses"}[result]] += 1
        totals["episodes"] += 1
        totals[{"win": "wins", "tie": "ties", "loss": "losses"}[result]] += 1
        pairs.append({"scale": key[0], "seed": key[1], "result": result})
    return {"per_scale": {scale: per_scale[scale] for scale in sorted(per_scale)},
            "pairs": pairs, **totals}


_DEFAULT_EVAL_CONFIG = {
    "map_size": 8.0, "assign_interval": 10, "max_steps": 1000,
    "max_capacity_modes": 6, "max_alliance_candidates": 32,
    "demand_weight": 0.1, "switch_weight": 0.5,
    "gate_config": {"min_dwell_steps": 20, "improvement_margin": 0.1},
}

_SHAPING_COMPONENTS = ("closure", "angular_closure", "ring")


def _eval_config(allocator_modules):
    config = dict(_DEFAULT_EVAL_CONFIG)
    config.update(allocator_modules.get("config", {}))
    return config


def _checked_load(module, state, name):
    expected = module.state_dict()
    if (not isinstance(state, dict) or set(expected) != set(state)
            or any(not isinstance(value, torch.Tensor)
                   or expected[key].shape != value.shape for key, value in state.items())):
        raise ValueError(f"incompatible {name} weights")
    module.load_state_dict(state)


def load_baseline_checkpoint(path, device="cpu", sinkhorn_iters=None) -> dict:
    """Strictly load a frozen V2 ``continuous_membership_v2`` baseline checkpoint."""
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture_version") != V2_SCHEMA:
        raise ValueError(f"baseline checkpoint schema must be {V2_SCHEMA}")
    for key in ("st_hgat", "relation_head", "demand_head", "actor"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"baseline checkpoint is missing {key}")
    args = payload.get("args") or {}
    hidden = int(payload.get("hidden_dim", 32))
    future_steps = int(payload.get("future_steps", 10))
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3,
        num_obstacles=int(payload.get("num_obstacles", 10)),
        map_size=float(payload.get("map_size", 8.0)), hidden_dim=hidden,
        future_steps=future_steps, seq_len=int(payload.get("seq_len", 10)),
        node_dim=9, device=device, use_ring_slot_features=False,
        use_stage_embedding=False, use_role_embedding=False,
        use_graph_embedding=True).to(device)
    encoder._skip_legacy_plan = True
    demand_head = ContinuousDemandHead(
        hidden, demand_init=float(args.get("demand_init", 3.0))).to(device)
    actor = SharedActor(v2_obs_dim(6, hidden, future_steps),
                        int(args.get("actor_hidden", 64))).to(device)
    _checked_load(encoder.st_hgat, payload["st_hgat"], "st_hgat")
    _checked_load(encoder.relation_head, payload["relation_head"], "relation_head")
    _checked_load(demand_head, payload["demand_head"], "demand_head")
    _checked_load(actor, payload["actor"], "actor")
    encoder.eval(), demand_head.eval(), actor.eval()
    return {
        "architecture_version": V2_SCHEMA, "path": str(path), "device": device,
        "encoder": encoder, "demand_head": demand_head, "actor": actor,
        "hidden_dim": hidden, "future_steps": future_steps,
        "map_size": float(payload.get("map_size", 8.0)),
        "num_obstacles": int(payload.get("num_obstacles", 10)),
        "d_cap": float(payload.get("d_cap", 1.5)),
        "hold_steps": int(payload.get("hold_steps", 5)),
        "v_max_e": (float(args["evader_speed"]) if args.get("evader_speed") is not None
                    else None),
        "demand_source": str(args.get("demand_source", "fixed")),
        "fixed_demand": float(args.get("fixed_demand", 3.0)),
        "target_context": str(payload.get(
            "target_context", args.get("target_context", "hard"))),
        "epsilon_ot": float(args.get("epsilon_ot", 0.5)),
        "sinkhorn_iters": int(sinkhorn_iters if sinkhorn_iters is not None
                              else args.get("sinkhorn_iters", 20)),
        "count_penalty": float(args.get("count_penalty", 1.0)),
        "switch_penalty": float(args.get("switch_penalty", 0.5)),
        "hysteresis": float(args.get("hysteresis", 0.05)),
        "min_dwell_steps": int(args.get("min_dwell_steps", 5)),
        "assign_interval": int(args.get("assign_interval", 5)),
    }


def build_evaluation_env(config, v_max_e=None, seed=0) -> StageCoalitionEnv:
    """Build an eval environment; pairs share every physical parameter."""
    kwargs = dict(length=float(config["map_size"]),
                  num_obstacle=int(config["num_obstacles"]),
                  max_pursuers=6, max_evaders=3,
                  d_cap=float(config["d_cap"]), hold_steps=int(config["hold_steps"]),
                  seed=int(seed),
                  reward_config=StageRewardConfig(**config["reward_config"]))
    if v_max_e is not None:
        kwargs["v_max_e"] = float(v_max_e)
    return StageCoalitionEnv(**kwargs)


def run_episode_eval(env, allocator_modules, seed, scale,
                     updates_disabled=True) -> dict:
    """Evaluate the V3 hybrid allocator on one fixed-seed episode.

    ``allocator_modules`` carries the system modules plus a ``"config"`` entry
    with the allocation/eval knobs.  Evaluation is inference-only; the
    ``updates_disabled`` flag is part of the contract and recorded so callers
    must opt out explicitly.
    """
    config = _eval_config(allocator_modules)
    modules = {name: module for name, module in allocator_modules.items()
               if name != "config"}
    missing = {"encoder", "demand_head", "value_head", "actor"} - set(modules)
    if missing:
        raise ValueError(f"allocator modules missing {sorted(missing)}")
    scale = parse_scale(scale) if isinstance(scale, str) else tuple(scale)
    num_agents, num_tasks = int(scale[0]), int(scale[1])
    for module in modules.values():
        module.eval()
    env.reset(num_pursuers=num_agents, num_evaders=num_tasks,
              scenario_seed=int(seed))
    policy = HybridPolicy(modules, config)
    max_steps = int(config["max_steps"])

    latency, candidates, zeros = [], [], []
    switches = zero_demand_calls = allocation_calls = dead_member_violations = 0
    max_dead_capacity = max_capacity_sum_error = 0.0
    initial_capacity = final_capacity = None
    first_capture = None
    raw_return = shaped_return = 0.0
    closure_return = angular_return = ring_return = 0.0
    steps = 0
    decision = policy.decision(env)
    for step in range(max_steps):
        active = np.asarray(env.task_mask[:num_tasks], dtype=bool)
        assigned_tasks = [int(task) for task in decision["assignment"] if int(task) >= 0]
        if any(not active[task] for task in assigned_tasks):
            dead_member_violations += 1
        actions = decision["actions"]
        reward, done, info = env.step(actions, max_steps=max_steps)
        if not np.isfinite(actions).all() or not np.isfinite(reward):
            raise FloatingPointError("nonfinite policy action or environment reward")
        steps += 1
        if first_capture is None and info["capture_events"]:
            first_capture = env.episode_step_count * env.time_step
        components = info.get("reward_components", {})
        closure, angular, ring = (float(components.get(name, 0.0))
                                  for name in _SHAPING_COMPONENTS)
        closure_return += closure
        angular_return += angular
        ring_return += ring
        shaping = closure + angular + ring
        shaped_return += shaping
        raw_return += reward - shaping
        allocation = decision["allocation"]
        capacity = allocation["capacity"].detach().cpu().numpy().astype(float)
        if initial_capacity is None:
            initial_capacity = capacity.tolist()
        final_capacity = capacity.tolist()
        max_capacity_sum_error = max(
            max_capacity_sum_error, abs(float(capacity.sum()) - num_agents))
        latency.append(float(decision["inference_ms"]))
        if decision["allocated"]:
            allocation_calls += 1
            candidates.append(int(allocation["candidate_count"]))
            demand = allocation["demand"].detach().cpu().numpy()
            live_zero_fraction = (float((demand[active] == 0).mean())
                                  if active.any() else 0.0)
            zeros.append(live_zero_fraction)
            zero_demand_calls += int(live_zero_fraction > 0.0)
            if (~active).any():
                max_dead_capacity = max(max_dead_capacity,
                                        float(capacity[~active].max()))
            switches += int(allocation["switched"] and step > 0)
        if done:
            break
        decision = policy.decision(env)
    captured = int(env.captured[:num_tasks].sum())
    return {
        "scale": f"{num_agents}v{num_tasks}", "seed": int(seed), "steps": steps,
        "num_tasks": num_tasks, "captured": captured,
        "full_capture": float(captured == num_tasks),
        "any_capture": float(captured > 0),
        "mean_captured": float(captured),
        "first_capture_time": first_capture,
        "raw_return": float(raw_return), "shaped_return": float(shaped_return),
        "environment_return": float(raw_return + shaped_return),
        "closure_return": float(closure_return),
        "angular_closure_return": float(angular_return),
        "ring_return": float(ring_return),
        "outcome": env.outcome,
        "initial_capacity": initial_capacity, "final_capacity": final_capacity,
        "demand_zero_fraction": float(np.mean(zeros)) if zeros else 0.0,
        "zero_demand_calls": zero_demand_calls,
        "allocation_calls": allocation_calls,
        "switch_count": switches,
        "switch_rate": switches / max(1, allocation_calls - 1),
        "mean_inference_ms": float(np.mean(latency)) if latency else 0.0,
        "max_inference_ms": float(np.max(latency)) if latency else 0.0,
        "mean_candidate_count": float(np.mean(candidates)) if candidates else 0.0,
        "max_candidate_count": int(max(candidates)) if candidates else 0,
        "max_dead_capacity": float(max_dead_capacity),
        "dead_member_violations": dead_member_violations,
        "max_capacity_sum_error": float(max_capacity_sum_error),
        "updates_disabled": bool(updates_disabled),
        "controller": "v3_sparse_demand_hybrid",
    }


def run_episode_baseline(env, baseline, seed, scale, max_steps=None) -> dict:
    """Evaluate the frozen V2 controller on one fixed-seed paired episode."""
    scale = parse_scale(scale) if isinstance(scale, str) else tuple(scale)
    num_agents, num_tasks = int(scale[0]), int(scale[1])
    max_steps = int(max_steps or 1000)
    env.reset(num_pursuers=num_agents, num_evaders=num_tasks,
              scenario_seed=int(seed))
    controller = V2Controller(
        baseline["encoder"], baseline["demand_head"], baseline["actor"],
        baseline["device"], max_pursuers=6, max_evaders=3,
        epsilon_ot=baseline["epsilon_ot"],
        sinkhorn_iters=baseline["sinkhorn_iters"],
        count_penalty=baseline["count_penalty"],
        switch_penalty=baseline["switch_penalty"],
        hysteresis=baseline["hysteresis"],
        min_dwell_steps=baseline["min_dwell_steps"],
        assign_interval=baseline["assign_interval"],
        demand_source=baseline["demand_source"],
        fixed_demand=baseline["fixed_demand"],
        target_context=baseline["target_context"])
    controller.reset()
    total = raw_return = shaped_return = 0.0
    closure_return = angular_return = ring_return = 0.0
    steps = switches = 0
    first_capture = None
    previous = None
    done = False
    while not done and steps < max_steps:
        actions, info = controller.controller_step(env)
        # controller_step returns only the active pursuers' commands; the
        # simulator expects a padded (max_pursuers, 2) array.
        actions = np.asarray(actions, dtype=np.float64)
        padded = np.zeros((controller.max_pursuers, actions.shape[1]),
                          dtype=np.float64)
        padded[:actions.shape[0]] = actions
        actions = padded
        assignment = np.asarray(info["assignment"], dtype=np.int64)
        if previous is not None and not np.array_equal(assignment, previous):
            switches += 1
        previous = assignment.copy()
        reward, done, step_info = env.step(actions, max_steps=max_steps)
        if not np.isfinite(actions).all() or not np.isfinite(reward):
            raise FloatingPointError("nonfinite baseline action or environment reward")
        steps += 1
        total += reward
        components = step_info.get("reward_components", {})
        closure, angular, ring = (float(components.get(name, 0.0))
                                  for name in _SHAPING_COMPONENTS)
        closure_return += closure
        angular_return += angular
        ring_return += ring
        shaping = closure + angular + ring
        shaped_return += shaping
        raw_return += reward - shaping
        if first_capture is None and step_info["capture_events"]:
            first_capture = env.episode_step_count * env.time_step
    captured = int(env.captured[:num_tasks].sum())
    return {
        "scale": f"{num_agents}v{num_tasks}", "seed": int(seed), "steps": steps,
        "num_tasks": num_tasks, "captured": captured,
        "full_capture": float(captured == num_tasks),
        "any_capture": float(captured > 0),
        "mean_captured": float(captured),
        "first_capture_time": first_capture,
        "raw_return": float(raw_return), "shaped_return": float(shaped_return),
        "environment_return": float(total),
        "closure_return": float(closure_return),
        "angular_closure_return": float(angular_return),
        "ring_return": float(ring_return),
        "outcome": env.outcome, "switch_count": switches,
        "updates_disabled": True, "controller": "v2_continuous_membership",
    }


def run_dry_run(allocator_modules, seed_plan, max_steps=None,
                updates_disabled=True, v_max_e=None, env_seed=0) -> dict:
    """Run one deterministic episode per scale and assert pipeline invariants."""
    config = _eval_config(allocator_modules)
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max steps must be positive")
        config["max_steps"] = int(max_steps)
    plan = {(parse_scale(scale) if isinstance(scale, str) else tuple(scale)):
            [int(seed) for seed in seeds] for scale, seeds in seed_plan.items()}
    if not plan:
        raise ValueError("dry run requires at least one scale")
    records = []
    max_candidate_count = 0
    max_capacity_sum_error = 0.0
    max_dead_capacity = 0.0
    dead_member_violations = 0
    for scale in plan:
        env = build_evaluation_env(config, v_max_e=v_max_e, seed=env_seed)
        try:
            for seed in plan[scale]:
                record = run_episode_eval(env, allocator_modules, seed, scale,
                                          updates_disabled=updates_disabled)
                records.append(record)
                max_candidate_count = max(max_candidate_count,
                                          record["max_candidate_count"])
                max_capacity_sum_error = max(max_capacity_sum_error,
                                             record["max_capacity_sum_error"])
                max_dead_capacity = max(max_dead_capacity, record["max_dead_capacity"])
                dead_member_violations += record["dead_member_violations"]
        finally:
            env.close()
    checks = {
        "episodes": len(records),
        # run_episode_eval raises FloatingPointError on nonfinite values.
        "finite_actions_rewards": True,
        "candidate_limit": int(config["max_alliance_candidates"]),
        "max_candidate_count": int(max_candidate_count),
        "candidates_within_limit": (max_candidate_count
                                    <= int(config["max_alliance_candidates"])),
        "max_capacity_sum_error": float(max_capacity_sum_error),
        "capacity_sum_ok": max_capacity_sum_error <= 1e-6,
        "max_dead_capacity": float(max_dead_capacity),
        "dead_member_violations": int(dead_member_violations),
        "dead_tasks_zero_members": (max_dead_capacity == 0.0
                                    and dead_member_violations == 0),
    }
    return {"kind": "v3_dry_run", "updates_disabled": bool(updates_disabled),
            "max_steps": int(config["max_steps"]),
            "scales": [f"{n}v{m}" for n, m in plan],
            "episodes": records, "checks": checks}


def load_v3_system(path, device="cpu"):
    """Strictly load a trained V3 checkpoint into a freshly built system."""
    payload = load_v3_hybrid_checkpoint(path)
    config = dict(payload["config"])
    config["init_policy_from"] = None
    config["init_value_from"] = None
    config["init_v3_from"] = None
    config["device"] = str(torch.device(device))
    modules, _, _ = build_system(config)
    load_v3_hybrid_checkpoint(path, modules)
    return modules, config


def build_v3_system_from_warm_start(path, device="cpu", seed=0):
    """Build fresh V3 modules warm-started from a compatible V2 checkpoint."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture_version") != V2_SCHEMA:
        raise ValueError(f"warm start requires a {V2_SCHEMA} checkpoint")
    args = payload.get("args") or {}
    actor_state = payload.get("actor") or {}
    first_layer = actor_state.get("net.0.weight")
    config = {
        "torch_threads": 1, "seed": int(seed), "device": str(torch.device(device)),
        "hidden_dim": int(payload.get("hidden_dim", 32)),
        "actor_hidden": (int(first_layer.shape[0])
                         if isinstance(first_layer, torch.Tensor)
                         else int(args.get("actor_hidden", 64))),
        "future_steps": int(payload.get("future_steps", 10)),
        "num_obstacles": int(payload.get("num_obstacles", 10)),
        "map_size": float(payload.get("map_size", 8.0)),
        "d_cap": float(payload.get("d_cap", 1.5)),
        "hold_steps": int(payload.get("hold_steps", 5)),
        "head_width": 64,
        "actor_lr": 3e-4, "critic_lr": 3e-4, "demand_lr": 3e-4, "value_lr": 3e-4,
        "init_policy_from": str(path), "init_value_from": None,
        "gate_config": {"min_dwell_steps": 20, "improvement_margin": 0.1},
        "assign_interval": 10, "max_steps": 1000,
        "max_capacity_modes": 6, "max_alliance_candidates": 32,
        "demand_weight": 0.1, "switch_weight": 0.5, "capture_size": 3,
        "reward_config": StageRewardConfig().to_dict(),
        "scale_weights": ((4, 2, 0.4), (6, 2, 0.3), (6, 3, 0.3)),
    }
    modules, _, metadata = build_system(config)
    if metadata["policy_warm_start"]["schema"] != V2_SCHEMA:
        raise ValueError("policy warm start did not come from a V2 checkpoint")
    return modules, config


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="trained V3 checkpoint (strict loader)")
    parser.add_argument("--warm-start-policy",
                        help="V2 checkpoint used to build fresh V3 modules (pre-training dry-run)")
    parser.add_argument("--baseline-checkpoint",
                        help="V2 continuous_membership_v2 checkpoint for the paired baseline")
    parser.add_argument("--scales", default="4v2,6v2,6v3")
    parser.add_argument("--seeds", default="1",
                        help="comma-separated seed list or a per-scale count")
    parser.add_argument("--eval-seed-base", type=int, default=EVAL_SEED_BASE)
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sinkhorn-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="environment constructor seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="one episode per scale with updates disabled and invariant checks")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(int(args.seed))
    scales = [parse_scale(token) for token in str(args.scales).split(",") if token.strip()]
    if not scales:
        raise ValueError("at least one scale is required")
    if args.checkpoint:
        modules, config = load_v3_system(args.checkpoint, device=args.device)
        source = {"kind": "v3_checkpoint", "path": str(Path(args.checkpoint).resolve())}
    elif args.warm_start_policy:
        modules, config = build_v3_system_from_warm_start(
            args.warm_start_policy, device=args.device, seed=args.seed)
        source = {"kind": "v2_warm_start", "path": str(Path(args.warm_start_policy).resolve())}
    else:
        raise SystemExit("one of --checkpoint or --warm-start-policy is required")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("max steps must be positive")
        config["max_steps"] = int(args.max_steps)

    baseline = None
    if args.baseline_checkpoint:
        baseline = load_baseline_checkpoint(args.baseline_checkpoint, device=args.device,
                                            sinkhorn_iters=args.sinkhorn_iters)
    v_max_e = baseline["v_max_e"] if baseline is not None else None
    plan = parse_seeds(args.seeds, scales, base=args.eval_seed_base)

    if args.dry_run:
        payload = run_dry_run({**modules, "config": config}, plan,
                              v_max_e=v_max_e, env_seed=args.seed)
        payload["source"] = source
        if baseline is not None:
            payload["baseline"] = {
                "path": str(Path(args.baseline_checkpoint).resolve()),
                "architecture_version": V2_SCHEMA}
        _write_json(payload, args.out)
        print(json.dumps(payload["checks"], indent=2, allow_nan=False))
        return payload

    allocator_modules = {**modules, "config": config}
    v3_records, baseline_records = [], []
    for scale in scales:
        env = build_evaluation_env(config, v_max_e=v_max_e, seed=args.seed)
        try:
            for seed in plan[scale]:
                v3_records.append(run_episode_eval(env, allocator_modules, seed, scale))
        finally:
            env.close()
        if baseline is not None:
            env = build_evaluation_env(config, v_max_e=v_max_e, seed=args.seed)
            try:
                for seed in plan[scale]:
                    baseline_records.append(run_episode_baseline(
                        env, baseline, seed, scale, max_steps=config["max_steps"]))
            finally:
                env.close()
    result = {"source": source,
              "baseline": ({"path": str(Path(args.baseline_checkpoint).resolve()),
                            "architecture_version": V2_SCHEMA}
                           if baseline is not None else None),
              "scales": [f"{n}v{m}" for n, m in scales],
              "seed_plan": {f"{n}v{m}": list(seeds) for (n, m), seeds in plan.items()},
              "max_steps": int(config["max_steps"]),
              "v3_records": v3_records, "baseline_records": baseline_records}
    if baseline is not None:
        result["paired_summary"] = paired_summary(baseline_records, v3_records)
        print(json.dumps(result["paired_summary"], indent=2, allow_nan=False))
    else:
        print(json.dumps({"episodes": len(v3_records)}, indent=2, allow_nan=False))
    _write_json(result, args.out)
    return result


if __name__ == "__main__":
    main()
