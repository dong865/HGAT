"""Paired 4v2 counterfactual evaluation for a proxy coalition value head.

The proxy ranks the initial non-empty assignments.  It does not control online
switching: the chosen assignment is held while both targets remain active and
is repaired only when the active task set changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch

from .coalition_value import (
    generate_candidate_assignments,
    geometric_proxy_value,
)
from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .v2_controller import V2Controller, v2_obs_dim
from .v2_env_adapter import apply_membership
from .v3_allocation import (
    ValueRankedAllocation,
    load_value_checkpoint,
    rank_value_candidates,
)


EVAL_SCHEMA = "v3_4v2_proxy_value_counterfactual_v1"


def pad_actions_for_env(actions: np.ndarray, max_pursuers: int) -> np.ndarray:
    """Pad active-agent actions to the environment's fixed command shape."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [N,2]")
    if len(actions) > int(max_pursuers):
        raise ValueError("actions exceed environment pursuer capacity")
    if len(actions) == int(max_pursuers):
        return actions
    padding = np.zeros(
        (int(max_pursuers) - len(actions), 2), dtype=actions.dtype,
    )
    return np.vstack([actions, padding])


def choose_balanced_assignment(
    candidates: np.ndarray,
    num_tasks: int,
    values: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray:
    """Choose the most balanced candidate, optionally value-breaking ties."""
    candidates = np.asarray(candidates, dtype=np.int64)
    counts = np.stack([
        np.bincount(row, minlength=num_tasks)[:num_tasks]
        for row in candidates
    ])
    imbalance = counts.max(axis=1) - counts.min(axis=1)
    eligible = np.flatnonzero(imbalance == imbalance.min())
    if values is None:
        index = int(eligible[0])
    else:
        score = np.asarray(torch.as_tensor(values).detach().cpu())
        index = int(eligible[np.argmax(score[eligible])])
    return candidates[index].copy()


def repair_assignment_for_active_tasks(
    initial_assignment: np.ndarray,
    active_task_mask: np.ndarray,
) -> np.ndarray:
    """Preserve the candidate unless task lifecycle makes it infeasible."""
    initial = np.asarray(initial_assignment, dtype=np.int64)
    active = np.flatnonzero(np.asarray(active_task_mask, dtype=bool))
    if len(active) == 0:
        return np.full_like(initial, -1)
    if len(active) == 1:
        return np.full_like(initial, int(active[0]))
    repaired = initial.copy()
    fallback = int(active[0])
    valid = np.isin(repaired, active)
    repaired[~valid] = fallback
    return repaired


def _method_metrics(rows: List[Mapping]) -> Dict[str, float]:
    captured = np.asarray([row["captured"] for row in rows], dtype=float)
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    first_times = [row["first_capture_time"] for row in rows
                   if row["first_capture_time"] is not None]
    full_times = [row["full_capture_time"] for row in rows
                  if row["full_capture_time"] is not None]
    return {
        "episodes": len(rows),
        "single_capture_rate": float(np.mean(captured >= 1)),
        "full_capture_rate": float(np.mean(captured >= 2)),
        "mean_captured": float(captured.mean()),
        "mean_reward": float(rewards.mean()),
        "mean_first_capture_time_success_only": (
            float(np.mean(first_times)) if first_times else None
        ),
        "mean_full_capture_time_success_only": (
            float(np.mean(full_times)) if full_times else None
        ),
    }


def _paired_score(row: Mapping) -> Tuple[float, float, float]:
    first_time = row.get("first_capture_time")
    time_score = -float(first_time) if first_time is not None else -float("inf")
    return float(row["captured"]), time_score, float(row["reward"])


def summarize_records(records: Iterable[Mapping]) -> Dict:
    rows = list(records)
    methods = sorted({str(row["method"]) for row in rows})
    summary = {
        "methods": {
            method: _method_metrics([
                row for row in rows if row["method"] == method
            ])
            for method in methods
        }
    }
    by_seed: Dict[int, Dict[str, Mapping]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    paired = {"wins": 0, "ties": 0, "losses": 0, "paired_seeds": 0}
    for method_rows in by_seed.values():
        if "value_top1" not in method_rows or "balanced_2plus2" not in method_rows:
            continue
        paired["paired_seeds"] += 1
        value_score = _paired_score(method_rows["value_top1"])
        balanced_score = _paired_score(method_rows["balanced_2plus2"])
        if value_score > balanced_score:
            paired["wins"] += 1
        elif value_score < balanced_score:
            paired["losses"] += 1
        else:
            paired["ties"] += 1
    summary["paired_value_vs_balanced"] = paired
    return summary


class LockedCandidateController:
    """Execute one selected candidate with lifecycle-only repair."""

    def __init__(self, base: V2Controller):
        self.base = base
        self.initial_assignment = np.empty(0, dtype=np.int64)
        self.assignment = np.empty(0, dtype=np.int64)

    def reset(self, initial_assignment: np.ndarray) -> None:
        self.base.reset()
        self.initial_assignment = np.asarray(initial_assignment, dtype=np.int64).copy()
        self.assignment = self.initial_assignment.copy()

    def controller_step(self, env: StageCoalitionEnv) -> Tuple[np.ndarray, Dict]:
        env.advance_lifecycle()
        assignment = repair_assignment_for_active_tasks(
            self.initial_assignment, env.task_mask[:env.num_evaders],
        )
        if not np.array_equal(assignment, self.assignment):
            self.assignment = assignment
        apply_membership(env, self.assignment)
        encoded = self.base.encoder.encode(env.snapshot(), differentiable=False)
        obs, _ = self.base._build_v2_obs(encoded, self.assignment, env)
        with torch.no_grad():
            actions = self.base.actor(obs)[:env.num_pursuers].cpu().numpy()
        return actions, {"assignment": self.assignment.tolist()}


def _load_frozen_policy(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[Mapping, StageCoalitionEnv, LockedCandidateController]:
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    if checkpoint.get("architecture_version") != "continuous_membership_v2":
        raise ValueError("policy checkpoint is not continuous_membership_v2")
    hidden = int(checkpoint.get("hidden_dim", 32))
    obstacles = int(checkpoint.get("num_obstacles", 10))
    map_size = float(checkpoint.get("map_size", 8.0))
    args = checkpoint.get("args", {})
    env = StageCoalitionEnv(
        length=map_size, num_obstacle=obstacles,
        max_pursuers=6, max_evaders=3,
        d_cap=float(checkpoint.get("d_cap", 1.5)),
        hold_steps=int(checkpoint.get("hold_steps", 5)), seed=0,
        v_max_e=float(args.get("evader_speed", 0.2)),
    )
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=obstacles,
        map_size=map_size, hidden_dim=hidden, future_steps=10, seq_len=10,
        node_dim=9, device=device, use_ring_slot_features=False,
        use_stage_embedding=False, use_role_embedding=False,
        use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True
    demand_head = ContinuousDemandHead(hidden, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, hidden, 10), hidden_dim=64).to(device)
    encoder.st_hgat.load_state_dict(checkpoint["st_hgat"])
    encoder.relation_head.load_state_dict(checkpoint["relation_head"])
    demand_head.load_state_dict(checkpoint["demand_head"])
    actor.load_state_dict(checkpoint["actor"])
    encoder.eval()
    demand_head.eval()
    actor.eval()
    base = V2Controller(
        encoder, demand_head, actor, device,
        demand_source="fixed", fixed_demand=3.0,
        target_context=str(checkpoint.get("target_context", "hard")),
    )
    return checkpoint, env, LockedCandidateController(base)


@torch.no_grad()
def _rank_scene(
    env: StageCoalitionEnv,
    controller: LockedCandidateController,
    value_model: torch.nn.Module,
    candidates: np.ndarray,
    temperature: float,
) -> Tuple[ValueRankedAllocation, np.ndarray]:
    controller.base.encoder.reset(step=0)
    snapshot = env.snapshot()
    encoded = controller.base.encoder.encode(snapshot, differentiable=False)
    n_p, n_e = env.num_pursuers, env.num_evaders
    positions = np.asarray(snapshot["positions"])
    pursuers = positions[:n_p, :2]
    tasks = positions[n_p:n_p + n_e, :2]
    ranked = rank_value_candidates(
        value_model,
        encoded["pursuer_embeddings"][:n_p],
        encoded["task_embeddings"][:n_e],
        pursuers, tasks, candidates, temperature=temperature,
    )
    proxy_values = np.asarray([
        geometric_proxy_value(pursuers, tasks, assignment, capture_size=3)
        for assignment in candidates
    ])
    oracle = candidates[int(proxy_values.argmax())].copy()
    return ranked, oracle


def _rollout(
    env: StageCoalitionEnv,
    controller: LockedCandidateController,
    assignment: np.ndarray,
    scenario_seed: int,
    max_steps: int,
) -> Dict:
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=scenario_seed)
    controller.reset(assignment)
    reward = 0.0
    first_capture_time = None
    full_capture_time = None
    previous_captured = 0
    steps = 0
    done = False
    while not done and steps < max_steps:
        actions, _ = controller.controller_step(env)
        actions = pad_actions_for_env(actions, env.max_pursuers)
        step_reward, done, _ = env.step(actions, max_steps=max_steps)
        reward += float(step_reward)
        steps += 1
        captured = int(np.count_nonzero(env.captured[:2]))
        if captured >= 1 and previous_captured == 0:
            first_capture_time = steps * float(env.time_step)
        if captured == 2 and full_capture_time is None:
            full_capture_time = steps * float(env.time_step)
        previous_captured = captured
    return {
        "captured": int(np.count_nonzero(env.captured[:2])),
        "first_capture_time": first_capture_time,
        "full_capture_time": full_capture_time,
        "reward": reward,
        "steps": steps,
        "outcome": str(getattr(env, "outcome", "")),
    }


def evaluate(
    policy_checkpoint: str | Path,
    value_checkpoint: str | Path,
    episodes: int,
    seed: int,
    max_steps: int,
    temperature: float,
    device: torch.device,
) -> Dict:
    policy, env, controller = _load_frozen_policy(policy_checkpoint, device)
    hidden = int(policy.get("hidden_dim", 32))
    value_model, value_payload = load_value_checkpoint(
        value_checkpoint, device, expected_hidden_dim=hidden,
    )
    candidates = np.stack(generate_candidate_assignments(4, 2))
    records: List[Dict] = []
    scene_rows: List[Dict] = []
    for episode in range(int(episodes)):
        scenario_seed = int(seed) + 50000 + episode
        env.reset(num_pursuers=4, num_evaders=2, scenario_seed=scenario_seed)
        ranked, oracle_assignment = _rank_scene(
            env, controller, value_model, candidates, temperature,
        )
        balanced = choose_balanced_assignment(
            candidates, 2, ranked.candidate_values,
        )
        methods = {
            "value_top1": ranked.assignment,
            "balanced_2plus2": balanced,
            "proxy_oracle_top1": oracle_assignment,
        }
        scene_rows.append({
            "seed": scenario_seed,
            "value_top1_assignment": ranked.assignment.tolist(),
            "value_top1_counts": ranked.task_counts.tolist(),
            "predicted_value": ranked.predicted_value,
            "demand": ranked.demand.tolist(),
            "capacity": ranked.capacity.tolist(),
        })
        for method, assignment in methods.items():
            result = _rollout(
                env, controller, assignment, scenario_seed, max_steps,
            )
            result.update({
                "seed": scenario_seed,
                "method": method,
                "initial_assignment": assignment.tolist(),
                "initial_counts": np.bincount(
                    assignment, minlength=2
                ).tolist(),
            })
            records.append(result)
    return {
        "schema": EVAL_SCHEMA,
        "policy_checkpoint": str(Path(policy_checkpoint).resolve()),
        "value_checkpoint": str(Path(value_checkpoint).resolve()),
        "value_supervision": value_payload.get("supervision"),
        "episodes": int(episodes),
        "seed": int(seed),
        "temperature": float(temperature),
        "scenes": scene_rows,
        "records": records,
        "summary": summarize_records(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-checkpoint",
        default="checkpoints/sdgcr_v2_dcap20/best_model.pth",
    )
    parser.add_argument(
        "--value-checkpoint",
        default="checkpoints/coalition_value_proxy_v1.pth",
    )
    parser.add_argument(
        "--output", default="checkpoints/v3_4v2_value_eval/eval.json",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    result = evaluate(
        args.policy_checkpoint, args.value_checkpoint, args.episodes,
        args.seed, args.max_steps, args.temperature,
        torch.device(args.device),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
