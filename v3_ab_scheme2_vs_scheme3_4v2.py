"""Paired 4v2 A/B test: online value ranking versus distilled demand only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .coalition_value import generate_candidate_assignments
from .v3_allocation import form_demand_guided_alliance
from .v3_demand_distillation import (
    NormalizedDemandHead,
    ScaleAwareCoalitionValueHead,
    V3_DEMAND_SCHEMA,
    build_demand_geometry,
    build_scale_aware_candidate_features,
    capacities_from_demand,
)
from .v3_eval_4v2_value import _load_frozen_policy, _rollout


SCHEMA = "v3_ab_scheme2_online_value_vs_scheme3_distilled_demand_v1"


def _score(row):
    first = row.get("first_capture_time")
    return (
        int(row["captured"]),
        -float(first) if first is not None else -float("inf"),
        float(row["reward"]),
    )


def compare_paired_records(records):
    grouped = {}
    for row in records:
        grouped.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    result = {"scheme2_wins": 0, "scheme3_wins": 0,
              "ties": 0, "paired_seeds": 0}
    for rows in grouped.values():
        if "scheme2" not in rows or "scheme3" not in rows:
            continue
        result["paired_seeds"] += 1
        left, right = _score(rows["scheme2"]), _score(rows["scheme3"])
        if left > right:
            result["scheme2_wins"] += 1
        elif right > left:
            result["scheme3_wins"] += 1
        else:
            result["ties"] += 1
    return result


def _aggregate(records, method):
    rows = [row for row in records if row["method"] == method]
    captured = np.asarray([row["captured"] for row in rows])
    first = [row["first_capture_time"] for row in rows
             if row["first_capture_time"] is not None]
    return {
        "episodes": len(rows),
        "single_capture_rate": float(np.mean(captured >= 1)),
        "full_capture_rate": float(np.mean(captured >= 2)),
        "mean_captured": float(captured.mean()),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_first_capture_time_success_only": (
            float(np.mean(first)) if first else None
        ),
        "three_plus_one_rate": float(np.mean([
            max(row["initial_counts"]) == 3 for row in rows
        ])),
    }


@torch.no_grad()
def evaluate(policy_path, distilled_path, episodes, seed, max_steps, device):
    policy, env, controller = _load_frozen_policy(policy_path, device)
    distilled = torch.load(distilled_path, map_location=device,
                           weights_only=False)
    if distilled.get("architecture_version") != V3_DEMAND_SCHEMA:
        raise ValueError("distilled checkpoint is not normalized_demand_student_v3")
    hidden = int(policy.get("hidden_dim", 32))
    if int(distilled.get("hidden_dim", -1)) != hidden:
        raise ValueError("policy and distilled hidden dimensions differ")
    teacher = ScaleAwareCoalitionValueHead(hidden).to(device)
    student = NormalizedDemandHead(hidden).to(device)
    teacher.load_state_dict(distilled["teacher"])
    student.load_state_dict(distilled["demand_head"])
    teacher.eval(); student.eval()
    candidates = np.stack(generate_candidate_assignments(4, 2))
    records, decisions = [], []
    for episode in range(int(episodes)):
        scenario_seed = int(seed) + 50000 + episode
        env.reset(num_pursuers=4, num_evaders=2, scenario_seed=scenario_seed)
        controller.base.encoder.reset(step=0)
        snapshot = env.snapshot()
        encoded = controller.base.encoder.encode(snapshot, differentiable=False)
        positions = np.asarray(snapshot["positions"])
        pursuers, tasks = positions[:4, :2], positions[4:6, :2]
        p_emb = encoded["pursuer_embeddings"][:4]
        t_emb = encoded["task_embeddings"][:2]
        candidate_features = torch.stack([
            build_scale_aware_candidate_features(
                p_emb, t_emb, pursuers, tasks, assignment, capture_size=3,
            ) for assignment in candidates
        ])
        teacher_values = teacher(candidate_features)
        scheme2 = candidates[int(teacher_values.argmax())].copy()
        geometry = build_demand_geometry(
            pursuers, tasks, float(policy.get("map_size", 8.0)),
        ).to(device)
        mask = torch.ones(2, dtype=torch.bool, device=device)
        demand = student(t_emb, mask, num_agents=4, capture_size=3,
                         task_geometry=geometry)
        capacity = capacities_from_demand(demand, mask, num_agents=4)
        scheme3 = form_demand_guided_alliance(
            encoded["relation_costs"][:4, :2], capacity,
        )
        decisions.append({
            "seed": scenario_seed,
            "teacher_assignment": scheme2.tolist(),
            "teacher_counts": np.bincount(scheme2, minlength=2).tolist(),
            "demand": demand.cpu().tolist(),
            "capacity": capacity.cpu().tolist(),
            "student_assignment": scheme3.tolist(),
            "student_counts": np.bincount(scheme3, minlength=2).tolist(),
            "same_assignment": bool(np.array_equal(scheme2, scheme3)),
        })
        for method, assignment in (("scheme2", scheme2), ("scheme3", scheme3)):
            row = _rollout(env, controller, assignment, scenario_seed, max_steps)
            row.update({
                "seed": scenario_seed, "method": method,
                "initial_assignment": assignment.tolist(),
                "initial_counts": np.bincount(assignment, minlength=2).tolist(),
            })
            records.append(row)
    summary = {
        "scheme2": _aggregate(records, "scheme2"),
        "scheme3": _aggregate(records, "scheme3"),
        "paired": compare_paired_records(records),
        "same_assignment_rate": float(np.mean([
            row["same_assignment"] for row in decisions
        ])),
    }
    return {"schema": SCHEMA, "episodes": int(episodes), "seed": int(seed),
            "decisions": decisions, "records": records, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint",
                        default="checkpoints/sdgcr_v2_dcap20/best_model.pth")
    parser.add_argument("--distilled-checkpoint",
                        default="checkpoints/v3_4v2_demand/distilled_v1.pth")
    parser.add_argument("--output",
                        default="checkpoints/v3_ab_scheme2_vs_scheme3/eval.json")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    result = evaluate(args.policy_checkpoint, args.distilled_checkpoint,
                      args.episodes, args.seed, args.max_steps,
                      torch.device(args.device))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
