"""Train a scale-aware 4v2 value teacher and distil an online demand head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .coalition_value import generate_candidate_assignments, geometric_proxy_value
from .stage_env import StageCoalitionEnv
from .stage_networks import StageFeatureEncoder
from .v3_demand_distillation import (
    NormalizedDemandHead,
    ScaleAwareCoalitionValueHead,
    V3_DEMAND_SCHEMA,
    V3_TEACHER_SCHEMA,
    build_demand_geometry,
    build_scale_aware_candidate_features,
    capacities_from_demand,
    demand_target_from_values,
    fit_demand_student,
    fit_scale_aware_teacher,
)
from .v3_allocation import form_demand_guided_alliance


def _load_encoder(checkpoint: dict, device: torch.device) -> StageFeatureEncoder:
    hidden = int(checkpoint.get("hidden_dim", 32))
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3,
        num_obstacles=int(checkpoint.get("num_obstacles", 10)),
        map_size=float(checkpoint.get("map_size", 8.0)), hidden_dim=hidden,
        future_steps=10, seq_len=10, node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True
    encoder.st_hgat.load_state_dict(checkpoint["st_hgat"])
    encoder.relation_head.load_state_dict(checkpoint["relation_head"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.no_grad()
def collect_dataset(checkpoint: dict, scenarios: int, seed: int,
                    device: torch.device):
    encoder = _load_encoder(checkpoint, device)
    args = checkpoint.get("args", {})
    env = StageCoalitionEnv(
        length=float(checkpoint.get("map_size", 8.0)),
        num_obstacle=int(checkpoint.get("num_obstacles", 10)),
        max_pursuers=6, max_evaders=3,
        d_cap=float(checkpoint.get("d_cap", 1.5)),
        hold_steps=int(checkpoint.get("hold_steps", 5)), seed=seed,
        v_max_e=float(args.get("evader_speed", 0.2)),
    )
    candidates = np.stack(generate_candidate_assignments(4, 2))
    all_features, all_labels, all_tasks, all_geometry, all_relations = [], [], [], [], []
    for index in range(int(scenarios)):
        env.reset(num_pursuers=4, num_evaders=2,
                  scenario_seed=int(seed) + 50000 + index)
        encoder.reset(step=0)
        snapshot = env.snapshot()
        encoded = encoder.encode(snapshot, differentiable=False)
        positions = np.asarray(snapshot["positions"])
        pursuers = positions[:4, :2]
        tasks = positions[4:6, :2]
        p_emb = encoded["pursuer_embeddings"][:4]
        t_emb = encoded["task_embeddings"][:2]
        features, labels = [], []
        for assignment in candidates:
            features.append(build_scale_aware_candidate_features(
                p_emb, t_emb, pursuers, tasks, assignment, capture_size=3,
            ))
            labels.append(geometric_proxy_value(
                pursuers, tasks, assignment, capture_size=3,
            ))
        all_features.append(torch.stack(features).cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.float32))
        all_tasks.append(t_emb.cpu())
        all_geometry.append(build_demand_geometry(
            pursuers, tasks, float(checkpoint.get("map_size", 8.0)),
        ))
        all_relations.append(encoded["relation_costs"][:4, :2].cpu())
    return (torch.stack(all_features), torch.stack(all_labels),
            torch.stack(all_tasks), torch.stack(all_geometry),
            torch.stack(all_relations), candidates)


@torch.no_grad()
def teacher_metrics(model, features, labels):
    prediction = model(features.flatten(0, 1)).reshape(labels.shape)
    pair_correct = pair_total = 0
    for pred, truth in zip(prediction, labels):
        truth_delta = truth[:, None] - truth[None, :]
        pred_delta = pred[:, None] - pred[None, :]
        mask = truth_delta > 1e-8
        pair_correct += int((pred_delta[mask] > 0).sum())
        pair_total += int(mask.sum())
    chosen = prediction.argmax(1)
    best = labels.max(1).values
    top1 = (labels.gather(1, chosen[:, None]).squeeze(1) >= best - 1e-8)
    return {
        "ranking_accuracy": pair_correct / max(pair_total, 1),
        "top1_accuracy": float(top1.float().mean()),
    }, prediction


@torch.no_grad()
def demand_metrics(model, embeddings, geometry, targets, relations=None,
                   num_agents=4):
    mask = torch.ones(2, dtype=torch.bool, device=embeddings.device)
    predicted = torch.stack([
        model(row, mask, num_agents=num_agents, capture_size=3,
              task_geometry=geometry[index])
        for index, row in enumerate(embeddings)
    ])
    capacities = torch.stack([
        capacities_from_demand(row, mask, num_agents) for row in predicted
    ])
    target_side = targets.argmax(1)
    predicted_side = predicted.argmax(1)
    metrics = {
        "demand_argmax_accuracy": float((target_side == predicted_side).float().mean()),
        "mean_target_max_demand": float(targets.max(1).values.mean()),
        "mean_predicted_max_demand": float(predicted.max(1).values.mean()),
        "asymmetric_capacity_rate": float(
            (capacities.max(1).values >= 2.5).float().mean()
        ),
        "near_3plus1_rate": float(
            (capacities.max(1).values >= 2.8).float().mean()
        ),
        "mean_capacity_max": float(capacities.max(1).values.mean()),
    }
    if relations is not None:
        hard_counts = []
        for costs, capacity in zip(relations, capacities):
            assignment = form_demand_guided_alliance(costs, capacity)
            hard_counts.append(np.bincount(assignment, minlength=2))
        hard_counts = np.stack(hard_counts)
        metrics["hard_3plus1_rate"] = float(
            np.mean(np.max(hard_counts, axis=1) == 3)
        )
        metrics["hard_2plus2_rate"] = float(
            np.mean(np.max(hard_counts, axis=1) == 2)
        )
    return metrics, predicted, capacities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenarios", type=int, default=240)
    parser.add_argument("--teacher-epochs", type=int, default=250)
    parser.add_argument("--demand-epochs", type=int, default=300)
    parser.add_argument("--teacher-lr", type=float, default=3e-3)
    parser.add_argument("--demand-lr", type=float, default=3e-3)
    parser.add_argument("--distill-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    options = parser.parse_args()
    torch.manual_seed(options.seed)
    device = torch.device(options.device)
    checkpoint = torch.load(options.checkpoint, map_location=device,
                            weights_only=False)
    features, labels, task_embeddings, task_geometry, relations, candidates = collect_dataset(
        checkpoint, options.scenarios, options.seed, device,
    )
    split = int(0.8 * len(features))
    if split <= 0 or split >= len(features):
        raise ValueError("at least five scenarios are required")
    train_x, valid_x = features[:split].to(device), features[split:].to(device)
    train_y, valid_y = labels[:split].to(device), labels[split:].to(device)
    train_tasks = task_embeddings[:split].to(device)
    valid_tasks = task_embeddings[split:].to(device)
    train_geometry = task_geometry[:split].to(device)
    valid_geometry = task_geometry[split:].to(device)
    train_relations = relations[:split].to(device)
    valid_relations = relations[split:].to(device)
    hidden = int(checkpoint.get("hidden_dim", 32))

    teacher = ScaleAwareCoalitionValueHead(hidden).to(device)
    teacher_fit = fit_scale_aware_teacher(
        teacher, train_x, train_y, epochs=options.teacher_epochs,
        lr=options.teacher_lr,
    )
    train_teacher_metrics, train_values = teacher_metrics(
        teacher, train_x, train_y,
    )
    valid_teacher_metrics, valid_values = teacher_metrics(
        teacher, valid_x, valid_y,
    )
    train_targets = torch.stack([
        demand_target_from_values(row, candidates, 2,
                                  options.distill_temperature)
        for row in train_values
    ])
    valid_targets = torch.stack([
        demand_target_from_values(row, candidates, 2,
                                  options.distill_temperature)
        for row in valid_values
    ])

    demand = NormalizedDemandHead(hidden).to(device)
    demand_fit = fit_demand_student(
        demand, train_tasks, train_targets, num_agents=4,
        epochs=options.demand_epochs, lr=options.demand_lr,
        task_geometry=train_geometry,
    )
    train_demand_metrics, _, _ = demand_metrics(
        demand, train_tasks, train_geometry, train_targets, train_relations,
    )
    valid_demand_metrics, valid_demands, valid_capacities = demand_metrics(
        demand, valid_tasks, valid_geometry, valid_targets, valid_relations,
    )
    metrics = {
        "teacher_fit": teacher_fit,
        "teacher_train": train_teacher_metrics,
        "teacher_validation": valid_teacher_metrics,
        "demand_fit": demand_fit,
        "demand_train": train_demand_metrics,
        "demand_validation": valid_demand_metrics,
        "train_scenarios": split,
        "validation_scenarios": len(features) - split,
        "candidates_per_scenario": len(candidates),
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture_version": V3_DEMAND_SCHEMA,
        "teacher_schema": V3_TEACHER_SCHEMA,
        "source_checkpoint": str(Path(options.checkpoint).resolve()),
        "hidden_dim": hidden,
        "num_pursuers": 4,
        "num_evaders": 2,
        "capture_size": 3,
        "teacher": teacher.state_dict(),
        "demand_head": demand.state_dict(),
        "metrics": metrics,
        "args": vars(options),
        "validation_demands": valid_demands.cpu(),
        "validation_capacities": valid_capacities.cpu(),
    }, output)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
