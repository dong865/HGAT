"""Train V3 4v2 demand modes from paired fixed-alliance rollouts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler, PROTOCOL_VERSION
from .v2_controller import V2Controller, v2_obs_dim
from .v3_candidate_generator import generate_bounded_candidates
from .v3_counterfactual_demand import (
    fit_counterfactual_demand, mode_targets_from_rollouts,
)
from .v3_demand_distillation import build_demand_geometry
from .v3_eval_4v2_value import LockedCandidateController, _rollout
from .v3_sparse_demand import SparseDemandHead


SCHEMA = "v3_counterfactual_mode_demand_4v2_v1"
MODES_4V2 = ((2, 2), (3, 1), (1, 3), (4, 0), (0, 4))


def _load_policy(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture_version") != "v3_sparse_demand_value_hybrid_v1":
        raise ValueError("counterfactual demand training requires a V3 hybrid checkpoint")
    config = payload["config"]
    hidden = int(config["hidden_dim"])
    obstacles = int(config["num_obstacles"])
    map_size = float(config["map_size"])
    d_cap = float(config.get("d_cap_final") or config["d_cap"])
    env = StageCoalitionEnv(
        length=map_size, num_obstacle=obstacles, max_pursuers=6,
        max_evaders=3, d_cap=d_cap, hold_steps=int(config["hold_steps"]),
        seed=0,
    )
    env.scenario_sampler = Deploy5x5Sampler()
    encoder = StageFeatureEncoder(
        6, 3, obstacles, map_size, hidden, int(config["future_steps"]),
        10, 9, device, use_role_embedding=False,
        use_stage_embedding=False, use_graph_embedding=True,
        use_ring_slot_features=False,
    ).to(device)
    encoder._skip_legacy_plan = True
    encoder.st_hgat.load_state_dict(payload["st_hgat"])
    encoder.relation_head.load_state_dict(payload["relation_head"])
    actor = SharedActor(v2_obs_dim(6, hidden, int(config["future_steps"])),
                        int(config["actor_hidden"])).to(device)
    actor.load_state_dict(payload["actor"])
    fixed_demand = ContinuousDemandHead(hidden, demand_init=3.0).to(device)
    encoder.eval(); actor.eval(); fixed_demand.eval()
    base = V2Controller(
        encoder, fixed_demand, actor, device, demand_source="fixed",
        fixed_demand=3.0, target_context="hard",
    )
    head = SparseDemandHead(hidden, width=int(config["head_width"])).to(device)
    missing, unexpected = head.load_state_dict(payload["demand_head"], strict=False)
    if unexpected or any(not key.startswith("mode_net.") for key in missing):
        raise ValueError("base demand state is incompatible with mode extension")
    return payload, env, LockedCandidateController(base), head


@torch.no_grad()
def collect_scenes(payload, env, controller, seeds, max_steps, temperature,
                   good_modes=3):
    config = payload["config"]
    device = next(controller.base.actor.parameters()).device
    scenes, diagnostics = [], []
    for scenario_seed in seeds:
        env.reset(num_pursuers=4, num_evaders=2,
                  scenario_seed=int(scenario_seed))
        controller.base.encoder.reset(step=0)
        snapshot = env.snapshot()
        encoded = controller.base.encoder.encode(snapshot, differentiable=False)
        positions = np.asarray(snapshot["positions"])
        geometry = build_demand_geometry(
            positions[:4, :2], positions[4:6, :2], float(config["map_size"]),
        ).to(device)
        assignments, rollouts = [], []
        for mode in MODES_4V2:
            candidate = generate_bounded_candidates(
                encoded["relation_costs"][:4, :2], [mode],
                np.ones(2, dtype=bool), max_candidates=1,
            )[0]
            assignments.append(candidate)
            rollouts.append(_rollout(
                env, controller, candidate, int(scenario_seed), int(max_steps),
            ))
        target, scores = mode_targets_from_rollouts(
            MODES_4V2, rollouts,
            horizon_seconds=float(max_steps) * float(env.time_step),
            temperature=float(temperature),
        )
        good_count = min(max(1, int(good_modes)), len(MODES_4V2))
        good_indices = np.argsort(-scores, kind="stable")[:good_count]
        good_mask = np.zeros(len(MODES_4V2), dtype=bool)
        good_mask[good_indices] = True
        scenes.append({
            "task_embeddings": encoded["task_embeddings"][:2].detach(),
            "alive_mask": torch.ones(2, dtype=torch.bool, device=device),
            "task_geometry": geometry.detach(), "num_agents": 4,
            "capture_size": 3, "modes": MODES_4V2, "target": target,
            "good_mode_mask": good_mask,
        })
        diagnostics.append({
            "seed": int(scenario_seed), "modes": [list(x) for x in MODES_4V2],
            "assignments": [x.tolist() for x in assignments],
            "rollouts": rollouts, "scores": scores.tolist(),
            "target": target.tolist(),
            "good_modes": [list(MODES_4V2[index]) for index in good_indices],
            "best_mode": list(MODES_4V2[int(target.argmax())]),
        })
    return scenes, diagnostics


@torch.no_grad()
def evaluate_head(head, scenes, diagnostics):
    rows, correct, covered, good_recall = [], 0, 0, []
    head.eval()
    for scene, diagnostic in zip(scenes, diagnostics):
        logits, probabilities = head.score_modes(
            scene["task_embeddings"], scene["alive_mask"], 4,
            scene["modes"], 3, scene["task_geometry"],
        )
        demand = head(
            scene["task_embeddings"], scene["alive_mask"], 4, 3,
            scene["task_geometry"],
        )
        predicted = int(probabilities.argmax())
        target = int(np.argmax(scene["target"]))
        correct += predicted == target
        top_k = min(3, len(probabilities))
        top_indices = torch.topk(probabilities, top_k).indices.cpu().tolist()
        covered += target in top_indices
        good = np.flatnonzero(scene["good_mode_mask"]).tolist()
        good_recall.append(len(set(good) & set(top_indices)) / len(good))
        rows.append({
            "seed": diagnostic["seed"], "demand": demand.cpu().tolist(),
            "mode_probabilities": probabilities.cpu().tolist(),
            "predicted_mode": list(MODES_4V2[predicted]),
            "target_mode": list(MODES_4V2[target]),
            "top_modes": [list(MODES_4V2[index]) for index in top_indices],
            "good_modes": [list(MODES_4V2[index]) for index in good],
        })
    counts = {}
    for row in rows:
        key = "+".join(map(str, row["predicted_mode"]))
        counts[key] = counts.get(key, 0) + 1
    return {"accuracy": correct / max(1, len(rows)),
            "top3_coverage": covered / max(1, len(rows)),
            "mean_good_mode_recall_at3": float(np.mean(good_recall)),
            "predicted_mode_counts": counts, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_hybrid_dcap08c/best_4v2.pth")
    parser.add_argument("--output-dir",
                        default="checkpoints/v3_counterfactual_demand_4v2")
    parser.add_argument("--train-scenes", type=int, default=12)
    parser.add_argument("--valid-scenes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=70080)
    parser.add_argument("--valid-seed", type=int, default=71080)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--good-modes", type=int, default=3)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload, env, controller, head = _load_policy(args.checkpoint, device)
    train_seeds = range(args.seed, args.seed + args.train_scenes)
    valid_seeds = range(args.valid_seed, args.valid_seed + args.valid_scenes)
    train, train_diag = collect_scenes(
        payload, env, controller, train_seeds, args.max_steps, args.temperature,
        args.good_modes,
    )
    valid, valid_diag = collect_scenes(
        payload, env, controller, valid_seeds, args.max_steps, args.temperature,
        args.good_modes,
    )
    before = {"train": evaluate_head(head, train, train_diag),
              "valid": evaluate_head(head, valid, valid_diag)}
    initial_non_demand = {
        "st_hgat": copy.deepcopy(payload["st_hgat"]),
        "relation_head": copy.deepcopy(payload["relation_head"]),
        "actor": copy.deepcopy(payload["actor"]),
        "value_head": copy.deepcopy(payload["value_head"]),
    }
    history = fit_counterfactual_demand(
        head, train, epochs=args.epochs, learning_rate=args.learning_rate,
    )
    after = {"train": evaluate_head(head, train, train_diag),
             "valid": evaluate_head(head, valid, valid_diag)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": SCHEMA, "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "scenario_protocol": PROTOCOL_VERSION,
        "config": vars(args), "before": before, "after": after,
        "loss_first": history[0], "loss_last": history[-1],
        "train_counterfactuals": train_diag,
        "valid_counterfactuals": valid_diag,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8",
    )
    torch.save({
        "architecture_version": SCHEMA, "base_checkpoint": result["base_checkpoint"],
        "hidden_dim": payload["config"]["hidden_dim"],
        "head_width": payload["config"]["head_width"],
        "demand_head": head.state_dict(), "modes_4v2": MODES_4V2,
        "training": {"supervision": "paired_fixed_alliance_rollout",
                     "non_demand_networks_frozen": True},
        "frozen_reference": initial_non_demand,
    }, output / "demand_head.pth")
    print(json.dumps({
        "before": {k: v["accuracy"] for k, v in before.items()},
        "after": {k: v["accuracy"] for k, v in after.items()},
        "after_modes": {k: v["predicted_mode_counts"] for k, v in after.items()},
        "loss": [history[0], history[-1]],
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
