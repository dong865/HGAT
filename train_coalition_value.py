"""Quick standalone training for the geometric coalition value head.

Example:
  python -m crazyflie_examples.six_v_two_hgat.train_coalition_value \
    --checkpoint checkpoints/sdgcr_v2_dcap20/best_model.pth \
    --output checkpoints/coalition_value_proxy_v1.pth

Only the new value head is optimized.  ST-HGAT and the relation head are
loaded from the supplied checkpoint, placed in eval mode, and never receive
gradients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .coalition_value import (
    VALUE_SCHEMA,
    CoalitionValueHead,
    build_candidate_features,
    generate_candidate_assignments,
    geometric_proxy_value,
    pairwise_ranking_loss,
)
from .stage_env import StageCoalitionEnv
from .stage_networks import StageFeatureEncoder


@torch.no_grad()
def ranking_accuracy(model: CoalitionValueHead, features: torch.Tensor,
                     labels: torch.Tensor) -> float:
    """Fraction of correctly ordered unequal-label candidate pairs."""
    model.eval()
    predictions = model(features.flatten(0, 1)).reshape(labels.shape)
    correct = 0
    total = 0
    for pred, target in zip(predictions, labels):
        target_delta = target[:, None] - target[None, :]
        pred_delta = pred[:, None] - pred[None, :]
        mask = target_delta > 1e-8
        correct += int((pred_delta[mask] > 0).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


@torch.no_grad()
def top1_accuracy(model: CoalitionValueHead, features: torch.Tensor,
                  labels: torch.Tensor) -> float:
    model.eval()
    predictions = model(features.flatten(0, 1)).reshape(labels.shape)
    chosen = predictions.argmax(dim=1)
    best = labels.max(dim=1).values
    return float((labels.gather(1, chosen[:, None]).squeeze(1) >=
                  best - 1e-8).float().mean())


def fit_value_head(model: CoalitionValueHead, features: torch.Tensor,
                   labels: torch.Tensor, epochs: int = 200,
                   lr: float = 3e-3, batch_scenarios: int = 16,
                   rank_weight: float = 1.0,
                   seed: int = 0) -> Dict[str, float]:
    """Fit only ``model`` on grouped scenario/candidate tensors."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    label_mean = labels.mean()
    label_std = labels.std().clamp_min(1e-6)
    normalized = (labels - label_mean) / label_std
    n_scenarios = int(features.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    last_loss = float("nan")
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(n_scenarios, generator=generator)
        for start in range(0, n_scenarios, int(batch_scenarios)):
            ids = order[start:start + int(batch_scenarios)].to(features.device)
            batch_x = features[ids]
            batch_y = normalized[ids]
            pred = model(batch_x.flatten(0, 1)).reshape(batch_y.shape)
            regression = F.smooth_l1_loss(pred, batch_y)
            ranking = torch.stack([
                pairwise_ranking_loss(p, y) for p, y in zip(pred, batch_y)
            ]).mean()
            loss = regression + float(rank_weight) * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
    return {
        "loss": last_loss,
        "label_mean": float(label_mean),
        "label_std": float(label_std),
    }


def _load_frozen_encoder(checkpoint: Dict, device: torch.device
                         ) -> StageFeatureEncoder:
    hidden = int(checkpoint.get("hidden_dim", 32))
    num_obstacles = int(checkpoint.get("num_obstacles", 10))
    map_size = float(checkpoint.get("map_size", 8.0))
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=num_obstacles,
        map_size=map_size, hidden_dim=hidden, future_steps=10, seq_len=10,
        node_dim=9, device=device, use_ring_slot_features=False,
        use_stage_embedding=False, use_role_embedding=False,
        use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True
    encoder.st_hgat.load_state_dict(checkpoint["st_hgat"])
    encoder.relation_head.load_state_dict(checkpoint["relation_head"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.no_grad()
def collect_proxy_dataset(checkpoint: Dict, num_scenarios: int,
                          num_pursuers: int, num_evaders: int,
                          seed: int, max_candidates: int,
                          device: torch.device
                          ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate grouped features/labels from fresh randomized scenarios."""
    if num_pursuers > 6 or num_evaders > 3:
        raise ValueError("current frozen encoder supports at most 6 pursuers/3 tasks")
    candidates = generate_candidate_assignments(num_pursuers, num_evaders)
    rng = np.random.default_rng(seed)
    if max_candidates > 0 and len(candidates) > max_candidates:
        chosen = np.sort(rng.choice(len(candidates), max_candidates,
                                    replace=False))
        candidates = [candidates[i] for i in chosen]
    candidate_array = np.stack(candidates)

    args = checkpoint.get("args", {})
    env = StageCoalitionEnv(
        length=float(checkpoint.get("map_size", 8.0)),
        num_obstacle=int(checkpoint.get("num_obstacles", 10)),
        max_pursuers=6, max_evaders=3,
        d_cap=float(checkpoint.get("d_cap", 1.5)),
        hold_steps=int(checkpoint.get("hold_steps", 5)), seed=seed,
        v_max_e=float(args.get("evader_speed", 0.2)),
    )
    encoder = _load_frozen_encoder(checkpoint, device)
    all_features = []
    all_labels = []
    for scenario_index in range(int(num_scenarios)):
        scenario_seed = seed + 50000 + scenario_index
        env.reset(num_pursuers=num_pursuers, num_evaders=num_evaders,
                  scenario_seed=scenario_seed)
        encoder.reset(step=0)
        snapshot = env.snapshot()
        encoded = encoder.encode(snapshot, differentiable=False)
        pursuer_embeddings = encoded["pursuer_embeddings"][:num_pursuers]
        task_embeddings = encoded["task_embeddings"][:num_evaders]
        positions = np.asarray(snapshot["positions"])
        pursuers = positions[:num_pursuers, :2]
        tasks = positions[num_pursuers:num_pursuers + num_evaders, :2]
        scene_features = []
        scene_labels = []
        for assignment in candidates:
            scene_features.append(build_candidate_features(
                pursuer_embeddings, task_embeddings, pursuers, tasks,
                assignment,
            ))
            scene_labels.append(geometric_proxy_value(
                pursuers, tasks, assignment, capture_size=3,
            ))
        all_features.append(torch.stack(scene_features).cpu())
        all_labels.append(torch.tensor(scene_labels, dtype=torch.float32))
    return torch.stack(all_features), torch.stack(all_labels), candidate_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenarios", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-pursuers", type=int, default=6)
    parser.add_argument("--num-evaders", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="0 keeps every non-empty labelled assignment")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    options = parser.parse_args()

    device = torch.device(options.device)
    checkpoint = torch.load(options.checkpoint, map_location=device,
                            weights_only=False)
    features, labels, candidates = collect_proxy_dataset(
        checkpoint, options.scenarios, options.num_pursuers,
        options.num_evaders, options.seed, options.max_candidates, device,
    )
    split = max(1, int(0.8 * len(features)))
    if split >= len(features):
        raise ValueError("at least two scenarios are required for train/validation")
    train_x, valid_x = features[:split].to(device), features[split:].to(device)
    train_y, valid_y = labels[:split].to(device), labels[split:].to(device)
    model = CoalitionValueHead(int(checkpoint.get("hidden_dim", 32))).to(device)
    before = ranking_accuracy(model, valid_x, valid_y)
    fit_metrics = fit_value_head(
        model, train_x, train_y, epochs=options.epochs, lr=options.lr,
        seed=options.seed,
    )
    metrics = {
        **fit_metrics,
        "ranking_accuracy_before": before,
        "ranking_accuracy": ranking_accuracy(model, valid_x, valid_y),
        "top1_accuracy": top1_accuracy(model, valid_x, valid_y),
        "train_scenarios": int(len(train_x)),
        "validation_scenarios": int(len(valid_x)),
        "candidates_per_scenario": int(len(candidates)),
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": VALUE_SCHEMA,
        "supervision": "geometric_proxy_not_rollout_return",
        "source_checkpoint": str(Path(options.checkpoint).resolve()),
        "hidden_dim": int(checkpoint.get("hidden_dim", 32)),
        "num_pursuers": int(options.num_pursuers),
        "num_evaders": int(options.num_evaders),
        "value_head": model.state_dict(),
        "candidates": candidates,
        "metrics": metrics,
        "args": vars(options),
    }, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

