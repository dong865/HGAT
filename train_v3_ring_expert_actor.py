"""Actor-only training from external ring-expert trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .stage_networks import SharedActor


def anchored_actor_loss(actor, original_actor, obs, expert_action,
                        original_action, farthest_member, farthest_weight=3.0,
                        anchor_weight=0.2, param_weight=1e-4):
    """Weighted behavior cloning with behavior and parameter anchors."""
    if obs.ndim != 3 or expert_action.shape != obs.shape[:2] + (2,):
        raise ValueError("trajectory tensors have incompatible shapes")
    batch, agents, width = obs.shape
    prediction = actor(obs.reshape(batch * agents, width)).reshape(batch, agents, 2)
    weights = prediction.new_ones(batch, agents)
    rows = torch.arange(batch, device=prediction.device)
    weights[rows, farthest_member.long()] = float(farthest_weight)
    expert = ((prediction - expert_action).square().mean(dim=-1) * weights).mean()
    anchor = (prediction - original_action).square().mean()
    parameter = prediction.new_zeros(())
    for current, baseline in zip(actor.parameters(), original_actor.parameters()):
        parameter = parameter + (current - baseline.detach()).square().sum()
    total = expert + float(anchor_weight) * anchor + float(param_weight) * parameter
    return total, {"expert": expert.detach(), "anchor": anchor.detach(),
                   "parameter": parameter.detach()}


def inject_actor_weights(payload, actor):
    result = copy.deepcopy(payload)
    state = copy.deepcopy(actor.state_dict())
    result["actor"] = state
    result["actor_target"] = copy.deepcopy(state)
    return result


def load_records(paths):
    trajectories = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            trajectories.extend(json.loads(line) for line in stream if line.strip())
    seeds = [int(row["seed"]) for row in trajectories]
    if not trajectories or len(seeds) != len(set(seeds)):
        raise ValueError("expert trajectories must contain unique seeds")
    if any(seed == 70086 or not 72000 <= seed <= 72999 for seed in seeds):
        raise ValueError("sealed or nontraining scene in expert data")
    return trajectories


def select_expert_trajectories(trajectories, require_full_capture=True):
    """Keep only demonstrations that satisfy the requested task outcome."""
    rows = list(trajectories)
    if require_full_capture:
        rows = [row for row in rows if bool(row.get("full_capture", False))]
    if not rows:
        raise ValueError("expert selection contains no usable trajectories")
    return rows


def flatten_trajectories(trajectories, post_capture_only=False):
    rows = []
    for trajectory in trajectories:
        seed = int(trajectory["seed"])
        records = list(trajectory["records"])
        if post_capture_only:
            initial = tuple(int(value) for value in trajectory["initial_assignment"])
            start = next((index for index, record in enumerate(records)
                          if tuple(record["assignment"]) != initial
                          and len(set(record["assignment"])) == 1), len(records))
            records = records[start:]
        for record in records:
            obs = np.asarray(record["obs"], dtype=np.float32)
            expert = np.asarray(record["expert_action"], dtype=np.float32)
            original = np.asarray(record["original_action"], dtype=np.float32)
            farthest = int(record["farthest_member"])
            if (obs.ndim != 2 or obs.shape[0] != 4 or expert.shape != (4, 2)
                    or original.shape != (4, 2) or not 0 <= farthest < 4
                    or not np.isfinite(obs).all() or not np.isfinite(expert).all()
                    or not np.isfinite(original).all()):
                raise ValueError("invalid expert record")
            rows.append((seed, obs, expert, original, farthest))
    if not rows:
        raise ValueError("expert dataset contains no frames")
    return rows


def _batch(rows, indices, device):
    selected = [rows[index] for index in indices]
    return (
        torch.as_tensor(np.stack([row[1] for row in selected]), device=device),
        torch.as_tensor(np.stack([row[2] for row in selected]), device=device),
        torch.as_tensor(np.stack([row[3] for row in selected]), device=device),
        torch.as_tensor([row[4] for row in selected], device=device),
    )


def train_actor(checkpoint, trajectories, output, epochs=80, batch_size=256,
                lr=1e-4, farthest_weight=3.0, anchor_weight=0.2,
                param_weight=1e-4, seed=3701, patience=12, device="cpu",
                require_full_capture=True, post_capture_only=False):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    input_dim = int(payload["actor"]["net.0.weight"].shape[1])
    actor = SharedActor(input_dim, hidden_dim=int(config["actor_hidden"])).to(device)
    actor.load_state_dict(payload["actor"])
    original_actor = copy.deepcopy(actor).eval()
    for parameter in original_actor.parameters():
        parameter.requires_grad_(False)
    trajectories = select_expert_trajectories(
        trajectories, require_full_capture=require_full_capture,
    )
    rows = flatten_trajectories(
        trajectories, post_capture_only=post_capture_only,
    )
    unique_seeds = sorted({row[0] for row in rows})
    if len(unique_seeds) < 2:
        raise ValueError("at least two successful scenario seeds are required")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_seeds)
    rng.shuffle(shuffled)
    validation_count = max(1, int(math.ceil(0.2 * len(shuffled))))
    validation_seeds = set(map(int, shuffled[:validation_count]))
    train_indices = np.asarray([i for i, row in enumerate(rows)
                                if row[0] not in validation_seeds])
    validation_indices = np.asarray([i for i, row in enumerate(rows)
                                     if row[0] in validation_seeds])
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    initial = inject_actor_weights(payload, actor)
    torch.save(initial, output / "epoch0.pth")
    best_state, best_loss, stale, history = copy.deepcopy(actor.state_dict()), float("inf"), 0, []
    for epoch in range(int(epochs)):
        actor.train()
        rng.shuffle(train_indices)
        totals = []
        for start in range(0, len(train_indices), int(batch_size)):
            tensors = _batch(rows, train_indices[start:start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = anchored_actor_loss(
                actor, original_actor, *tensors, farthest_weight=farthest_weight,
                anchor_weight=anchor_weight, param_weight=param_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        actor.eval()
        with torch.no_grad():
            validation, parts = anchored_actor_loss(
                actor, original_actor, *_batch(rows, validation_indices, device),
                farthest_weight=farthest_weight, anchor_weight=anchor_weight,
                param_weight=param_weight,
            )
        value = float(validation.cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(totals)),
                        "validation_loss": value,
                        **{key: float(item.cpu()) for key, item in parts.items()}})
        if value < best_loss:
            best_loss, stale = value, 0
            best_state = copy.deepcopy(actor.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    actor.load_state_dict(best_state)
    result = inject_actor_weights(payload, actor)
    result.setdefault("metadata", {})["ring_expert_actor"] = {
        "source_checkpoint": str(Path(checkpoint).resolve()),
        "training_seeds": sorted(set(unique_seeds) - validation_seeds),
        "validation_seeds": sorted(validation_seeds), "best_validation_loss": best_loss,
        "farthest_weight": farthest_weight, "anchor_weight": anchor_weight,
        "param_weight": param_weight, "seed": seed,
        "require_full_capture": bool(require_full_capture),
        "post_capture_only": bool(post_capture_only),
    }
    torch.save(result, output / "best.pth")
    (output / "metrics.json").write_text(json.dumps({
        "history": history, "best_validation_loss": best_loss,
        "training_seeds": sorted(set(unique_seeds) - validation_seeds),
        "validation_seeds": sorted(validation_seeds),
    }, indent=2) + "\n", encoding="utf-8")
    return result["metadata"]["ring_expert_actor"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trajectories", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--farthest-weight", type=float, default=3.0)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--param-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-partial-trajectories", action="store_true",
                        help="ablation only: also train on trajectories that captured one target")
    parser.add_argument("--post-capture-only", action="store_true",
                        help="train a finisher only on frames after dead-task repair")
    args = parser.parse_args()
    trajectories = load_records(args.trajectories)
    summary = train_actor(args.checkpoint, trajectories, args.output,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, farthest_weight=args.farthest_weight,
                          anchor_weight=args.anchor_weight,
                          param_weight=args.param_weight, seed=args.seed,
                          patience=args.patience, device=args.device,
                          require_full_capture=not args.allow_partial_trajectories,
                          post_capture_only=args.post_capture_only)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
