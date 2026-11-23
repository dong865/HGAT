"""Train only the 81-input V3 value head on near-capture fork labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .v3_demand_distillation import ScaleAwareCoalitionValueHead
from .v3cf2_dataset import BASE_FEATURE_DIM, FEATURE_DIM, load_dataset

TRAINING_SCHEMA = "v3_near_capture_potential_value_offline_v2"


class LookaheadValueHead(ScaleAwareCoalitionValueHead):
    """Original mean+max value head with four extra per-task input columns."""

    def __init__(self, hidden_dim, width=64):
        super().__init__(hidden_dim, width)
        original = self.task_encoder[0]
        self.task_encoder[0] = torch.nn.Linear(FEATURE_DIM, original.out_features)
        self.base_feature_dim = BASE_FEATURE_DIM
        self.lookahead_feature_dim = FEATURE_DIM - BASE_FEATURE_DIM


def standardize_within_state(labels, epsilon=1e-6):
    values = torch.as_tensor(labels, dtype=torch.float32)
    if values.ndim != 1 or not values.numel():
        raise ValueError("labels must be a nonempty vector")
    if not torch.isfinite(values).all() or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("labels and epsilon must be finite; epsilon must be positive")
    return (values - values.mean()) / (values.std(unbiased=False) + float(epsilon))


def split_by_seed(records, validation_fraction=0.2):
    """Split whole seeds, never individual states, into train and validation."""
    seeds = sorted({int(row["seed"]) for row in records})
    if len(seeds) < 2:
        raise ValueError("offline training requires at least two distinct seeds")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    validation_count = min(len(seeds) - 1, max(1, int(round(len(seeds) * validation_fraction))))
    validation_seeds = set(seeds[-validation_count:])
    train = [row for row in records if int(row["seed"]) not in validation_seeds]
    validation = [row for row in records if int(row["seed"]) in validation_seeds]
    if not train or not validation:
        raise ValueError("seed split produced an empty partition")
    return train, validation, sorted(validation_seeds)


# Backward-compatible internal name used by the preceding offline trainer.
_seed_split = split_by_seed


def _record_horizon(record):
    return int(record.get("meta", {}).get("horizon", 300))


def _near_steps(record):
    if "near_capture_step" in record:
        return list(record["near_capture_step"])
    return [row.get("near_capture_step") for row in record["capture_labels"]]


def _min_distances(record):
    if "min_d_max" in record:
        return list(record["min_d_max"])
    return [row.get("min_d_max") for row in record["capture_labels"]]


def ranking_pairs(record):
    """Yield (better, worse, weight, kind) for near-capture and weak geometry labels."""
    steps = _near_steps(record)
    distances = _min_distances(record)
    if len(steps) != len(distances):
        raise ValueError("near-capture labels must align with candidates")
    pairs = []
    for left in range(len(steps)):
        for right in range(left + 1, len(steps)):
            a, b = steps[left], steps[right]
            if a is not None and b is None:
                pairs.append((left, right, 1.0, "achievement"))
            elif a is None and b is not None:
                pairs.append((right, left, 1.0, "achievement"))
            elif a is not None and b is not None and int(a) != int(b):
                better, worse = (left, right) if int(a) < int(b) else (right, left)
                pairs.append((better, worse, 1.0, "timing"))
            elif a is None and b is None:
                da, db = distances[left], distances[right]
                if da is None or db is None or not math.isfinite(da) or not math.isfinite(db):
                    continue
                if float(da) != float(db):
                    better, worse = (left, right) if float(da) < float(db) else (right, left)
                    pairs.append((better, worse, 0.1, "weak_distance"))
    return pairs


def regression_targets(record, device):
    horizon = _record_horizon(record)
    values = [horizon + 1 if step is None else int(step) for step in _near_steps(record)]
    return standardize_within_state(torch.tensor(values, dtype=torch.float32)).to(device)


def same_state_loss(predictions, record, margin=0.1, regression_weight=0.1):
    """Strong near-capture hinge + 0.1-weight all-missing geometry hinge + Huber."""
    if predictions.ndim != 1 or len(predictions) != len(_near_steps(record)):
        raise ValueError("predictions and near-capture labels must be aligned")
    hinges, weights = [], []
    for better, worse, weight, _ in ranking_pairs(record):
        hinges.append(F.relu(
            predictions.new_tensor(float(margin)) - predictions[better] + predictions[worse]
        ))
        weights.append(float(weight))
    if hinges:
        weight_tensor = predictions.new_tensor(weights)
        ranking = (torch.stack(hinges) * weight_tensor).sum() / weight_tensor.sum()
    else:
        ranking = predictions.sum() * 0.0
    regression = F.smooth_l1_loss(predictions, regression_targets(record, predictions.device))
    total = ranking + float(regression_weight) * regression
    return total, ranking, regression


def _tensors(record, device):
    features = torch.as_tensor(record["features"], dtype=torch.float32, device=device)
    alive = torch.as_tensor(record["alive"], dtype=torch.bool, device=device)
    if (features.ndim != 3 or features.shape[-1] != FEATURE_DIM
            or alive.shape != features.shape[:2]
            or len(_near_steps(record)) != len(features)):
        raise ValueError(f"invalid same-state v3cf2 record {record.get('state_id')}")
    return features, alive


def _mode(record, index):
    if "modes" in record:
        return tuple(int(value) for value in record["modes"][index])
    return tuple(np.bincount(
        np.asarray(record["assignments"][index], dtype=np.int64), minlength=2,
    ).tolist())


def _target_best_index(record):
    steps = _near_steps(record)
    distances = _min_distances(record)
    scores = np.asarray(record["G_H"], dtype=np.float64)
    achieved = [index for index, step in enumerate(steps) if step is not None]
    if achieved:
        earliest = min(int(steps[index]) for index in achieved)
        finalists = [index for index in achieved if int(steps[index]) == earliest]
    else:
        finite = [index for index, value in enumerate(distances)
                  if value is not None and math.isfinite(value)]
        if not finite:
            finalists = list(range(len(steps)))
        else:
            closest = min(float(distances[index]) for index in finite)
            finalists = [index for index in finite if float(distances[index]) == closest]
    return int(max(finalists, key=lambda index: scores[index]))


class _MetricAccumulator:
    def __init__(self):
        self.totals = dict.fromkeys((
            "pairs", "correct_pairs", "strong_pairs", "correct_strong_pairs",
            "weak_pairs", "correct_weak_pairs", "achievement_pairs",
            "correct_achievement_pairs", "top1", "correct_top1",
        ), 0)
        self.states = 0

    def update(self, record, predictions):
        pred = predictions.detach().cpu().numpy()
        for better, worse, _, kind in ranking_pairs(record):
            correct = float(pred[better] > pred[worse])
            self.totals["pairs"] += 1
            self.totals["correct_pairs"] += correct
            if kind == "weak_distance":
                self.totals["weak_pairs"] += 1
                self.totals["correct_weak_pairs"] += correct
            else:
                self.totals["strong_pairs"] += 1
                self.totals["correct_strong_pairs"] += correct
            if kind == "achievement":
                self.totals["achievement_pairs"] += 1
                self.totals["correct_achievement_pairs"] += correct
        self.states += 1
        self.totals["top1"] += 1
        true_best = _target_best_index(record)
        pred_best = int(np.argmax(pred))
        self.totals["correct_top1"] += float(
            _mode(record, true_best) == _mode(record, pred_best)
        )

    def result(self):
        def ratio(correct, total):
            return (float(self.totals[correct] / self.totals[total])
                    if self.totals[total] else None)
        return {
            "states": self.states,
            "pairwise_ranking_accuracy": ratio("correct_pairs", "pairs"),
            "strong_pairwise_ranking_accuracy": ratio(
                "correct_strong_pairs", "strong_pairs"),
            "weak_distance_ranking_accuracy": ratio("correct_weak_pairs", "weak_pairs"),
            "achieved_vs_unachieved_accuracy": ratio(
                "correct_achievement_pairs", "achievement_pairs"),
            "top1_mode_accuracy": ratio("correct_top1", "top1"),
            "pair_count": self.totals["pairs"],
            "strong_pair_count": self.totals["strong_pairs"],
            "weak_pair_count": self.totals["weak_pairs"],
        }


@torch.no_grad()
def evaluate_value_head(model, records, device, margin=0.1, regression_weight=0.1):
    model.eval()
    metrics = _MetricAccumulator()
    losses = []
    for record in records:
        features, alive = _tensors(record, device)
        predictions = model(features, alive)
        loss, _, _ = same_state_loss(predictions, record, margin, regression_weight)
        losses.append(float(loss.detach().cpu()))
        metrics.update(record, predictions)
    result = metrics.result()
    result["loss"] = float(np.mean(losses)) if losses else None
    return result


def _initialize_head(checkpoint, hidden_dim, head_width, device):
    if checkpoint is None:
        if hidden_dim is None:
            raise ValueError("hidden_dim is required without a warm-start checkpoint")
        return LookaheadValueHead(hidden_dim, head_width).to(device), None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference = payload["value_head"]
    config = payload.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 32) if hidden_dim is None else hidden_dim)
    inferred_width = int(reference["task_encoder.0.weight"].shape[0])
    head_width = inferred_width if head_width is None else int(head_width)
    model = LookaheadValueHead(hidden_dim, head_width).to(device)
    state = model.state_dict()
    for key, value in reference.items():
        if key not in state:
            raise ValueError(f"unknown warm-start value_head key: {key}")
        if key == "task_encoder.0.weight":
            if value.shape != (head_width, BASE_FEATURE_DIM):
                raise ValueError(
                    f"expected warm-start first layer [{head_width}, {BASE_FEATURE_DIM}], "
                    f"got {tuple(value.shape)}")
            state[key].zero_()
            state[key][:, :BASE_FEATURE_DIM] = value
        else:
            if tuple(state[key].shape) != tuple(value.shape):
                raise ValueError(f"warm-start shape mismatch for {key}")
            state[key].copy_(value)
    model.load_state_dict(state)
    return model, {
        "path": str(Path(checkpoint).resolve()),
        "architecture_version": payload.get("architecture_version"),
        "initialization": "original_77_columns_copied; lookahead_columns_zero_initialized",
    }


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_value_offline(records, output="checkpoints/v3cf2", epochs=200,
                        checkpoint="checkpoints/v3_08_parallel/baseline/dcap08c_best_4v2.pth",
                        hidden_dim=None, head_width=64, learning_rate=1e-3,
                        batch_size=16, margin=0.1, regression_weight=0.1,
                        validation_fraction=0.2, patience=30, device="cpu", seed=17):
    records = list(records)
    train, validation, validation_seeds = split_by_seed(records, validation_fraction)
    device = torch.device(device)
    torch.manual_seed(seed)
    model, warm_start = _initialize_head(checkpoint, hidden_dim, head_width, device)
    # The model contains only value-head parameters; actor and encoder are absent.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(seed)
    history, best_accuracy, best_epoch, best_state = [], -math.inf, 0, None
    stale_epochs = 0
    output = None if output is None else Path(output)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for index in order[start:start + int(batch_size)]:
                record = train[int(index)]
                features, alive = _tensors(record, device)
                predictions = model(features, alive)
                loss, _, _ = same_state_loss(
                    predictions, record, margin, regression_weight,
                )
                batch_losses.append(loss)
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_value_head(model, train, device, margin, regression_weight)
        validation_metrics = evaluate_value_head(model, validation, device, margin, regression_weight)
        train_metrics["optimization_loss"] = float(np.mean(losses))
        history.append({"epoch": epoch, "train": train_metrics,
                        "validation": validation_metrics})
        accuracy = validation_metrics["pairwise_ranking_accuracy"]
        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_state = _cpu_state(model)
            stale_epochs = 0
            if output is not None:
                torch.save({
                    "schema": TRAINING_SCHEMA,
                    "value_head": best_state,
                    "hidden_dim": model.hidden_dim,
                    "head_width": int(model.task_encoder[0].out_features),
                    "base_feature_dim": BASE_FEATURE_DIM,
                    "feature_dim": FEATURE_DIM,
                    "best_epoch": best_epoch,
                    "best_validation_pairwise_ranking_accuracy": best_accuracy,
                    "warm_start": warm_start,
                    "objective": {
                        "ranking": "near_capture_achievement_and_timing_hinge",
                        "weak_all_missing": "minimum_assigned_d_max",
                        "weak_pair_weight": 0.1,
                        "margin": margin,
                        "regression": "huber_on_within_state_near_capture_time_zscore",
                        "unachieved_regression_time": "horizon_plus_one",
                        "regression_weight": regression_weight,
                    },
                }, output / "value_head_trained.pth")
        else:
            stale_epochs += 1
        if patience is not None and stale_epochs >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "schema": TRAINING_SCHEMA,
        "epochs_requested": int(epochs), "epochs_run": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_pairwise_ranking_accuracy": (
            None if not math.isfinite(best_accuracy) else float(best_accuracy)),
        "train_states": len(train), "validation_states": len(validation),
        "train_seeds": sorted({int(row["seed"]) for row in train}),
        "validation_seeds": validation_seeds,
        "unseen_seed_metrics": history[-1]["validation"] if history else None,
        "warm_start": warm_start, "history": history,
    }
    if output is not None:
        text = json.dumps(metrics, indent=2, allow_nan=False)
        (output / "training_metrics.json").write_text(text, encoding="utf-8")
        (output / "training_curves.json").write_text(text, encoding="utf-8")
    return {"model": model, "history": history, "metrics": metrics}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/v3cf2/dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output", default="checkpoints/v3cf2")
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_08_parallel/baseline/dcap08c_best_4v2.pth")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--regression-weight", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main():
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size and patience must be positive")
    records = load_dataset(args.dataset)
    result = train_value_offline(
        records, args.output, epochs=args.epochs, checkpoint=args.checkpoint,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        margin=args.margin, regression_weight=args.regression_weight,
        validation_fraction=args.validation_fraction, patience=args.patience,
        device=args.device, seed=args.seed,
    )
    print(json.dumps({
        "epochs_run": result["metrics"]["epochs_run"],
        "best_epoch": result["metrics"]["best_epoch"],
        "best_validation_pairwise_ranking_accuracy":
            result["metrics"]["best_validation_pairwise_ranking_accuracy"],
        "output": str(Path(args.output) / "value_head_trained.pth"),
    }, indent=2))


if __name__ == "__main__":
    main()
