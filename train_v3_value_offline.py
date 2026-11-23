"""Offline same-state ranking training for the frozen-architecture V3 value head."""

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
from .v3cf_dataset import load_dataset

TRAINING_SCHEMA = "v3_same_state_value_offline_v1"


def standardize_within_state(labels, epsilon=1e-6):
    """Z-score only across candidates of one decision state."""
    values = torch.as_tensor(labels, dtype=torch.float32)
    if values.ndim != 1 or not values.numel():
        raise ValueError("labels must be a nonempty vector")
    if not torch.isfinite(values).all() or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("labels and epsilon must be finite; epsilon must be positive")
    return (values - values.mean()) / (values.std(unbiased=False) + float(epsilon))


def same_state_loss(predictions, labels, margin=0.1, regression_weight=0.1):
    if predictions.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("predictions and labels must be aligned vectors")
    hinges = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            sign = torch.sign(labels[left] - labels[right])
            if sign != 0:
                hinges.append(torch.relu(
                    predictions.new_tensor(float(margin))
                    - sign * (predictions[left] - predictions[right])
                ))
    ranking = (torch.stack(hinges).mean() if hinges
               else predictions.sum() * 0.0)
    regression = F.smooth_l1_loss(
        predictions, standardize_within_state(labels).to(predictions),
    )
    return ranking + float(regression_weight) * regression, ranking, regression


def _seed_split(records, validation_fraction=0.2):
    seeds = sorted({int(row["seed"]) for row in records})
    if len(seeds) < 2:
        raise ValueError("offline training requires at least two distinct seeds")
    validation_count = max(1, int(round(len(seeds) * validation_fraction)))
    validation_seeds = set(seeds[-validation_count:])
    train = [row for row in records if int(row["seed"]) not in validation_seeds]
    validation = [row for row in records if int(row["seed"]) in validation_seeds]
    if not train or not validation:
        raise ValueError("seed split produced an empty partition")
    return train, validation, sorted(validation_seeds)


def _tensors(record, device):
    features = torch.as_tensor(record["features"], dtype=torch.float32, device=device)
    alive = torch.as_tensor(record["alive"], dtype=torch.bool, device=device)
    labels = torch.as_tensor(record["G_H"], dtype=torch.float32, device=device)
    if (features.ndim != 3 or alive.shape != features.shape[:2]
            or labels.shape != (len(features),)):
        raise ValueError(f"invalid same-state record {record.get('state_id')}")
    return features, alive, labels


def _mode(record, index):
    if "modes" in record:
        return tuple(int(value) for value in record["modes"][index])
    return tuple(np.bincount(
        np.asarray(record["assignments"][index], dtype=np.int64), minlength=2,
    ).tolist())


class _MetricAccumulator:
    def __init__(self):
        self.totals = dict.fromkeys((
            "pairs", "correct_pairs", "mode_pairs", "correct_mode_pairs",
            "top1", "correct_top1", "critical_pairs", "correct_critical_pairs",
        ), 0)
        self.states = 0

    def update(self, record, predictions, labels):
        pred = predictions.detach().cpu().numpy()
        target = labels.detach().cpu().numpy()
        critical = record.get("meta", {}).get("sampling_reason") == "capture_critical"
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                difference = target[left] - target[right]
                if difference == 0:
                    continue
                correct = float((pred[left] - pred[right]) * difference > 0)
                self.totals["pairs"] += 1
                self.totals["correct_pairs"] += correct
                if critical:
                    self.totals["critical_pairs"] += 1
                    self.totals["correct_critical_pairs"] += correct
                modes = {_mode(record, left), _mode(record, right)}
                if modes == {(2, 2), (3, 1)}:
                    self.totals["mode_pairs"] += 1
                    self.totals["correct_mode_pairs"] += correct
        self.states += 1
        true_best = int(np.argmax(target))
        pred_best = int(np.argmax(pred))
        self.totals["top1"] += 1
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
            "mode_2plus2_vs_3plus1_accuracy": ratio("correct_mode_pairs", "mode_pairs"),
            "top1_mode_accuracy": ratio("correct_top1", "top1"),
            "capture_critical_accuracy": ratio("correct_critical_pairs", "critical_pairs"),
            "pair_count": self.totals["pairs"],
        }


@torch.no_grad()
def evaluate_value_head(model, records, device):
    model.eval()
    metrics = _MetricAccumulator()
    losses = []
    for record in records:
        features, alive, labels = _tensors(record, device)
        predictions = model(features, alive)
        loss, _, _ = same_state_loss(predictions, labels)
        losses.append(float(loss.detach().cpu()))
        metrics.update(record, predictions, labels)
    result = metrics.result()
    result["loss"] = float(np.mean(losses)) if losses else None
    return result


def _initialize_head(checkpoint, hidden_dim, head_width, device):
    if checkpoint is None:
        if hidden_dim is None:
            raise ValueError("hidden_dim is required without a warm-start checkpoint")
        return ScaleAwareCoalitionValueHead(hidden_dim, head_width).to(device), None
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 32) if hidden_dim is None else hidden_dim)
    head_width = int(config.get("head_width", head_width))
    model = ScaleAwareCoalitionValueHead(hidden_dim, head_width).to(device)
    model.load_state_dict(payload["value_head"], strict=True)
    return model, {
        "path": str(Path(checkpoint).resolve()),
        "architecture_version": payload.get("architecture_version"),
        "initialization": "loaded_value_head_no_reinitialization",
    }


def train_value_offline(records, output, epochs=200, checkpoint=None,
                        hidden_dim=None, head_width=64, learning_rate=1e-3,
                        batch_size=16, margin=0.1, regression_weight=0.1,
                        validation_fraction=0.2, device="cpu", seed=17):
    records = list(records)
    train, validation, validation_seeds = _seed_split(records, validation_fraction)
    device = torch.device(device)
    torch.manual_seed(seed)
    model, warm_start = _initialize_head(checkpoint, hidden_dim, head_width, device)
    # This module is the entire trainable system; no encoder/actor parameters exist here.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(seed)
    history, best_accuracy, best_epoch = [], -math.inf, 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), int(batch_size)):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for index in order[start:start + int(batch_size)]:
                features, alive, labels = _tensors(train[int(index)], device)
                predictions = model(features, alive)
                loss, _, _ = same_state_loss(
                    predictions, labels, margin, regression_weight,
                )
                batch_losses.append(loss)
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_value_head(model, train, device)
        validation_metrics = evaluate_value_head(model, validation, device)
        train_metrics["optimization_loss"] = float(np.mean(losses))
        row = {"epoch": epoch, "train": train_metrics,
               "validation": validation_metrics}
        history.append(row)
        accuracy = validation_metrics["pairwise_ranking_accuracy"]
        if accuracy is not None and accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save({
                "schema": TRAINING_SCHEMA,
                "value_head": model.state_dict(),
                "hidden_dim": model.hidden_dim,
                "head_width": head_width,
                "best_epoch": best_epoch,
                "best_validation_pairwise_ranking_accuracy": best_accuracy,
                "warm_start": warm_start,
                "objective": {"ranking": "same_state_hinge", "margin": margin,
                              "regression": "huber_on_within_state_zscore",
                              "regression_weight": regression_weight},
            }, output / "value_head_trained.pth")
    metrics = {
        "schema": TRAINING_SCHEMA,
        "epochs": int(epochs), "best_epoch": int(best_epoch),
        "best_validation_pairwise_ranking_accuracy": float(best_accuracy),
        "train_states": len(train), "validation_states": len(validation),
        "validation_seeds": validation_seeds,
        "warm_start": warm_start, "history": history,
    }
    text = json.dumps(metrics, indent=2, allow_nan=False)
    (output / "training_metrics.json").write_text(text, encoding="utf-8")
    (output / "training_curves.json").write_text(text, encoding="utf-8")
    return {"model": model, "history": history, "metrics": metrics}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/v3cf_value/dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output", default="checkpoints/v3cf_value")
    parser.add_argument("--checkpoint",
                        default="checkpoints/v3_hybrid_dcap08c/best_4v2.pth")
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
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    records = load_dataset(args.dataset)
    result = train_value_offline(
        records, args.output, epochs=args.epochs, checkpoint=args.checkpoint,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        margin=args.margin, regression_weight=args.regression_weight,
        validation_fraction=args.validation_fraction, device=args.device,
        seed=args.seed,
    )
    print(json.dumps({
        "epochs": args.epochs,
        "best_epoch": result["metrics"]["best_epoch"],
        "best_validation_pairwise_ranking_accuracy":
            result["metrics"]["best_validation_pairwise_ranking_accuracy"],
        "output": str(Path(args.output) / "value_head_trained.pth"),
    }, indent=2))


if __name__ == "__main__":
    main()
