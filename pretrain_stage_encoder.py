"""Self-supervised ST-HGAT/relation-head pretraining for SDGCR.

The learning target is a delayed future-interception relation.  Exploratory
commands only generate state coverage; no action is ever used as a label.
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .encoder_pretraining import (
    module_grad_norm,
    module_parameter_delta,
    save_encoder_checkpoint,
    snapshot_parameters,
)
from .stage_env import StageCoalitionEnv
from .stage_networks import StageFeatureEncoder


def run_auxiliary_update(
    encoder: StageFeatureEncoder,
    optimizer: torch.optim.Optimizer,
    current_step: int,
    grad_clip: float,
    batch_size: int = 64,
    demand_info: Optional[Dict] = None,
    demand_weight: float = 0.3,
) -> Dict[str, float]:
    losses = encoder.auxiliary_loss(current_step=current_step, batch_size=batch_size)
    if losses is None:
        return {"updated": False}
    loss = losses["total"]
    demand_ce = None
    if demand_info is not None:
        logits = demand_info["logits"]
        mask = demand_info["task_mask"]
        if bool(mask.any()):
            targets = demand_info["stages"][mask].clamp(0, 2)
            demand_ce = torch.nn.functional.cross_entropy(logits[mask], targets)
            loss = loss + float(demand_weight) * demand_ce
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite encoder auxiliary loss")
    st_before = snapshot_parameters(encoder.st_hgat)
    relation_before = snapshot_parameters(encoder.relation_head)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    st_grad = module_grad_norm(encoder.st_hgat)
    relation_grad = module_grad_norm(encoder.relation_head)
    total_grad = float(torch.nn.utils.clip_grad_norm_(
        encoder.auxiliary_parameters(), float(grad_clip),
    ))
    if not all(math.isfinite(value) for value in (st_grad, relation_grad, total_grad)):
        raise FloatingPointError("non-finite encoder auxiliary gradient")
    optimizer.step()
    result = {
        "updated": True,
        "loss": float(loss.detach().cpu()),
        "relation_loss": float(losses["relation_loss"].detach().cpu()),
        "st_hgat_grad_norm": st_grad,
        "relation_head_grad_norm": relation_grad,
        "total_grad_norm": total_grad,
        "st_hgat_parameter_delta": module_parameter_delta(st_before, encoder.st_hgat),
        "relation_head_parameter_delta": module_parameter_delta(
            relation_before, encoder.relation_head,
        ),
    }
    if demand_ce is not None:
        result["demand_ce"] = float(demand_ce.detach().cpu())
    return result


def exploratory_actions(env: StageCoalitionEnv, rng: np.random.Generator) -> np.ndarray:
    """Generate diverse pursuit/tangential commands without learning labels."""
    actions = np.zeros((env.max_pursuers, 2), dtype=np.float32)
    evaders = env.evader_positions()[:, :2]
    pursuers = env.pursuer_positions()[:, :2]
    for i in range(env.num_pursuers):
        distances = np.linalg.norm(evaders - pursuers[i], axis=1)
        target = evaders[int(np.argmin(distances))]
        direction = target - pursuers[i]
        norm = float(np.linalg.norm(direction))
        radial = direction / max(norm, 1e-6)
        tangent = np.array([-radial[1], radial[0]], dtype=np.float32)
        command = 0.55 * radial + rng.uniform(-0.8, 0.8) * tangent
        command += rng.normal(0.0, 0.18, size=2)
        command_norm = float(np.linalg.norm(command))
        actions[i] = command / max(1.0, command_norm)
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--d-cap", type=float, default=2.0)
    parser.add_argument("--hold-steps", type=int, default=5)
    parser.add_argument("--pursuer-speed", type=float, default=0.35)
    parser.add_argument("--evader-speed", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--update-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--demand-weight", type=float, default=0.3,
                        help="weight of the stage-classification CE that pretrains DemandHead")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output", default="checkpoints/encoder_pretrain_6v2")
    parser.add_argument("--tensorboard-dir", default="runs")
    parser.add_argument("--run-name", default="SDGCR_encoder_pretrain_6v2")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-stage-embedding", action="store_true")
    parser.add_argument("--no-role-embedding", action="store_true")
    parser.add_argument("--no-ring-slots", action="store_true")
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))

    env = StageCoalitionEnv(
        length=args.map_size,
        num_obstacle=args.num_obstacles,
        max_pursuers=6,
        max_evaders=3,
        d_cap=args.d_cap,
        hold_steps=args.hold_steps,
        v_max=args.pursuer_speed,
        seed=args.seed,
    )
    env.v_max_e = float(args.evader_speed)
    encoder = StageFeatureEncoder(
        max_pursuers=6,
        max_evaders=3,
        num_obstacles=args.num_obstacles,
        map_size=args.map_size,
        hidden_dim=args.hidden_dim,
        future_steps=args.future_steps,
        seq_len=args.seq_len,
        node_dim=9,
        device=device,
        use_ring_slot_features=not args.no_ring_slots,
        use_stage_embedding=not args.no_stage_embedding,
        use_role_embedding=not args.no_role_embedding,
        output_layer_norm=True, output_gain=0.8,
    ).to(device)
    encoder.train()
    optimizer = torch.optim.Adam(encoder.auxiliary_parameters(), lr=args.lr)
    metadata = {
        "protocol": "strict_6v2",
        "seed": args.seed,
        "pursuer_speed": args.pursuer_speed,
        "evader_speed": args.evader_speed,
        "d_cap": args.d_cap,
        "hold_steps": args.hold_steps,
        "num_obstacles": args.num_obstacles,
        "use_ring_slot_features": True,
    }
    total_steps = 0
    updates = 0
    best_loss = float("inf")
    started = time.time()
    last_demand_ce = float("nan")
    for episode in range(args.episodes):
        env.reset(num_pursuers=6, num_evaders=2, scenario_seed=args.seed + episode)
        encoder.reset(step=0)
        for _ in range(args.max_steps):
            feats = encoder.encode(env.snapshot())
            demand_info = {
                "logits": feats["demand_logits"][: env.num_evaders],
                "stages": torch.as_tensor(
                    feats["stages"][: env.num_evaders], device=device,
                ).long(),
                "task_mask": torch.as_tensor(
                    feats["task_mask"][: env.num_evaders], device=device,
                ).bool(),
            }
            _, done, _ = env.step(exploratory_actions(env, rng), max_steps=args.max_steps)
            total_steps += 1
            if total_steps % max(1, args.update_interval) == 0:
                metrics = run_auxiliary_update(
                    encoder, optimizer, encoder._step, args.grad_clip, args.batch_size,
                    demand_info=demand_info, demand_weight=args.demand_weight,
                )
                if metrics["updated"]:
                    updates += 1
                    last_demand_ce = metrics.get("demand_ce", last_demand_ce)
                    for name, value in metrics.items():
                        if writer is not None and name != "updated":
                            writer.add_scalar(f"pretrain/{name}", value, total_steps)
                    if metrics["loss"] < best_loss:
                        best_loss = metrics["loss"]
                        save_encoder_checkpoint(
                            output / "best_encoder.pth", encoder, optimizer,
                            {**metadata, "best_loss": best_loss}, total_steps,
                        )
            if done:
                break
        if (episode + 1) % max(1, args.save_interval) == 0:
            save_encoder_checkpoint(
                output / f"checkpoint_{episode + 1:06d}.pth",
                encoder, optimizer, metadata, total_steps,
            )
        print(
            f"[encoder-pretrain] ep={episode + 1}/{args.episodes} "
            f"steps={total_steps} updates={updates} best_loss={best_loss:.6f} "
            f"demand_ce={last_demand_ce:.4f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    save_encoder_checkpoint(
        output / "final_encoder.pth", encoder, optimizer,
        {**metadata, "best_loss": best_loss, "updates": updates}, total_steps,
    )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
