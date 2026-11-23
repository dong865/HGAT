"""Utilities shared by SDGCR representation pretraining experiments."""

from pathlib import Path
from typing import Dict, Mapping, Union

import torch
import torch.nn as nn

from .stage_networks import StageFeatureEncoder


ENCODER_CHECKPOINT_TYPE = "sdgcr_encoder_pretrain"
ENCODER_CHECKPOINT_VERSION = 1


def encoder_architecture(encoder: StageFeatureEncoder) -> Dict[str, object]:
    return {
        "obs_dim": int(encoder.obs_dim),
        "hidden_dim": int(encoder.hidden_dim),
        "node_encoder": str(encoder.node_encoder_name),
        "max_pursuers": int(encoder.max_pursuers),
        "max_evaders": int(encoder.max_evaders),
        "future_steps": int(encoder.future_steps),
        "seq_len": int(encoder.seq_len),
    }


def snapshot_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def module_grad_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().cpu())
    return squared ** 0.5


def module_parameter_delta(
    before: Mapping[str, torch.Tensor], module: nn.Module,
) -> float:
    squared = 0.0
    for name, parameter in module.named_parameters():
        if name not in before:
            raise KeyError(f"missing parameter snapshot for {name}")
        difference = parameter.detach().cpu().float() - before[name].float()
        squared += float(difference.square().sum())
    return squared ** 0.5


def save_encoder_checkpoint(
    path: Union[str, Path],
    encoder: StageFeatureEncoder,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, object],
    step: int,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_type": ENCODER_CHECKPOINT_TYPE,
            "version": ENCODER_CHECKPOINT_VERSION,
            "architecture": encoder_architecture(encoder),
            "st_hgat": encoder.st_hgat.state_dict(),
            "relation_head": encoder.relation_head.state_dict(),
            "demand_head": encoder.demand_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": dict(metadata),
            "step": int(step),
        },
        target,
    )


def load_encoder_checkpoint(
    path: Union[str, Path],
    encoder: StageFeatureEncoder,
    device: torch.device,
) -> Dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != ENCODER_CHECKPOINT_TYPE:
        raise ValueError("not an SDGCR encoder-pretraining checkpoint")
    required = ("architecture", "st_hgat", "relation_head")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"encoder checkpoint missing keys: {missing}")
    expected = encoder_architecture(encoder)
    saved = payload["architecture"]
    for field, expected_value in expected.items():
        if saved.get(field) != expected_value:
            raise ValueError(
                f"encoder checkpoint {field} mismatch: "
                f"saved={saved.get(field)!r}, expected={expected_value!r}"
            )
    encoder.st_hgat.load_state_dict(payload["st_hgat"])
    encoder.relation_head.load_state_dict(payload["relation_head"])
    # DemandHead warmup is optional (older checkpoints lack it); load when both
    # sides support it so a pretrained demand head survives the warm start.
    if "demand_head" in payload and hasattr(encoder, "demand_head"):
        encoder.demand_head.load_state_dict(payload["demand_head"])
    return payload
