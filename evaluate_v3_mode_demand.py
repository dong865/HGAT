"""Paired closed-loop evaluation for an optional V3 mode-demand head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_v3_hybrid import (
    build_evaluation_env, load_v3_system, parse_scale, run_episode_eval,
)
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler, PROTOCOL_VERSION


SCHEMA = "v3_mode_demand_paired_closed_loop_v1"


def _means(rows):
    return {
        "episodes": len(rows),
        "any_capture_rate": float(np.mean([r["any_capture"] for r in rows])),
        "full_capture_rate": float(np.mean([r["full_capture"] for r in rows])),
        "mean_captured": float(np.mean([r["mean_captured"] for r in rows])),
        "mean_switch_count": float(np.mean([r["switch_count"] for r in rows])),
    }


def summarize_paired(records):
    grouped = {}
    for row in records:
        grouped.setdefault(int(row["seed"]), {})[str(row["method"])] = row
    pairs = [value for value in grouped.values()
             if "base" in value and "mode" in value]
    base = _means([value["base"] for value in pairs])
    mode = _means([value["mode"] for value in pairs])
    keys = ("any_capture_rate", "full_capture_rate", "mean_captured",
            "mean_switch_count")
    return {
        "paired_seeds": len(pairs), "base": base, "mode": mode,
        "mode_minus_base": {key: mode[key] - base[key] for key in keys},
    }


def _load_mode_head(modules, base_path, mode_path, config):
    payload = torch.load(mode_path, map_location=config["device"],
                         weights_only=False)
    if payload.get("architecture_version") != (
            "v3_counterfactual_mode_demand_4v2_v1"):
        raise ValueError("mode checkpoint has incompatible schema")
    if Path(payload.get("base_checkpoint", "")).resolve() != Path(base_path).resolve():
        raise ValueError("mode checkpoint belongs to a different base checkpoint")
    modules["demand_head"].load_state_dict(payload["demand_head"])


def evaluate(base_checkpoint, mode_checkpoint, seeds, scale="4v2",
             mode_prior_weight=0.2, device="cpu", deploy_5x5=True):
    base_modules, base_config = load_v3_system(base_checkpoint, device=device)
    mode_modules, mode_config = load_v3_system(base_checkpoint, device=device)
    _load_mode_head(mode_modules, base_checkpoint, mode_checkpoint, mode_config)
    base_config["mode_prior_weight"] = 0.0
    mode_config["mode_prior_weight"] = float(mode_prior_weight)
    parsed_scale = parse_scale(scale)
    records = []
    for method, modules, config in (
            ("base", base_modules, base_config),
            ("mode", mode_modules, mode_config)):
        env = build_evaluation_env(config, seed=0)
        if deploy_5x5:
            env.scenario_sampler = Deploy5x5Sampler()
        try:
            for seed in seeds:
                row = run_episode_eval(env, {**modules, "config": config},
                                       int(seed), parsed_scale)
                row["method"] = method
                records.append(row)
        finally:
            env.close()
    return {
        "schema": SCHEMA,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "mode_checkpoint": str(Path(mode_checkpoint).resolve()),
        "scenario_protocol": PROTOCOL_VERSION if deploy_5x5 else "default",
        "scale": f"{parsed_scale[0]}v{parsed_scale[1]}",
        "mode_prior_weight": float(mode_prior_weight),
        "records": records, "summary": summarize_paired(records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--mode-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-base", type=int, default=72000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--scale", default="4v2")
    parser.add_argument("--mode-prior-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deploy-5x5", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    result = evaluate(
        args.base_checkpoint, args.mode_checkpoint,
        range(args.seed_base, args.seed_base + args.episodes), args.scale,
        args.mode_prior_weight, args.device, args.deploy_5x5,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
