"""Full-task TD3 training for sparse-demand, value-ranked V3 allocation."""

from __future__ import annotations

from collections import deque
import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from .coalition_value import VALUE_SCHEMA, pairwise_ranking_loss
from .stage_env import StageCoalitionEnv, StageRewardConfig
from .stage_networks import StageFeatureEncoder, SharedActor, SetInvariantCritic
from .traditional_ring_baseline import (
    ContinuousRingGuidanceController, RingGuidanceController,
)
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .training_protocol import DEFAULT_D_CAP, DEFAULT_HOLD_STEPS
from .v2_controller import v2_obs_dim
from .v2_env_adapter import apply_membership
from .v2_training import (build_v2_obs_torch, hard_K_from_assignment,
                          actor_los_auxiliary_loss, actor_ring_auxiliary_loss)
from .v3_candidate_generator import generate_bounded_candidates
from .v3_demand_distillation import (V3_TEACHER_SCHEMA, ScaleAwareCoalitionValueHead,
                                     build_demand_geometry, build_scale_aware_candidate_features,
                                     best_mode_demand_target)
from .v3_dynamic_counterfactual import fork_environment, summarize_branch
from .v3_hybrid_controller import V3HybridAllocator
from .v3_sparse_demand import (SparseDemandHead, capacities_from_sparse_demand,
                               neighbor_capacity_modes)


SCALE_WEIGHTS = ((4, 2, 0.4), (6, 2, 0.3), (6, 3, 0.3))
STRATA = ("full", "partial", "progress", "ordinary")
CHECKPOINT_SCHEMA = "v3_sparse_demand_value_hybrid_v1"
COUNTERFACT_REPLAY_STATES = 2000
NETWORK_KEYS = (
    "encoder", "demand_head", "value_head", "actor", "critic1", "critic2",
    "actor_target", "critic1_target", "critic2_target", "value_head_target",
)


class CounterfactualReplay:
    """Independent replay of whole counterfactual decision-state sample sets.

    Transition-level replay dilutes branched supervision to near zero because
    only a handful of rows carry samples. Storing complete same-state sample
    sets here lets every TD3 update draw fresh ranking/regression targets
    regardless of which transitions were sampled.
    """

    def __init__(self, max_states=COUNTERFACT_REPLAY_STATES, seed=0):
        if max_states < 1:
            raise ValueError("max_states must be positive")
        self.states = deque(maxlen=int(max_states))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.states)

    def add(self, state):
        samples = state.get("samples") if isinstance(state, dict) else None
        if not samples:
            return
        alive = state.get("alive")
        if (not isinstance(alive, (list, tuple)) or not len(alive)
                or any(item["features"].shape[0] != len(alive) for item in samples)):
            raise ValueError("counterfactual state requires an alive mask aligned "
                             "with every sample feature row")
        self.states.append(state)

    def sample(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.states:
            return []
        count = min(int(batch_size), len(self.states))
        indices = self.rng.choice(len(self.states), size=count, replace=False)
        return [self.states[int(index)] for index in indices]


def sample_scale(rng, weights=SCALE_WEIGHTS):
    """Sample one complete task without an environment curriculum."""
    rows = tuple(weights)
    if not rows or any((n, m) not in {(4, 2), (6, 2), (6, 3)} for n, m, _ in rows):
        raise ValueError("scale weights must describe 4v2, 6v2 or 6v3")
    probabilities = np.asarray([w for _, _, w in rows], dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("scale weights must be finite, nonnegative and nonzero")
    n, m, _ = rows[int(rng.choice(len(rows), p=probabilities / probabilities.sum()))]
    return int(n), int(m)


class V3StratifiedReplay:
    """Episode-bounded replay with a uniform floor and exact mixture weights.

    ``fractions`` assigns relative sampling mass to uniform/full/partial/
    progress/ordinary. Empty strata redirect their draws to uniform. Importance
    probabilities use the actual integer draw counts, including that fallback.
    """

    def __init__(self, max_episodes=200, seed=0, progress_threshold=0.05):
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        self.episodes = deque(maxlen=int(max_episodes))
        self.rng = np.random.default_rng(seed)
        self.progress_threshold = float(progress_threshold)

    def __len__(self):
        return sum(len(rows) for _, rows in self.episodes)

    @property
    def strata(self):
        result = {name: [] for name in STRATA}
        offset = 0
        for name, rows in self.episodes:
            result[name].extend(range(offset, offset + len(rows)))
            offset += len(rows)
        return result

    def add_episode(self, transitions, outcome, geometry_progress):
        rows = list(transitions)
        if not rows:
            return
        if isinstance(outcome, dict):
            captured, total = int(outcome["captured"]), int(outcome["num_tasks"])
            stratum = "full" if total > 0 and captured >= total else (
                "partial" if captured > 0 else "ordinary")
        else:
            stratum = {"all_captured": "full"}.get(str(outcome), str(outcome))
            if stratum not in STRATA:
                stratum = "ordinary"
        if stratum == "ordinary" and geometry_progress >= self.progress_threshold:
            stratum = "progress"
        self.episodes.append((stratum, rows))

    def sample(self, batch_size, fractions=None, beta=0.4):
        if batch_size < 1 or len(self) == 0:
            raise ValueError("sampling requires a nonempty replay and positive batch_size")
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be between zero and one")
        fractions = dict(fractions if fractions is not None else {
            "uniform": 0.5, "full": 0.2, "partial": 0.15,
            "progress": 0.1, "ordinary": 0.05,
        })
        if (set(fractions) - {"uniform", *STRATA}
                or any(not np.isfinite(v) or v < 0 for v in fractions.values())):
            raise ValueError("invalid replay fractions")
        total_mass = sum(fractions.values())
        uniform_fraction = max(0.5, fractions.get("uniform", 0) / (total_mass or 1))
        counts = {"uniform": int(math.ceil(batch_size * uniform_fraction))}
        remaining = batch_size - counts["uniform"]
        masses = np.asarray([fractions.get(name, 0) for name in STRATA])
        if masses.sum() > 0:
            expected = remaining * masses / masses.sum()
            allocated = np.floor(expected).astype(int)
            for i in np.argsort(-(expected - allocated), kind="stable")[:remaining - allocated.sum()]:
                allocated[i] += 1
            counts.update(zip(STRATA, allocated.tolist()))
        else:
            counts["uniform"] += remaining
        strata = self.strata
        for name in STRATA:
            if not strata[name]:
                counts["uniform"] += counts.get(name, 0)
                counts[name] = 0
        size = len(self)
        probabilities = np.full(size, counts["uniform"] / (batch_size * size))
        indices, sources = [], []
        for source, count in counts.items():
            if count == 0:
                continue
            pool = np.arange(size) if source == "uniform" else np.asarray(strata[source])
            if source != "uniform":
                probabilities[pool] += count / (batch_size * len(pool))
            indices.extend(self.rng.choice(pool, size=count, replace=True).tolist())
            sources.extend([source] * count)
        order = self.rng.permutation(batch_size)
        indices = np.asarray(indices, dtype=np.int64)[order]
        selected_probabilities = probabilities[indices]
        weights = (size * selected_probabilities) ** (-float(beta))
        weights /= weights.max()
        flat = [row for _, rows in self.episodes for row in rows]
        return {
            "transitions": [flat[i] for i in indices], "indices": indices,
            "draw_sources": np.asarray(sources)[order],
            "probabilities": selected_probabilities,
            "importance_weights": weights.astype(np.float32),
        }


def build_checkpoint_payload(modules, optimizers, config, episode=0,
                             update_step=0, metrics=None, metadata=None):
    """Snapshot complete training state under the isolated V3 schema."""
    payload = {
        "architecture_version": CHECKPOINT_SCHEMA,
        "demand_normalizer": "masked_sparsemax",
        "episode": int(episode), "update_step": int(update_step),
        "config": copy.deepcopy(config),
        "metrics": copy.deepcopy(metrics or {}),
        "metadata": copy.deepcopy(metadata or {}),
        "optimizers": {key: copy.deepcopy(value.state_dict())
                       for key, value in optimizers.items()},
        "st_hgat": copy.deepcopy(modules["encoder"].st_hgat.state_dict()),
        "relation_head": copy.deepcopy(modules["encoder"].relation_head.state_dict()),
    }
    for key in ("max_capacity_modes", "max_alliance_candidates", "scale_weights",
                "reward_config", "gate_config"):
        payload[key] = copy.deepcopy(config[key])
    payload.update({key: copy.deepcopy(modules[key].state_dict()) for key in NETWORK_KEYS})
    return payload


def atomic_save_checkpoint(payload, path):
    """Commit a checkpoint only after the complete temporary file is flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_state(module, state, name):
    expected = module.state_dict()
    if set(expected) != set(state) or any(
            not isinstance(state[key], torch.Tensor) or expected[key].shape != state[key].shape
            for key in expected):
        raise ValueError(f"incompatible {name} weights")


def load_v3_hybrid_checkpoint(path, modules=None, optimizers=None, map_location="cpu"):
    """Reject incompatible schemas and incomplete state before loading weights."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint schema must be {CHECKPOINT_SCHEMA}")
    if payload.get("demand_normalizer") != "masked_sparsemax":
        raise ValueError("checkpoint demand_normalizer must be masked_sparsemax")
    required = {*NETWORK_KEYS, "st_hgat", "relation_head", "optimizers", "config",
                "max_capacity_modes", "max_alliance_candidates", "scale_weights",
                "reward_config", "gate_config", "metrics", "metadata", "episode", "update_step"}
    if required - payload.keys():
        raise ValueError(f"checkpoint is missing {sorted(required - payload.keys())}")
    for name, module in (modules or {}).items():
        if name not in NETWORK_KEYS:
            raise ValueError(f"unknown network {name}")
        if name == "demand_head":
            expected = module.state_dict()
            state = payload[name]
            missing = set(expected) - set(state)
            if (set(state) - set(expected)
                    or any(not key.startswith("mode_net.") for key in missing)
                    or any(not isinstance(value, torch.Tensor)
                           or expected[key].shape != value.shape
                           for key, value in state.items())):
                raise ValueError(f"incompatible {name} weights")
        else:
            _validate_state(module, payload[name], name)
    for name, optimizer in (optimizers or {}).items():
        state = payload["optimizers"].get(name)
        if (state is None or len(state.get("param_groups", [])) != len(optimizer.param_groups)
                or any(len(left["params"]) != len(right["params"])
                       for left, right in zip(state["param_groups"], optimizer.param_groups))):
            raise ValueError(f"incompatible optimizer {name}")
    for name, module in (modules or {}).items():
        module.load_state_dict(payload[name], strict=(name != "demand_head"))
    for name, optimizer in (optimizers or {}).items():
        optimizer.load_state_dict(payload["optimizers"][name])
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/v3_hybrid_training")
    parser.add_argument("--init-policy-from")
    parser.add_argument("--init-value-from")
    parser.add_argument("--init-v3-from")
    parser.add_argument("--device", default="cpu")
    integers = {
        "episodes": 1000, "max-steps": 2000, "seed": 37, "eval-seed": None,
        "eval-interval": 20, "eval-episodes": 5, "hidden-dim": None,
        "actor-hidden": None, "head-width": 64, "future-steps": None,
        "num-obstacles": None, "hold-steps": None, "batch-size": 64,
        "warmup-transitions": 256, "replay-episodes": 200,
        "updates-per-episode": 4, "policy-delay": 2, "assign-interval": 10,
        "max-capacity-modes": 6, "max-alliance-candidates": 32,
        "min-dwell-steps": 20, "torch-threads": 1,
        "d-cap-anneal-episodes": 500,
        "counterfact-h": 0, "counterfact-per-episode": 1, "counterfact-modes": 3,
        "counterfact-batch": 4,
    }
    floats = {
        "map-size": None, "d-cap": None, "gamma": 0.999, "tau": 0.005,
        "d-cap-final": None, "ring-radius": None,
        "actor-lr": 3e-4, "critic-lr": 3e-4, "demand-lr": 3e-4,
        "value-lr": 3e-4, "reward-scale": 0.1, "exploration-noise": 0.15,
        "target-policy-noise": 0.2, "target-noise-clip": 0.5,
        "actor-los-weight": 0.0, "actor-ring-weight": 0.0,
        "demand-weight": 0.1, "switch-weight": 0.5,
        "improvement-margin": 0.1, "distill-temperature": 0.2,
        "value-consistency-weight": 0.1, "rank-weight": 0.1,
        "importance-beta": 0.4, "closure-weight": 0.5,
        "angular-closure-weight": 0.3, "ring-weight": 0.1,
        "freeze-distance-ratio": 0.8, "freeze-release-ratio": 1.15,
        "infeasible-mode-penalty": 0.0,
        "infeasible-mode-warmup-fraction": 0.5,
        "capture-urgency-distance": 1.5,
        "counterfact-rank-weight": 1.0, "counterfact-reg-weight": 0.5,
        "counterfact-margin": 0.1,
    }
    for name, default in integers.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in floats.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--scale-weights", nargs=3, type=float, default=(0.4, 0.3, 0.3),
                        metavar=("4V2", "6V2", "6V3"))
    parser.add_argument("--freeze-policy", action="store_true",
                        help="keep Actor, Critics and their targets fixed")
    parser.add_argument("--freeze-demand", action="store_true",
                        help="keep the sparse demand head fixed")
    parser.add_argument("--scenario-protocol", choices=("stage", "deploy5x5"),
                        default="stage")
    return parser


def config_from_args(args):
    config = vars(args).copy()
    source = (torch.load(config["init_policy_from"], map_location="cpu", weights_only=False)
              if config["init_policy_from"] else {})
    v3_config = {}
    if config["init_v3_from"]:
        # The strict loader rejects foreign or incomplete schemas before any
        # default is taken from the checkpoint.
        v3_config = load_v3_hybrid_checkpoint(config["init_v3_from"])["config"]
    config["warm_start_v3_from"] = (str(Path(config["init_v3_from"]).resolve())
                                    if config["init_v3_from"] else None)
    defaults = {"hidden_dim": 32, "actor_hidden": 64, "future_steps": 10,
                "num_obstacles": 10, "map_size": 8.0,
                "d_cap": DEFAULT_D_CAP, "hold_steps": DEFAULT_HOLD_STEPS}
    for name, default in defaults.items():
        if config[name] is None:
            if name in v3_config and v3_config[name] is not None:
                config[name] = v3_config[name]
            elif name == "actor_hidden":
                actor_state = source.get("actor")
                if isinstance(actor_state, dict) and "net.0.weight" in actor_state:
                    config[name] = int(actor_state["net.0.weight"].shape[0])
                elif config["init_policy_from"]:
                    raise ValueError(
                        "policy warm-start source is missing the actor network")
                else:
                    config[name] = default
            else:
                config[name] = source.get(name, default)
    if config["d_cap_anneal_episodes"] <= 0:
        raise ValueError("d-cap anneal episodes must be positive")
    if config["d_cap_final"] is not None:
        # config["d_cap"] still holds the starting value here; it is only
        # annealed per-episode inside run_training, never mutated in config.
        if config["d_cap_final"] <= 0:
            raise ValueError("d-cap final must be positive")
        if config["d_cap_final"] >= config["d_cap"]:
            raise ValueError("d-cap final must be below the starting d-cap")
    config["scale_weights"] = tuple((n, m, float(w)) for (n, m, _), w in
                                    zip(SCALE_WEIGHTS, config["scale_weights"]))
    sample_scale(np.random.default_rng(0), config["scale_weights"])
    reward_overrides = {}
    if config["ring_radius"] is not None:
        reward_overrides["ring_radius"] = config["ring_radius"]
    config["reward_config"] = StageRewardConfig(
        closure_weight=config["closure_weight"],
        angular_closure_weight=config["angular_closure_weight"],
        ring_weight=config["ring_weight"],
        **reward_overrides,
    ).to_dict()
    config["ring_radius"] = config["reward_config"]["ring_radius"]
    config["gate_config"] = {name: config[name] for name in
                             ("min_dwell_steps", "improvement_margin")}
    config["counterfact_config"] = {
        "h": config["counterfact_h"],
        "per_episode": config["counterfact_per_episode"],
        "modes": config["counterfact_modes"],
        "batch": config["counterfact_batch"],
        "rank_weight": config["counterfact_rank_weight"],
        "reg_weight": config["counterfact_reg_weight"],
        "margin": config["counterfact_margin"],
    }
    if (config["counterfact_h"] < 0 or config["counterfact_per_episode"] < 1
            or config["counterfact_modes"] < 1 or config["counterfact_batch"] < 1):
        raise ValueError(
            "counterfactual horizon must be nonnegative and per-episode states, "
            "mode count and batch size must be positive")
    for name in ("counterfact_rank_weight", "counterfact_reg_weight",
                 "counterfact_margin"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError("counterfactual loss weights and margin must be "
                             "nonnegative and finite")
    config["allocation_config"] = {name: config[name] for name in
        ("max_capacity_modes", "max_alliance_candidates", "assign_interval",
         "demand_weight", "switch_weight", "infeasible_mode_penalty",
         "infeasible_mode_warmup_fraction", "capture_urgency_distance")}
    config["capture_size"] = 3
    config["encoder_frozen"] = True
    if config["eval_seed"] is None:
        config["eval_seed"] = config["seed"] + 1_000_000_000
    positive = ("episodes", "max_steps", "eval_interval", "eval_episodes", "hidden_dim",
                "actor_hidden", "head_width", "future_steps", "batch_size", "replay_episodes",
                "policy_delay", "assign_interval", "torch_threads", "map_size", "d_cap",
                "hold_steps", "distill_temperature", "max_capacity_modes")
    if any(config[name] <= 0 for name in positive):
        raise ValueError("episode, model, candidate and interval sizes must be positive")
    if config["max_alliance_candidates"] < config["max_capacity_modes"]:
        raise ValueError("candidate limit must cover all capacity modes")
    if config["num_obstacles"] < 0 or config["updates_per_episode"] < 0 or config["seed"] < 0:
        raise ValueError("obstacle, update and seed counts cannot be negative")
    if not 0 <= config["gamma"] <= 1 or not 0 < config["tau"] <= 1:
        raise ValueError("invalid TD3 discount or target update rate")
    if (not 0 < config["freeze_distance_ratio"] <= 1
            or config["freeze_release_ratio"] < 1):
        raise ValueError("invalid capture-freeze ratios")
    if (not np.isfinite(config["infeasible_mode_penalty"])
            or config["infeasible_mode_penalty"] < 0
            or not 0 <= config["infeasible_mode_warmup_fraction"] <= 1
            or not np.isfinite(config["capture_urgency_distance"])
            or config["capture_urgency_distance"] <= 0):
        raise ValueError("invalid capture-feasibility configuration")
    for seeds in evaluation_seeds(config).values():
        if any(config["seed"] <= seed < config["seed"] + config["episodes"] for seed in seeds):
            raise ValueError("evaluation seeds overlap training seeds")
    return config


def evaluation_seeds(config):
    count = int(config["eval_episodes"])
    if not 0 < count < 1_000_000 or config["eval_seed"] < 0:
        raise ValueError("invalid evaluation seed range")
    return {(n, m): tuple(config["eval_seed"] + i * 1_000_000 + j for j in range(count))
            for i, (n, m, _) in enumerate(SCALE_WEIGHTS)}


def _warm_start_value(module, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    schema = payload.get("teacher_schema", payload.get("schema", payload.get("architecture_version")))
    if schema == VALUE_SCHEMA:
        if payload.get("supervision") != "geometric_proxy_not_rollout_return":
            raise ValueError("proxy weights must be explicitly marked as geometric supervision")
        state = copy.deepcopy(payload["value_head"])
        # The existing proxy lacks six scale features. Zero columns preserve
        # its exact initial function; actual rollout updates can learn them.
        state["task_encoder.0.weight"] = F.pad(state["task_encoder.0.weight"], (0, 6))
    elif schema == V3_TEACHER_SCHEMA:
        state = payload.get("teacher", payload.get("value_head"))
    else:
        raise ValueError("value warm-start schema must explicitly identify proxy or scale-aware teacher")
    if state is None:
        raise ValueError("marked value checkpoint has no value weights")
    _validate_state(module, state, "value warm start")
    module.load_state_dict(state)
    return {"path": str(Path(path).resolve()), "schema": schema, "warm_start_only": True,
            "supervision": "initialization_only_then_actual_replay_outcomes"}


def _optimizer_groups_match(state, optimizer):
    groups = state.get("param_groups", [])
    return (len(groups) == len(optimizer.param_groups)
            and all(len(left["params"]) == len(right["params"])
                    for left, right in zip(groups, optimizer.param_groups)))


def build_system(config):
    """Build frozen graph features and trainable policy, Critics and V3 heads."""
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    hidden, actor_hidden = config["hidden_dim"], config["actor_hidden"]
    encoder = StageFeatureEncoder(
        6, 3, config["num_obstacles"], config["map_size"], hidden,
        config["future_steps"], 10, 9, device,
        use_role_embedding=False, use_stage_embedding=False,
        use_graph_embedding=True, use_ring_slot_features=False,
    )
    encoder._skip_legacy_plan = True
    obs_dim = v2_obs_dim(6, hidden, config["future_steps"])
    modules = {
        "encoder": encoder,
        "demand_head": SparseDemandHead(hidden, width=config["head_width"]),
        "value_head": ScaleAwareCoalitionValueHead(hidden, width=config["head_width"]),
        "actor": SharedActor(obs_dim, actor_hidden),
        "critic1": SetInvariantCritic(obs_dim, actor_hidden),
        "critic2": SetInvariantCritic(obs_dim, actor_hidden),
    }
    modules = {name: module.to(device) for name, module in modules.items()}
    metadata = {"encoder_frozen": True, "policy_warm_start": None, "value_warm_start": None,
                "value_supervision": "actual_replay_outcomes_and_pairwise_ranking",
                "observation_schema": "v2_accepted_assignment",
                "raw_reward_definition": "environment_reward_minus_closure_angular_closure_ring"}
    if config.get("init_policy_from"):
        path = config["init_policy_from"]
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("architecture_version") != "continuous_membership_v2":
            raise ValueError("policy warm start requires compatible V2 schema")
        targets = {"st_hgat": encoder.st_hgat, "relation_head": encoder.relation_head,
                   **{name: modules[name] for name in ("actor", "critic1", "critic2")}}
        for name, module in targets.items():
            _validate_state(module, payload.get(name, {}), name)
        for name, module in targets.items():
            module.load_state_dict(payload[name])
        metadata["policy_warm_start"] = {"path": str(Path(path).resolve()),
                                         "schema": "continuous_membership_v2"}
    if config.get("init_value_from"):
        metadata["value_warm_start"] = _warm_start_value(
            modules["value_head"], config["init_value_from"], device)
    metadata["warm_start_v3_from"] = None
    v3_payload = None
    if config.get("init_v3_from"):
        path = config["init_v3_from"]
        v3_payload = load_v3_hybrid_checkpoint(
            path, map_location=device,
            modules={name: modules[name] for name in
                     ("encoder", "demand_head", "value_head", "actor", "critic1", "critic2")})
        metadata["warm_start_v3_from"] = {"path": str(Path(path).resolve()),
                                          "optimizer_state": "reset",
                                          "d_cap": config["d_cap"]}
    encoder.requires_grad_(False).eval()
    for name in ("actor", "critic1", "critic2", "value_head"):
        modules[f"{name}_target"] = copy.deepcopy(modules[name]).requires_grad_(False).eval()
    optimizers = {
        "actor": torch.optim.Adam(modules["actor"].parameters(), lr=config["actor_lr"]),
        "critics": torch.optim.Adam(list(modules["critic1"].parameters()) +
                                     list(modules["critic2"].parameters()), lr=config["critic_lr"]),
        # Preserve optimizer compatibility with V3 checkpoints predating the
        # optional mode branch. Counterfactual mode training has its own
        # isolated optimizer and checkpoint.
        "demand": torch.optim.Adam(modules["demand_head"].net.parameters(),
                                     lr=config["demand_lr"]),
        "value": torch.optim.Adam(modules["value_head"].parameters(), lr=config["value_lr"]),
    }
    if v3_payload is not None:
        saved_optimizers = v3_payload.get("optimizers") or {}
        if all(name in saved_optimizers and
               _optimizer_groups_match(saved_optimizers[name], optimizer)
               for name, optimizer in optimizers.items()):
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(saved_optimizers[name])
            metadata["warm_start_v3_from"]["optimizer_state"] = "loaded"
    return modules, optimizers, metadata


def make_environment(config, d_cap=None):
    environment = StageCoalitionEnv(
        length=config["map_size"], num_obstacle=config["num_obstacles"],
        max_pursuers=6, max_evaders=3,
        d_cap=config["d_cap"] if d_cap is None else d_cap,
        hold_steps=config["hold_steps"], seed=config["seed"],
        reward_config=StageRewardConfig(**config["reward_config"]),
    )
    if config.get("scenario_protocol", "stage") == "deploy5x5":
        environment.scenario_sampler = Deploy5x5Sampler()
        environment.v_max_e = 0.35
    return environment


class HybridPolicy:
    """Shared Actor with periodic allocation and immediate dead-task repair."""

    def __init__(self, modules, config):
        self.modules, self.config = modules, config
        self.allocator = V3HybridAllocator(
            modules["demand_head"], modules["value_head"], config["map_size"],
            capture_size=3, mode_limit=config["max_capacity_modes"],
            max_candidates=config["max_alliance_candidates"],
            demand_weight=config["demand_weight"], switch_weight=config["switch_weight"],
            freeze_distance=(config.get("freeze_distance_ratio", 0.8)
                             * config.get("d_cap", DEFAULT_D_CAP)),
            freeze_release_ratio=config.get("freeze_release_ratio", 1.15),
            mode_prior_weight=config.get("mode_prior_weight", 0.0),
            infeasible_mode_penalty=config.get("infeasible_mode_penalty", 0.0),
            infeasible_mode_warmup_fraction=config.get(
                "infeasible_mode_warmup_fraction", 0.5),
            **config["gate_config"],
        )
        self.reset()

    def reset(self):
        self.modules["encoder"].reset()
        self.allocator.reset()
        self.last_allocation = None
        self.first_single_task_step = None
        self.post_capture_ring = None

    @torch.no_grad()
    def decision(self, env):
        started = time.perf_counter()
        # The simulator's physical step owns capture hold/lifecycle updates.
        # Advancing here as well would double-count the same physical state.
        snapshot = env.snapshot()
        encoded = self.modules["encoder"].encode(snapshot, differentiable=False)
        previous = self.allocator.gate.assignment
        invalid = previous is None or any(not env.task_mask[j] for j in previous)
        allocate = invalid or env.episode_step_count % self.config["assign_interval"] == 0
        if allocate:
            remaining_fraction = max(
                0.0, 1.0 - env.episode_step_count / self.config["max_steps"]
            )
            live_ids = np.flatnonzero(env.task_mask)
            capture_urgent = bool(live_ids.size and np.any(np.linalg.norm(
                env.positions[env.num_pursuers + live_ids, :2]
                - env.evader_goals[live_ids, :2], axis=1,
            ) <= self.config.get("capture_urgency_distance", 1.5)))
            self.last_allocation = self.allocator.allocate(
                encoded, snapshot, env.task_mask, env.episode_step_count, force=invalid,
                hold_counters=env.hold_counters,
                remaining_fraction=remaining_fraction,
                capture_urgent=capture_urgent,
            )
            apply_membership(env, self.last_allocation["assignment"])
        snapshot = env.snapshot()
        assignment = self.last_allocation["assignment"].copy()
        membership = hard_K_from_assignment(assignment, env.task_mask, 6, 3,
                                             encoded["pursuer_embeddings"].device)
        obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
        mask = torch.as_tensor(env.pursuer_mask, dtype=torch.float32, device=obs.device)
        live_count = int(np.count_nonzero(env.task_mask[:env.num_evaders]))
        if live_count == 1 and self.first_single_task_step is None:
            self.first_single_task_step = int(env.episode_step_count)
        elif live_count != 1:
            self.first_single_task_step = None
            self.post_capture_ring = None
        ring_delay = self.config.get("post_capture_ring_delay")
        ring_active = (ring_delay is not None and should_use_post_capture_ring(
            env.task_mask[:env.num_evaders], self.first_single_task_step,
            env.episode_step_count, env.time_step, ring_delay,
        ))
        if ring_active:
            if self.post_capture_ring is None:
                self.post_capture_ring = build_post_capture_guidance(
                    env, self.config.get("post_capture_guidance", "slots"),
                )
                self.post_capture_ring.reset(env)
            self.post_capture_ring.assign = assignment.copy()
            execution_actor = select_execution_actor(self.modules, env.task_mask)
            actor_actions = (execution_actor(obs) * mask.unsqueeze(-1)).cpu().numpy()
            ring_actions = np.clip(self.post_capture_ring.act(env), -1.0, 1.0)
            elapsed = ((env.episode_step_count - self.first_single_task_step)
                       * env.time_step)
            actions = actor_actions.copy()
            actions[:env.num_pursuers] = blend_post_capture_ring_actions(
                actor_actions[:env.num_pursuers], ring_actions,
                elapsed_seconds=elapsed, delay_seconds=ring_delay,
                transition_seconds=self.config.get(
                    "post_capture_ring_transition", 0.0),
            )
        else:
            execution_actor = select_execution_actor(self.modules, env.task_mask)
            actions = (execution_actor(obs) * mask.unsqueeze(-1)).cpu().numpy()
        if obs.device.type == "cuda":
            torch.cuda.synchronize(obs.device)
        return {"obs": obs, "mask": mask, "actions": actions,
                "encoded": encoded, "snapshot": snapshot, "assignment": assignment,
                "allocation": self.last_allocation, "allocated": bool(allocate),
                "post_capture_ring_active": bool(ring_active),
                "inference_ms": 1000 * (time.perf_counter() - started)}


def select_execution_actor(modules, task_mask):
    """Use an optional finisher only after capture leaves one live task."""
    alive = np.asarray(task_mask, dtype=bool)
    count = int(alive.sum())
    if count < 1:
        raise ValueError("execution requires at least one live task")
    if count == 1 and "post_capture_actor" in modules:
        return modules["post_capture_actor"]
    return modules["actor"]


def build_post_capture_guidance(env, mode="slots"):
    controllers = {
        "slots": RingGuidanceController,
        "continuous": ContinuousRingGuidanceController,
    }
    try:
        controller = controllers[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unknown post-capture guidance mode {mode!r}") from exc
    return controller(d_cap=env.d_cap, v_max=env.v_max,
                      n_p=env.num_pursuers)


def should_use_post_capture_ring(task_mask, first_single_task_step,
                                 current_step, time_step, delay_seconds):
    alive = np.asarray(task_mask, dtype=bool)
    if int(alive.sum()) != 1 or first_single_task_step is None:
        return False
    values = (current_step, first_single_task_step)
    if (any(int(value) != value or value < 0 for value in values)
            or not np.isfinite(time_step) or time_step <= 0
            or not np.isfinite(delay_seconds) or delay_seconds < 0):
        raise ValueError("invalid post-capture ring timing")
    return (int(current_step) - int(first_single_task_step)) * float(time_step) >= float(delay_seconds)


def blend_post_capture_ring_actions(actor_actions, ring_actions,
                                    elapsed_seconds, delay_seconds,
                                    transition_seconds):
    """Continuously hand control from the learned actor to ring guidance."""
    actor = np.asarray(actor_actions, dtype=np.float32)
    ring = np.asarray(ring_actions, dtype=np.float32)
    if actor.shape != ring.shape:
        raise ValueError("actor and ring actions must have the same shape")
    values = (elapsed_seconds, delay_seconds, transition_seconds)
    if (not all(np.isfinite(value) for value in values)
            or delay_seconds < 0 or transition_seconds < 0):
        raise ValueError("invalid post-capture ring blend timing")
    if transition_seconds == 0:
        alpha = float(elapsed_seconds >= delay_seconds)
    else:
        alpha = np.clip(
            (elapsed_seconds - delay_seconds) / transition_seconds, 0.0, 1.0,
        )
    return ((1.0 - alpha) * actor + alpha * ring).astype(np.float32, copy=False)


def _geometry_progress(snapshot):
    n, m = snapshot["num_pursuers"], snapshot["num_evaders"]
    positions = snapshot["positions"]
    geometry = build_demand_geometry(positions[:n], positions[n:n + m], snapshot["length"])
    alive = np.asarray(snapshot["task_mask"][:m], dtype=bool)
    return float((1 - geometry[alive, 3]).mean()) if alive.any() else 1.0


@torch.no_grad()
def _rollout_locked_branch(env, encoder, actor, first_encoded, assignment,
                           horizon_steps, max_steps, gamma):
    """Fork at a decision state and hold one alliance for H steps.

    Mirrors ``v3_dynamic_counterfactual.rollout_forced_assignment`` (kept
    read-only) but discounts raw reward: the closure/angular/ring shaping
    terms are subtracted per step so G_H matches the value head's existing
    raw-return supervision. The encoder temporal state is deep-copied at the
    fork exactly like the reference rollout, keeping the GRU state aligned.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    branch = fork_environment(env)
    branch_encoder = copy.deepcopy(encoder)
    initial = np.asarray(assignment, dtype=np.int64).copy()
    if initial.shape != (branch.num_pursuers,):
        raise ValueError("assignment must contain every active pursuer")
    captured_before = int(branch.captured[:branch.num_evaders].sum())
    rewards, first_capture_step = [], None
    for offset in range(int(horizon_steps)):
        active = np.flatnonzero(branch.task_mask[:branch.num_evaders])
        if len(active) == 0 or branch.done:
            break
        repaired = initial.copy()
        repaired[~np.isin(repaired, active)] = int(active[0])
        apply_membership(branch, repaired)
        snapshot = branch.snapshot()
        encoded = first_encoded if offset == 0 else branch_encoder.encode(
            snapshot, differentiable=False
        )
        membership = hard_K_from_assignment(
            repaired, branch.task_mask, 6, 3,
            encoded["pursuer_embeddings"].device,
        )
        obs = build_v2_obs_torch(encoded, membership, snapshot, 6, 3)
        mask = torch.as_tensor(
            branch.pursuer_mask, dtype=torch.float32, device=obs.device,
        )
        actions = (actor(obs) * mask.unsqueeze(-1)).cpu().numpy()
        reward, done, info = branch.step(actions, max_steps=max_steps)
        shaping = sum(float(info["reward_components"].get(name, 0.0))
                      for name in ("closure", "angular_closure", "ring"))
        rewards.append(float(reward) - shaping)
        if first_capture_step is None and info.get("capture_events"):
            first_capture_step = offset + 1
        if done:
            break
    summary = summarize_branch(
        rewards=rewards, captured_before=captured_before,
        captured_after=int(branch.captured[:branch.num_evaders].sum()),
        elapsed_steps=len(rewards), time_step=branch.time_step,
        first_capture_step=first_capture_step, gamma=gamma,
    )
    summary.update({"steps": len(rewards), "assignment": initial.tolist(),
                    "outcome": str(branch.outcome)})
    return summary


def _counterfact_branch_samples(modules, config, context, counterfact):
    """Branch the top-scored distinct alliances of one recorded decision state."""
    snapshot = context["snapshot"]
    n, m = snapshot["num_pursuers"], snapshot["num_evaders"]
    encoded = context["encoded"]
    positions = snapshot["positions"]
    alive = np.asarray(snapshot["task_mask"][:m], dtype=bool)
    candidates = np.asarray(context["candidates"], dtype=np.int64)
    scores = np.asarray(context["scores"], dtype=np.float64)
    if candidates.ndim != 2 or scores.ndim != 1 or len(candidates) != len(scores):
        raise ValueError("counterfactual candidates and scores must be aligned")
    # Select the best member permutation inside each capacity-count mode
    # before ranking modes.  Selecting the globally highest assignments lets
    # several 2+2 permutations consume the whole branch budget and makes it
    # impossible to learn whether 3+1 or 1+3 is actually better.
    best_by_mode = {}
    for index in np.argsort(-scores, kind="stable"):
        mode = tuple(np.bincount(candidates[index], minlength=m).tolist())
        best_by_mode.setdefault(mode, int(index))
    chosen = sorted(
        best_by_mode.values(), key=lambda index: (-scores[index], index)
    )[:int(counterfact["modes"])]
    pursuers = encoded["pursuer_embeddings"][:n]
    tasks = encoded["task_embeddings"][:m]
    samples = []
    for index in chosen:
        candidate = candidates[index]
        summary = _rollout_locked_branch(
            context["env"], modules["encoder"], modules["actor"], encoded,
            candidate, int(counterfact["h"]), int(config["max_steps"]),
            float(config["gamma"]),
        )
        features = build_scale_aware_candidate_features(
            pursuers, tasks, positions[:n], positions[n:n + m], candidate,
            int(config["capture_size"]),
        )
        samples.append({
            "features": features.detach().cpu(),
            "discounted_return": float(summary["discounted_return"]),
            "new_captures": int(summary["new_captures"]),
            "steps": int(summary["steps"]),
            "outcome": str(summary["outcome"]),
            "assignment": [int(value) for value in candidate.tolist()],
            "capacity_mode": list(np.bincount(candidate, minlength=m)),
        })
    return samples, alive.tolist()


def counterfact_pairwise_value_loss(predictions, returns, margin):
    """Same-state hinge ranking by the sign of G_H plus Huber regression."""
    predictions = torch.as_tensor(predictions)
    returns = torch.as_tensor(returns, dtype=predictions.dtype,
                              device=predictions.device)
    if (predictions.ndim != 1 or returns.ndim != 1
            or predictions.numel() < 1 or predictions.shape != returns.shape):
        raise ValueError("predictions and returns must be aligned one-dimensional tensors")
    if not torch.isfinite(predictions).all() or not torch.isfinite(returns).all():
        raise ValueError("predictions and returns must be finite")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("rank margin must be nonnegative and finite")
    hinges = []
    for left in range(predictions.numel()):
        for right in range(left + 1, predictions.numel()):
            sign = torch.sign(returns[left] - returns[right])
            if sign != 0:
                hinges.append(torch.relu(
                    predictions.new_tensor(float(margin))
                    - sign * (predictions[left] - predictions[right])
                ))
    ranking = torch.stack(hinges).mean() if hinges else predictions.sum() * 0.0
    return ranking, F.smooth_l1_loss(predictions, returns)


def _reservoir_offer(items, item, seen_count, limit, rng):
    """Offer one item to a fixed-size reservoir without leaving empty slots."""
    if int(limit) < 1 or int(seen_count) < 1:
        raise ValueError("reservoir limit and seen count must be positive")
    if len(items) < int(limit):
        items.append(item() if callable(item) else item)
        return
    slot = int(rng.integers(0, int(seen_count)))
    if slot < int(limit):
        items[slot] = item() if callable(item) else item


def collect_episode(modules, config, scale, seed, exploration_rng=None, d_cap=None,
                    counterfact=None):
    """Collect a whole task, including all subsequent captures after the first.

    With ``counterfact`` (the ``counterfact_config`` group), up to
    ``per_episode`` decision states that ran an allocation are uniformly
    reservoir-sampled; after the episode the trainer forks each recorded
    state and rolls the top-scored ``modes`` alliances for ``h`` steps,
    attaching same-state ranking/regression samples to the matching row so
    the training loop can store them in the ``CounterfactualReplay``.
    Branch rollouts run on independent environment copies, so the main
    episode trajectory and metrics are untouched.
    """
    env = make_environment(config, d_cap)
    env.reset(num_pursuers=scale[0], num_evaders=scale[1], scenario_seed=seed)
    for module in modules.values():
        module.eval()
    policy = HybridPolicy(modules, config)
    rows, first_capture, initial_geometry = [], None, _geometry_progress(env.snapshot())
    initial_capacity, final_capacity, latency, candidates, zeros, switches = None, None, [], [], [], 0
    fork_contexts = []
    allocate_index = 0
    fork_rng = (np.random.default_rng(int(seed) + 9_000_000)
                if counterfact is not None else None)
    decision = policy.decision(env)
    for step in range(config["max_steps"]):
        if counterfact is not None and decision["allocated"]:
            # Reservoir-sample the requested number of decision states uniformly
            # over the steps that ran an allocation.
            allocate_index += 1
            _reservoir_offer(
                fork_contexts, lambda: {
                    "env": fork_environment(env),
                    "encoded": decision["encoded"],
                    "snapshot": decision["snapshot"],
                    "candidates": decision["allocation"]["candidates"],
                    "scores": decision["allocation"]["unified_scores"]
                                  .detach().cpu().numpy().copy(),
                    "row_index": len(rows),
                }, allocate_index,
                counterfact["per_episode"], fork_rng,
            )
        actions = decision["actions"].copy()
        if exploration_rng is not None:
            actions += exploration_rng.normal(0, config["exploration_noise"], actions.shape)
            actions = np.clip(actions, -1, 1).astype(np.float32)
            actions *= env.pursuer_mask[:, None]
        reward, done, info = env.step(actions, max_steps=config["max_steps"])
        if not np.isfinite(actions).all() or not np.isfinite(reward):
            raise FloatingPointError("nonfinite policy action or environment reward")
        if first_capture is None and info["capture_events"]:
            first_capture = (step + 1) * env.time_step
        allocation = decision["allocation"]
        capacity = allocation["capacity"].detach().cpu().numpy().copy()
        if initial_capacity is None:
            initial_capacity = capacity.tolist()
        final_capacity = capacity.tolist()
        latency.append(decision["inference_ms"])
        if decision["allocated"]:
            candidates.append(allocation["candidate_count"])
            active = decision["snapshot"]["task_mask"][:scale[1]]
            zeros.append(float((allocation["demand"].cpu().numpy()[active] == 0).mean()))
            switches += int(allocation["switched"] and step > 0)
        shaping = {name: float(info["reward_components"].get(name, 0))
                   for name in ("closure", "angular_closure", "ring")}
        row = {
            "obs": decision["obs"].cpu().clone(), "mask": decision["mask"].cpu().clone(),
            "actions": actions, "reward": float(reward),
            "raw_reward": float(reward - sum(shaping.values())),
            "shaping_components": shaping, "reward_components": info["reward_components"],
            "done": bool(done), "snapshot": decision["snapshot"],
            "assignment": decision["assignment"].copy(), "capacity": capacity,
            "encoded": {key: decision["encoded"][key].detach().cpu().clone() for key in
                        ("pursuer_embeddings", "task_embeddings", "relation_costs")},
        }
        if done:
            row["next_obs"] = torch.zeros_like(row["obs"])
            row["next_mask"] = torch.zeros_like(row["mask"])
        else:
            decision = policy.decision(env)
            row["next_obs"] = decision["obs"].cpu().clone()
            row["next_mask"] = decision["mask"].cpu().clone()
        rows.append(row)
        if done:
            break
    if fork_contexts:
        started = time.perf_counter()
        for context in fork_contexts:
            samples, alive = _counterfact_branch_samples(
                modules, config, context, counterfact)
            rows[context["row_index"]]["counterfact"] = {
                "samples": samples, "alive": alive,
            }
        branch_ms = 1000.0 * (time.perf_counter() - started)
    captured = int(env.captured[:scale[1]].sum())
    raw_return, environment_return = sum(r["raw_reward"] for r in rows), sum(r["reward"] for r in rows)
    metrics = {
        "scale": f"{scale[0]}v{scale[1]}", "seed": int(seed), "steps": len(rows),
        "num_tasks": scale[1], "captured": captured, "full_capture": float(captured == scale[1]),
        "any_capture": float(captured > 0), "mean_captured": float(captured),
        "first_capture_time": first_capture, "raw_return": raw_return,
        "environment_return": environment_return, "outcome": env.outcome,
        "geometry_progress": max(0.0, _geometry_progress(env.snapshot()) - initial_geometry),
        "demand_zero_fraction": float(np.mean(zeros)), "mean_candidate_count": float(np.mean(candidates)),
        "max_candidate_count": max(candidates), "switches": switches,
        "switch_rate": switches / max(1, len(candidates) - 1),
        "inference_ms": float(np.mean(latency)), "allocation_calls": len(candidates),
        "initial_capacity": initial_capacity, "final_capacity": final_capacity,
    }
    if fork_contexts:
        metrics["counterfact_ms"] = branch_ms
    # Supervision describes outcomes of the actually executed assignment. It
    # does not invent returns for unexecuted candidates.
    return_to_go = 0.0
    horizon = config["max_steps"] * env.time_step
    for row in reversed(rows):
        return_to_go = row["raw_reward"] + config["gamma"] * return_to_go
        row["return_to_go"] = return_to_go
        row["value_target"] = (
            4 * metrics["full_capture"] + 2 * metrics["any_capture"] + captured / scale[1]
            + (0.5 * (1 - first_capture / horizon) if first_capture is not None else 0.0)
            - float(env.outcome == "evader_goal_success")
            + 0.1 * math.tanh(return_to_go * config["reward_scale"])
        )
    env.close()
    return rows, metrics


def _head_losses(modules, rows, config, device, counterfact_replay=None):
    demand_losses, value_predictions, value_targets, scales = [], [], [], []
    for row in rows:
        snapshot = row["snapshot"]
        n, m = snapshot["num_pursuers"], snapshot["num_evaders"]
        encoded = {key: value.to(device) for key, value in row["encoded"].items()}
        pursuers, tasks = encoded["pursuer_embeddings"][:n], encoded["task_embeddings"][:m]
        positions = snapshot["positions"]
        alive = torch.as_tensor(snapshot["task_mask"][:m], dtype=torch.bool, device=device)
        geometry = build_demand_geometry(positions[:n], positions[n:n + m], snapshot["length"]).to(device)
        demand, demand_logits = modules["demand_head"](
            tasks, alive, n, 3, geometry, return_logits=True,
        )
        capacity = capacities_from_sparse_demand(demand, alive, n)
        with torch.no_grad():
            executed_mode = tuple(np.bincount(
                row["assignment"], minlength=m
            ).tolist())
            modes = neighbor_capacity_modes(
                capacity, alive, n, config["max_capacity_modes"],
                current_mode=executed_mode,
            )
            candidates = generate_bounded_candidates(
                encoded["relation_costs"][:n, :m], modes, alive,
                current_assignment=row["assignment"], max_candidates=config["max_alliance_candidates"],
            )
            features = torch.stack([build_scale_aware_candidate_features(
                pursuers, tasks, positions[:n], positions[n:n + m], candidate, 3,
            ) for candidate in candidates])
            values = modules["value_head_target"](features, alive)
            teacher_demand, best_mode = best_mode_demand_target(
                values, candidates, n, m,
            )
            selected_features = build_scale_aware_candidate_features(
                pursuers, tasks, positions[:n], positions[n:n + m], row["assignment"], 3)
        log_probs = F.log_softmax(demand_logits[alive], dim=0)
        logit_loss = -(teacher_demand[alive] * log_probs).sum()
        best_counts = torch.as_tensor(
            best_mode, dtype=capacity.dtype, device=capacity.device,
        )
        consistency = F.mse_loss(capacity, best_counts)
        demand_losses.append(
            logit_loss + config["value_consistency_weight"] * consistency
        )
        value_predictions.append(modules["value_head"](selected_features, alive))
        value_targets.append(row["value_target"])
        scales.append((n, m))
    prediction = torch.stack(value_predictions)
    targets = torch.as_tensor(value_targets, dtype=prediction.dtype, device=device)
    ranking = []
    for scale in sorted(set(scales)):
        index = [i for i, value in enumerate(scales) if value == scale]
        ranking.append(pairwise_ranking_loss(prediction[index], targets[index]))
    rank_loss = torch.stack(ranking).mean()
    value_loss = (F.smooth_l1_loss(prediction, targets)
                  + config["rank_weight"] * rank_loss)
    # Counterfactual supervision comes from its own replay so every update
    # sees branched samples; ranking pairs stay within one decision state.
    # Sampling whole states from the transition batch diluted this signal to
    # zero because almost no transition row carries samples.
    cf_rank_losses, cf_reg_losses = [], []
    cf_pair_correct, cf_pair_total, cf_distinct_modes = 0, 0, []
    if config.get("counterfact_h", 0) > 0 and counterfact_replay is not None:
        for state in counterfact_replay.sample(config["counterfact_batch"]):
            features = torch.stack(
                [item["features"] for item in state["samples"]]).to(device)
            alive_cf = torch.as_tensor(state["alive"], dtype=torch.bool,
                                       device=device)
            predictions = modules["value_head"](features, alive_cf)
            returns = torch.as_tensor(
                [item["discounted_return"] for item in state["samples"]],
                dtype=predictions.dtype, device=device)
            ranking_cf, regression_cf = counterfact_pairwise_value_loss(
                predictions, returns, config["counterfact_margin"])
            cf_rank_losses.append(ranking_cf)
            cf_reg_losses.append(regression_cf)
            with torch.no_grad():
                for left in range(len(predictions)):
                    for right in range(left + 1, len(predictions)):
                        return_delta = returns[left] - returns[right]
                        if return_delta != 0:
                            cf_pair_total += 1
                            cf_pair_correct += int(
                                (predictions[left] - predictions[right])
                                * return_delta > 0
                            )
                modes = {
                    tuple(item.get("capacity_mode", np.bincount(
                        item["assignment"], minlength=len(state["alive"])
                    ).tolist()))
                    for item in state["samples"]
                }
                cf_distinct_modes.append(len(modes))
    zero = prediction.sum() * 0.0
    cf_rank_loss = torch.stack(cf_rank_losses).mean() if cf_rank_losses else zero
    cf_reg_loss = torch.stack(cf_reg_losses).mean() if cf_reg_losses else zero
    value_loss = (value_loss
                  + config["counterfact_rank_weight"] * cf_rank_loss
                  + config["counterfact_reg_weight"] * cf_reg_loss)
    pair_accuracy = cf_pair_correct / max(1, cf_pair_total)
    distinct_modes = float(np.mean(cf_distinct_modes)) if cf_distinct_modes else 0.0
    return (torch.stack(demand_losses).mean(), value_loss, rank_loss,
            cf_rank_loss, cf_reg_loss, pair_accuracy, distinct_modes)


def _soft_update(target, source, tau):
    with torch.no_grad():
        for destination, origin in zip(target.parameters(), source.parameters()):
            destination.lerp_(origin, tau)


def td3_update(modules, optimizers, batch, config, update_step,
               counterfact_replay=None):
    """Update weighted twin Q losses, delayed Actor and separate V3 heads."""
    rows = batch["transitions"]
    device = next(modules["actor"].parameters()).device
    for name in ("actor", "critic1", "critic2", "value_head", "demand_head"):
        modules[name].train()
    obs, mask, next_obs, next_mask = [torch.stack([row[name] for row in rows]).to(device)
                                      for name in ("obs", "mask", "next_obs", "next_mask")]
    actions = torch.as_tensor(np.stack([row["actions"] for row in rows]), dtype=torch.float32, device=device)
    reward = torch.as_tensor([row["reward"] for row in rows], dtype=torch.float32, device=device)
    done = torch.as_tensor([row["done"] for row in rows], dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch["importance_weights"], dtype=torch.float32, device=device)
    if weights.shape != reward.shape or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("importance weights must be positive, finite and aligned")
    with torch.no_grad():
        next_actions = modules["actor_target"](next_obs)
        noise = (torch.randn_like(next_actions) * config["target_policy_noise"]).clamp(
            -config["target_noise_clip"], config["target_noise_clip"])
        next_actions = (next_actions + noise).clamp(-1, 1) * next_mask.unsqueeze(-1)
        next_q = torch.minimum(modules["critic1_target"](next_obs, next_actions, next_mask),
                               modules["critic2_target"](next_obs, next_actions, next_mask))
        target = config["reward_scale"] * reward + config["gamma"] * (1 - done) * next_q
    critic_loss = sum((weights * (modules[name](obs, actions, mask) - target).square()).mean()
                      for name in ("critic1", "critic2"))
    if not config.get("freeze_policy", False):
        optimizers["critics"].zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(modules["critic1"].parameters()) +
                                       list(modules["critic2"].parameters()), 5.0)
        optimizers["critics"].step()
    actor_updated = (not config.get("freeze_policy", False)
                     and update_step % config["policy_delay"] == 0)
    actor_loss = los = ring = torch.zeros((), device=device)
    if actor_updated:
        modules["critic1"].requires_grad_(False)
        policy_actions = modules["actor"](obs) * mask.unsqueeze(-1)
        actor_loss = -modules["critic1"](obs, policy_actions, mask).mean()
        if config["actor_los_weight"]:
            los = actor_los_auxiliary_loss(obs, policy_actions, mask, 6)
            actor_loss = actor_loss + config["actor_los_weight"] * los
        if config["actor_ring_weight"]:
            ring = actor_ring_auxiliary_loss([row["snapshot"] for row in rows], policy_actions,
                                              mask, 6, config["d_cap"])
            actor_loss = actor_loss + config["actor_ring_weight"] * ring
        optimizers["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(modules["actor"].parameters(), 1.0)
        optimizers["actor"].step()
        modules["critic1"].requires_grad_(True)
        for name in ("actor", "critic1", "critic2"):
            _soft_update(modules[f"{name}_target"], modules[name], config["tau"])
    (demand_loss, value_loss, rank_loss, cf_rank_loss, cf_reg_loss,
     cf_pair_accuracy, cf_distinct_modes) = _head_losses(
         modules, rows, config, device, counterfact_replay=counterfact_replay)
    head_updates = [("value", value_loss, modules["value_head"])]
    if not config.get("freeze_demand", False):
        head_updates.insert(0, ("demand", demand_loss, modules["demand_head"]))
    for name, loss, module in head_updates:
        optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizers[name].step()
    _soft_update(modules["value_head_target"], modules["value_head"], config["tau"])
    metrics = {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
               "demand_loss": float(demand_loss.detach()), "value_loss": float(value_loss.detach()),
               "rank_loss": float(rank_loss.detach()), "los_loss": float(los.detach()),
               "ring_loss": float(ring.detach()), "actor_updated": bool(actor_updated)}
    metrics["policy_frozen"] = bool(config.get("freeze_policy", False))
    metrics["demand_frozen"] = bool(config.get("freeze_demand", False))
    if config.get("counterfact_h", 0) > 0:
        metrics["counterfact_rank_loss"] = float(cf_rank_loss.detach())
        metrics["counterfact_reg_loss"] = float(cf_reg_loss.detach())
        metrics["counterfact_pair_accuracy"] = float(cf_pair_accuracy)
        metrics["counterfact_distinct_modes"] = float(cf_distinct_modes)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("nonfinite V3 training loss")
    return metrics


def selection_key(metrics):
    first = metrics["first_capture_time"]
    return (float(metrics["full_capture"]), float(metrics["any_capture"]),
            float(metrics["mean_captured"]), -float(first) if first is not None else -math.inf,
            float(metrics["raw_return"]))


def summarize_episodes(rows):
    times = [row["first_capture_time"] for row in rows if row["first_capture_time"] is not None]
    names = ("full_capture", "any_capture", "mean_captured", "raw_return", "environment_return",
             "demand_zero_fraction", "mean_candidate_count", "switch_rate", "inference_ms")
    return {"episodes": len(rows), "first_capture_time": float(np.mean(times)) if times else None,
            **{name: float(np.mean([row[name] for row in rows])) for name in names}}


def annealed_d_cap(config, episode):
    """Training-time d_cap for one episode: linear start→final over the window.

    Episode ``episode`` uses ``config["d_cap"] + (final - start) * e / window``,
    clamped so every episode at or beyond the window keeps the final value.
    Returns the starting d_cap when no final value is configured.
    """
    final = config["d_cap_final"]
    if final is None:
        return config["d_cap"]
    progress = min(1.0, episode / config["d_cap_anneal_episodes"])
    return config["d_cap"] + (final - config["d_cap"]) * progress


def evaluate_scales(modules, config):
    records, metrics = [], {}
    # Evaluation measures the target metric, so it always runs at the final
    # annealed d_cap (or the plain start value when annealing is disabled).
    eval_d_cap = (config["d_cap_final"] if config["d_cap_final"] is not None
                  else config["d_cap"])
    for scale, seeds in evaluation_seeds(config).items():
        scale_rows = []
        for seed in seeds:
            _, row = collect_episode(modules, config, scale, seed, d_cap=eval_d_cap)
            scale_rows.append(row)
        metrics[f"{scale[0]}v{scale[1]}"] = summarize_episodes(scale_rows)
        records.extend(scale_rows)
    metrics["joint"] = summarize_episodes(records)
    return metrics, records


def run_training(config):
    """Run mixed-scale full tasks and save lexicographic per-scale bests."""
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists() or (output / "last.pth").exists():
        raise ValueError("output already contains a training run; choose a new directory")
    modules, optimizers, metadata = build_system(config)
    replay = V3StratifiedReplay(config["replay_episodes"], seed=config["seed"])
    scale_rng = np.random.default_rng(config["seed"])
    exploration_rng = np.random.default_rng(config["seed"] + 1)
    counterfact = (config["counterfact_config"] if config["counterfact_h"] > 0
                   else None)
    counterfact_replay = (CounterfactualReplay(seed=config["seed"] + 2)
                          if counterfact is not None else None)
    best_keys, metrics, last_update, update_step = {}, {}, {}, 0
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for episode in range(config["episodes"]):
            scale = sample_scale(scale_rng, config["scale_weights"])
            rows, episode_metrics = collect_episode(
                modules, config, scale, config["seed"] + episode, exploration_rng,
                d_cap=annealed_d_cap(config, episode), counterfact=counterfact)
            if counterfact_replay is not None:
                # Rows only carry samples from the collector to the replay;
                # the transition replay itself stays free of branch payloads.
                for row in rows:
                    state = row.pop("counterfact", None)
                    if state is not None:
                        counterfact_replay.add(state)
            replay.add_episode(rows, episode_metrics, episode_metrics["geometry_progress"])
            if len(replay) >= max(1, config["warmup_transitions"]):
                for _ in range(config["updates_per_episode"]):
                    update_step += 1
                    batch = replay.sample(config["batch_size"], beta=config["importance_beta"])
                    last_update = td3_update(modules, optimizers, batch, config, update_step,
                                             counterfact_replay=counterfact_replay)
            train_record = {"kind": "train", "episode": episode, **episode_metrics,
                            "update": last_update, "replay_strata": {
                                key: len(indices) for key, indices in replay.strata.items()}}
            stream.write(json.dumps(train_record, allow_nan=False) + "\n")
            stream.flush()
            if (episode + 1) % config["eval_interval"] == 0 or episode + 1 == config["episodes"]:
                metrics, records = evaluate_scales(modules, config)
                stream.write(json.dumps({"kind": "evaluation", "episode": episode,
                                          "metrics": metrics, "records": records}, allow_nan=False) + "\n")
                stream.flush()
                payload = build_checkpoint_payload(modules, optimizers, config, episode,
                                                     update_step, metrics, metadata)
                for name, scale_metrics in metrics.items():
                    key = selection_key(scale_metrics)
                    if name not in best_keys or key > best_keys[name]:
                        best_keys[name] = key
                        atomic_save_checkpoint(payload, output / f"best_{name}.pth")
                atomic_save_checkpoint(payload, output / "last.pth")
                (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False),
                                                     encoding="utf-8")
    return {"episodes": config["episodes"], "updates": update_step, "metrics": metrics,
            "last_update": last_update, "output": str(output.resolve()),
            "replay_strata": {key: len(indices) for key, indices in replay.strata.items()}}


def main(argv=None):
    config = config_from_args(build_parser().parse_args(argv))
    print(json.dumps(run_training(config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
