"""Ring expert demonstrations from assigned 4v2 training scenes only."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .stage_env import StageCoalitionEnv
from .traditional_ring_baseline import RingGuidanceController
from .v2_env_adapter import apply_membership
from .v2_training import build_v2_obs_torch, hard_K_from_assignment


SCHEMA = "v3_ring_expert_4v2_v1"


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("scenario seed must be an integer")
    return int(value)


def validate_training_seeds(seeds: Iterable, assigned_seeds: Iterable) -> list[int]:
    """Reject unassigned, held-out, sealed and derived scenes."""
    assigned = {_seed(x) for x in assigned_seeds}
    if not assigned or any(not 72000 <= x <= 72999 for x in assigned):
        raise ValueError("assigned seeds must be in the training pool")
    accepted = []
    for item in seeds:
        if isinstance(item, Mapping):
            if set(item) != {"seed"}:
                raise ValueError("derived or unrecognized scene provenance")
            value = _seed(item["seed"])
        else:
            value = _seed(item)
        if not 72000 <= value <= 72999 or value not in assigned or value == 70086:
            raise ValueError(f"seed {value} is not assigned training data")
        if value in accepted:
            raise ValueError(f"duplicate seed {value}")
        accepted.append(value)
    return accepted


def initial_assignments() -> list[np.ndarray]:
    return [np.asarray(bits, dtype=np.int64) for bits in itertools.product((0, 1), repeat=4)]


def repair_dead_tasks(assignment: Sequence[int], task_mask: Sequence[bool]) -> np.ndarray:
    repaired = np.asarray(assignment, dtype=np.int64).copy()
    active = np.flatnonzero(np.asarray(task_mask, dtype=bool))
    if len(active) == 0:
        repaired[:] = -1
    else:
        repaired[~np.isin(repaired, active)] = int(active[0])
    return repaired


def assigned_member_distances(pursuers, evaders, assignment, task_mask):
    """Return per-member assigned distance and farthest assigned member."""
    pursuers = np.asarray(pursuers, dtype=np.float64)
    evaders = np.asarray(evaders, dtype=np.float64)
    assignment = np.asarray(assignment, dtype=np.int64)
    active = np.asarray(task_mask, dtype=bool)
    if len(pursuers) != len(assignment):
        raise ValueError("assignment size differs from pursuer count")
    distances = np.full(len(assignment), np.nan)
    for i, target in enumerate(assignment):
        if 0 <= target < len(evaders) and target < len(active) and active[target]:
            distances[i] = np.linalg.norm(pursuers[i] - evaders[target])
    valid = np.flatnonzero(np.isfinite(distances))
    farthest = int(valid[np.argmax(distances[valid])]) if len(valid) else -1
    return distances.tolist(), farthest


def _bounded_action(actions, count):
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] not in (count, 6):
        raise ValueError("action shape must be [N,2] or [6,2]")
    array = array[:count]
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > 1.0 + 1e-6):
        raise ValueError("actions must be finite and in [-1,1]")
    return np.clip(array, -1.0, 1.0)


@torch.no_grad()
def build_expert_record(env, assignment, encoded, actor, expert_action) -> dict:
    """Use exactly the expert's hard alliance for the original Actor input."""
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.shape != (env.num_pursuers,):
        raise ValueError("assignment must cover active pursuers")
    apply_membership(env, assignment)
    snap = env.snapshot()
    if not np.array_equal(snap["assignment"][:env.num_pursuers], assignment):
        raise ValueError("hard membership differs from expert assignment")
    device = encoded["pursuer_embeddings"].device
    hard_k = hard_K_from_assignment(
        assignment, snap["task_mask"], env.max_pursuers, env.max_evaders, device,
    )
    obs = build_v2_obs_torch(encoded, hard_k, snap, env.max_pursuers, env.max_evaders)
    original = actor(obs)
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    original = _bounded_action(original, env.num_pursuers)
    expert = _bounded_action(expert_action, env.num_pursuers)
    obs_array = obs.detach().cpu().numpy()[:env.num_pursuers]
    if not np.all(np.isfinite(obs_array)):
        raise ValueError("nonfinite observation")
    distances, farthest = assigned_member_distances(
        env.pursuer_positions(), env.evader_positions(), assignment, snap["task_mask"],
    )
    return {
        "seed": int(env.scenario_seed), "assignment": assignment.tolist(),
        "obs": obs_array.tolist(), "expert_action": expert.tolist(),
        "original_action": original.tolist(), "farthest_member": farthest,
        "assigned_distance": distances,
        "captured": int(np.count_nonzero(env.captured[:env.num_evaders])),
        "full_capture": bool(env.all_captured()),
    }


def _environment(seed: int) -> StageCoalitionEnv:
    env = StageCoalitionEnv(
        length=8.0, num_obstacle=10, max_pursuers=6, max_evaders=3,
        d_cap=0.8, hold_steps=5, v_max=0.35, v_max_e=0.35, seed=seed,
    )
    env.scenario_sampler = Deploy5x5Sampler()
    return env


def rollout_assignment(seed: int, initial_assignment, encoder, actor,
                       assigned_seeds, max_steps: int = 700) -> dict:
    """Roll one alliance with dead-task repair and the ring baseline."""
    validate_training_seeds([seed], assigned_seeds)
    initial = np.asarray(initial_assignment, dtype=np.int64)
    if initial.shape != (4,) or np.any((initial < 0) | (initial > 1)):
        raise ValueError("initial assignment must be one 4v2 candidate")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    env = _environment(seed)
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=seed)
    encoder.reset()
    controller = RingGuidanceController(d_cap=0.8, v_max=0.35, n_p=4)
    controller.reset(env)
    records, reward_total = [], 0.0
    try:
        for step in range(max_steps):
            if env.done or env.all_captured():
                break
            current = repair_dead_tasks(initial, env.task_mask[:2])
            controller.assign = current.copy()
            apply_membership(env, current)
            encoded = encoder.encode(env.snapshot(), differentiable=False)
            expert = _bounded_action(controller.act(env), 4)
            record = build_expert_record(env, current, encoded, actor, expert)
            padded = np.zeros((env.max_pursuers, 2))
            padded[:4] = expert
            reward, done, _ = env.step(padded, max_steps=max_steps)
            records.append(record)
            reward_total += float(reward)
            if done:
                break
        captured = int(np.count_nonzero(env.captured[:2]))
        full_capture = bool(env.all_captured())
        for record in records:
            record["captured"] = captured
            record["full_capture"] = full_capture
        return {
            "seed": seed, "initial_assignment": initial.tolist(),
            "captured": captured, "full_capture": full_capture,
            "outcome": str(env.outcome), "steps": len(records),
            "reward": reward_total, "records": records,
        }
    except (ValueError, FloatingPointError) as exc:
        return {
            "seed": seed, "initial_assignment": initial.tolist(),
            "captured": int(np.count_nonzero(env.captured[:2])),
            "full_capture": bool(env.all_captured()), "outcome": "invalid_sample",
            "steps": len(records), "reward": reward_total,
            "error": str(exc), "records": [],
        }


def generate_seed(seed: int, assigned_seeds, encoder, actor, max_steps: int = 700) -> list[dict]:
    validate_training_seeds([seed], assigned_seeds)
    return [rollout_assignment(seed, a, encoder, actor, assigned_seeds, max_steps)
            for a in initial_assignments()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as fh:
        fh.write(contents)
        fh.flush()
        os.fsync(fh.fileno())
    temporary.replace(path)


def write_shard(output_dir, seeds, assigned_seeds, trials, source_sha256: str) -> dict:
    """Validate first, then write an immutable trajectories/failures manifest."""
    accepted = validate_training_seeds(seeds, assigned_seeds)
    if not isinstance(source_sha256, str) or not source_sha256:
        raise ValueError("source_sha256 is required")
    by_seed = {seed: [] for seed in accepted}
    trials = list(trials)
    for trial in trials:
        seed = _seed(trial["seed"])
        if seed not in by_seed:
            raise ValueError("trial seed outside assigned shard")
        if len(trial["initial_assignment"]) != 4:
            raise ValueError("trial assignment must have four members")
        for record in trial.get("records", []):
            if _seed(record["seed"]) != seed:
                raise ValueError("record seed differs from trial seed")
            if len(record["assignment"]) != 4:
                raise ValueError("record assignment must have four members")
            _bounded_action(record["expert_action"], 4)
            _bounded_action(record["original_action"], 4)
            if not np.all(np.isfinite(np.asarray(record["obs"], dtype=np.float64))):
                raise ValueError("nonfinite observation")
            if not np.all(np.isfinite(np.asarray(record["assigned_distance"], dtype=np.float64))):
                raise ValueError("nonfinite assigned distance")
        by_seed[seed].append(trial)
    successes, failures = [], []
    for seed, attempts in by_seed.items():
        wins = [t for t in attempts if t["captured"] >= 1 and t.get("records")]
        if wins:
            best = max(wins, key=lambda t: (
                int(t["full_capture"]), int(t["captured"]), -int(t["steps"]),
                float(t.get("reward", 0.0)),
            ))
            successes.append({
                "seed": seed, "initial_assignment": best["initial_assignment"],
                "captured": best["captured"], "full_capture": bool(best["full_capture"]),
                "outcome": best["outcome"], "steps": best["steps"], "records": best["records"],
            })
        failures.extend({k: v for k, v in t.items() if k != "records"}
                        for t in attempts if t["captured"] < 1 or not t.get("records"))
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"shard already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        paths = {"trajectories.jsonl": successes, "failures.jsonl": failures}
        for name, rows in paths.items():
            _write_atomic(output / name, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        manifest = {
            "schema": SCHEMA, "protocol": "deploy5x5", "d_cap": 0.8,
            "seeds": accepted, "successful_seeds": [row["seed"] for row in successes],
            "attempts": len(trials), "failed_attempts": len(failures),
            "source_sha256": source_sha256,
            "files": {name: _sha256(output / name) for name in paths},
        }
        _write_atomic(output / "artifact_manifest.json", json.dumps(manifest, indent=2) + "\n")
        return manifest
    except Exception:
        for child in output.iterdir():
            child.unlink()
        output.rmdir()
        raise


def generate_shard(output_dir, seeds, assigned_seeds, encoder, actor,
                   source_sha256: str, max_steps: int = 700) -> dict:
    """Generate all 16 candidates per assigned training seed."""
    accepted = validate_training_seeds(seeds, assigned_seeds)
    trials = [trial for seed in accepted for trial in
              generate_seed(seed, assigned_seeds, encoder, actor, max_steps)]
    return write_shard(output_dir, accepted, assigned_seeds, trials, source_sha256)


def validate_shard_manifests(manifest_paths, assigned_seeds,
                             expected_source_sha256: str) -> dict:
    """Read-only central validation before shards can be frozen for training."""
    assigned = validate_training_seeds(assigned_seeds, assigned_seeds)
    seen, shards, successful = set(), [], set()
    for path in manifest_paths:
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != SCHEMA or manifest.get("protocol") != "deploy5x5"
                or manifest.get("d_cap") != 0.8):
            raise ValueError("shard schema or protocol mismatch")
        if manifest.get("source_sha256") != expected_source_sha256:
            raise ValueError("shard source hash mismatch")
        seeds = validate_training_seeds(manifest["seeds"], assigned)
        if seen.intersection(seeds):
            raise ValueError("duplicate seed across shards")
        seen.update(seeds)
        for name in ("trajectories.jsonl", "failures.jsonl"):
            file_path = path.parent / name
            if _sha256(file_path) != manifest.get("files", {}).get(name):
                raise ValueError("shard file hash mismatch")
        listed_success = set()
        with (path.parent / "trajectories.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                seed = _seed(row["seed"])
                if seed not in seeds or seed in listed_success or int(row["captured"]) < 1:
                    raise ValueError("invalid successful trajectory")
                listed_success.add(seed)
                for record in row["records"]:
                    if _seed(record["seed"]) != seed or len(record["assignment"]) != 4:
                        raise ValueError("trajectory record seed or assignment mismatch")
                    _bounded_action(record["expert_action"], 4)
                    _bounded_action(record["original_action"], 4)
                    if not np.all(np.isfinite(np.asarray(record["obs"], dtype=np.float64))):
                        raise ValueError("nonfinite observation")
                    if not np.all(np.isfinite(np.asarray(record["assigned_distance"], dtype=np.float64))):
                        raise ValueError("nonfinite assigned distance")
        if listed_success != set(manifest.get("successful_seeds", [])):
            raise ValueError("successful seed index mismatch")
        with (path.parent / "failures.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if (_seed(row["seed"]) not in seeds or
                        (int(row["captured"]) >= 1 and row.get("outcome") != "invalid_sample")):
                    raise ValueError("invalid failure metadata")
        successful.update(listed_success)
        shards.append({"path": str(path), "sha256": _sha256(path), "seeds": seeds})
    return {
        "schema": SCHEMA, "source_sha256": expected_source_sha256,
        "seeds": sorted(seen), "successful_seeds": sorted(successful),
        "shards": shards,
    }
