"""Paired evaluation contracts for the V3 hybrid allocator against the V2 baseline."""

import json

import numpy as np
import pytest
import torch

from . import evaluate_v3_hybrid as evaluate
from . import train_v3_hybrid as training


def _record(scale="4v2", seed=10, full_capture=0.0, any_capture=0.0,
            mean_captured=0.0, first_capture_time=None, raw_return=0.0):
    return {"scale": scale, "seed": seed, "full_capture": full_capture,
            "any_capture": any_capture, "mean_captured": mean_captured,
            "first_capture_time": first_capture_time, "raw_return": raw_return}


def test_selection_key_orders_full_any_mean_first_then_raw_return():
    floor = _record(full_capture=0.0, any_capture=0.0, mean_captured=0.0,
                    first_capture_time=None, raw_return=-1e9)
    full = _record(full_capture=1.0, any_capture=1.0, mean_captured=2.0,
                   first_capture_time=500.0, raw_return=-1e12)
    assert evaluate.selection_key(full) > evaluate.selection_key(floor)

    any_capture = _record(full_capture=0.0, any_capture=1.0, mean_captured=0.1,
                          first_capture_time=900.0, raw_return=-1e12)
    assert evaluate.selection_key(any_capture) > evaluate.selection_key(floor)

    mean = _record(full_capture=0.0, any_capture=0.0, mean_captured=0.5,
                   first_capture_time=None, raw_return=-1e12)
    assert evaluate.selection_key(mean) > evaluate.selection_key(floor)

    earlier = _record(full_capture=0.0, any_capture=1.0, mean_captured=1.0,
                      first_capture_time=5.0, raw_return=-1e12)
    later = _record(full_capture=0.0, any_capture=1.0, mean_captured=1.0,
                    first_capture_time=50.0, raw_return=1e12)
    assert evaluate.selection_key(earlier) > evaluate.selection_key(later)

    richer = _record(full_capture=0.0, any_capture=0.0, mean_captured=0.0,
                     first_capture_time=None, raw_return=1e12)
    assert evaluate.selection_key(richer) > evaluate.selection_key(floor)
    assert evaluate.selection_key(_record()) == evaluate.selection_key(_record())


def test_selection_key_treats_missing_first_capture_as_worst():
    missing = _record(any_capture=1.0, mean_captured=1.0, first_capture_time=None,
                      raw_return=1e9)
    late = _record(any_capture=1.0, mean_captured=1.0, first_capture_time=1e6,
                   raw_return=-1e9)
    assert evaluate.selection_key(missing) < evaluate.selection_key(late)


def _paired_records():
    baseline = [
        _record("4v2", 10, full_capture=1.0, any_capture=1.0, mean_captured=2.0,
                first_capture_time=10.0, raw_return=1.0),
        _record("4v2", 11, any_capture=1.0, mean_captured=1.0,
                first_capture_time=20.0, raw_return=2.0),
        _record("6v2", 12, raw_return=0.0),
    ]
    v3 = [
        _record("4v2", 10, full_capture=1.0, any_capture=1.0, mean_captured=2.0,
                first_capture_time=8.0, raw_return=-5.0),
        _record("4v2", 11, any_capture=1.0, mean_captured=1.0,
                first_capture_time=20.0, raw_return=2.0),
        _record("6v2", 12, any_capture=1.0, mean_captured=1.0,
                first_capture_time=30.0, raw_return=-1.0),
    ]
    return baseline, v3


def test_paired_summary_reports_wins_ties_losses_per_scale():
    baseline, v3 = _paired_records()
    summary = evaluate.paired_summary(baseline, v3)
    assert summary["per_scale"]["4v2"] == {"episodes": 2, "wins": 1, "ties": 1, "losses": 0}
    assert summary["per_scale"]["6v2"] == {"episodes": 1, "wins": 1, "ties": 0, "losses": 0}
    assert summary["episodes"] == 3 and summary["wins"] == 2
    assert summary["ties"] == 1 and summary["losses"] == 0
    assert [pair["result"] for pair in summary["pairs"]] == ["win", "tie", "win"]


def test_paired_summary_rejects_missing_or_mismatched_seeds():
    baseline, v3 = _paired_records()
    with pytest.raises(ValueError):
        evaluate.paired_summary(baseline[:2], v3)
    with pytest.raises(ValueError):
        evaluate.paired_summary(baseline, v3[:2])
    with pytest.raises(ValueError):
        evaluate.paired_summary(baseline + [baseline[0]], v3)


def test_paired_summary_is_deterministic_under_input_permutation():
    baseline, v3 = _paired_records()
    first = evaluate.paired_summary(baseline, v3)
    flipped_baseline = [baseline[2], baseline[0], baseline[1]]
    flipped_v3 = [v3[1], v3[2], v3[0]]
    second = evaluate.paired_summary(flipped_baseline, flipped_v3)
    assert first == second
    assert list(first["per_scale"]) == ["4v2", "6v2"]


def _tiny_config(tmp_path, extra=()):
    args = training.build_parser().parse_args([
        "--output", str(tmp_path), "--episodes", "2", "--max-steps", "3",
        "--hidden-dim", "8", "--actor-hidden", "16", "--head-width", "16",
        "--future-steps", "2", "--num-obstacles", "0", "--batch-size", "2",
        "--warmup-transitions", "1", "--eval-interval", "1", "--eval-episodes", "1",
        "--assign-interval", "2", "--device", "cpu", *extra,
    ])
    return training.config_from_args(args)


V3_FINITE_FIELDS = ("steps", "num_tasks", "captured", "full_capture", "any_capture",
                    "mean_captured", "raw_return", "shaped_return", "environment_return",
                    "closure_return", "angular_closure_return", "ring_return",
                    "demand_zero_fraction", "switch_count", "switch_rate",
                    "mean_inference_ms", "max_inference_ms", "mean_candidate_count",
                    "max_candidate_count", "allocation_calls", "dead_member_violations",
                    "max_dead_capacity", "max_capacity_sum_error")


def test_run_episode_eval_records_finite_paired_diagnostics(tmp_path):
    config = _tiny_config(tmp_path)
    modules, _, _ = training.build_system(config)
    env = training.make_environment(config)
    try:
        record = evaluate.run_episode_eval(
            env, {**modules, "config": {"max_steps": 6, "assign_interval": 2,
                                        "gate_config": {"min_dwell_steps": 1,
                                                        "improvement_margin": 0.0}}},
            seed=43, scale=(4, 2))
    finally:
        env.close()
    assert record["scale"] == "4v2" and record["seed"] == 43
    assert record["num_tasks"] == 2 and record["steps"] <= 6
    assert record["updates_disabled"] is True
    assert record["controller"] == "v3_sparse_demand_hybrid"
    for name in V3_FINITE_FIELDS:
        assert name in record, name
        assert np.isfinite(record[name]), name
    assert record["max_candidate_count"] <= 32
    for name in ("initial_capacity", "final_capacity"):
        assert record[name] is not None
        assert len(record[name]) == 2
        assert np.isclose(sum(record[name]), 4)
    assert record["dead_member_violations"] == 0
    assert record["max_dead_capacity"] == 0.0
    assert record["max_capacity_sum_error"] <= 1e-6
    first = record["first_capture_time"]
    assert first is None or np.isfinite(first)
    assert isinstance(record["outcome"], str)


def test_run_episode_baseline_pairs_on_scale_and_seed(tmp_path):
    baseline = evaluate.load_baseline_checkpoint(_tiny_v2_checkpoint(tmp_path))
    config = _tiny_config(tmp_path)
    env = training.make_environment(config)
    try:
        record = evaluate.run_episode_baseline(env, baseline, seed=43, scale=(4, 2),
                                               max_steps=6)
    finally:
        env.close()
    assert record["scale"] == "4v2" and record["seed"] == 43
    assert record["controller"] == "v2_continuous_membership"
    for name in ("steps", "captured", "full_capture", "any_capture", "mean_captured",
                 "raw_return", "shaped_return", "environment_return", "switch_count"):
        assert name in record and np.isfinite(record[name]), name


class _ScriptedCaptureEnv:
    """Minimal stand-in for StageCoalitionEnv emitting scripted capture events.

    ``capture_schedule`` maps an episode step (1-based, after that step's
    ``step`` call) to the evader index captured at that step, so tests can
    construct partial- and full-capture episodes deterministically.
    """

    def __init__(self, capture_schedule=None, max_steps=8, time_step=0.5):
        self.capture_schedule = dict(capture_schedule or {})
        self.max_steps = max_steps
        self.time_step = time_step
        self.num_tasks = 2
        self.task_mask = np.zeros(3, dtype=bool)
        self.captured = np.zeros(3, dtype=bool)
        self.episode_step_count = 0
        self.outcome = "timeout"

    def reset(self, num_pursuers, num_evaders, scenario_seed):
        self.num_tasks = num_evaders
        self.task_mask[:] = False
        self.task_mask[:num_evaders] = True
        self.captured[:] = False
        self.episode_step_count = 0
        self.outcome = "timeout"

    def step(self, actions, max_steps=None):
        self.episode_step_count += 1
        captured = self.capture_schedule.get(self.episode_step_count)
        events = [] if captured is None else [captured]
        for j in events:
            self.captured[j] = True
            self.task_mask[j] = False
        done = self.episode_step_count >= min(max_steps or self.max_steps,
                                              self.max_steps)
        if self.all_captured():
            self.outcome = "all_captured"
        return 0.0, done, {"capture_events": events, "reward_components": {}}

    def all_captured(self):
        return bool(np.all(self.captured[: self.num_tasks]))

    def close(self):
        pass


class _ScriptedDecisionPolicy:
    """Stand-in for HybridPolicy returning one fixed feasible decision."""

    def __init__(self, modules, config):
        pass

    def decision(self, env):
        return {"actions": np.zeros((6, 2)),
                "assignment": np.array([0, 0, 1, 1, -1, -1]),
                "allocation": {"capacity": torch.tensor([2.0, 2.0]),
                               "candidate_count": 2,
                               "demand": torch.ones(2),
                               "switched": False},
                "allocated": True, "inference_ms": 0.0}


class _ScriptedV2Controller:
    """Stand-in for V2Controller issuing fixed assignments and zero actions."""

    def __init__(self, *args, **kwargs):
        self.max_pursuers = 6

    def reset(self):
        pass

    def controller_step(self, env):
        return np.zeros((4, 2)), {"assignment": [0, 0, 1, 1, -1, -1]}


class _ModuleStub:
    def eval(self):
        return self


def _scripted_allocator_modules():
    return {name: _ModuleStub()
            for name in ("encoder", "demand_head", "value_head", "actor")}


def _scripted_baseline_kwargs():
    return {name: None for name in (
        "encoder", "demand_head", "actor", "device", "epsilon_ot",
        "sinkhorn_iters", "count_penalty", "switch_penalty", "hysteresis",
        "min_dwell_steps", "assign_interval", "demand_source",
        "fixed_demand", "target_context")}


def test_baseline_first_capture_time_uses_first_capture_events(monkeypatch):
    # Partial capture: one evader captured at step 3, the other stays free,
    # so the episode ends without all_captured() ever becoming True.
    monkeypatch.setattr(evaluate, "V2Controller", _ScriptedV2Controller)
    env = _ScriptedCaptureEnv(capture_schedule={3: 0}, max_steps=8)
    record = evaluate.run_episode_baseline(
        env, _scripted_baseline_kwargs(), seed=7, scale=(4, 2), max_steps=8)
    assert record["captured"] == 1
    assert record["full_capture"] == 0.0 and record["any_capture"] == 1.0
    assert record["first_capture_time"] == 3 * env.time_step


def test_baseline_first_capture_time_precedes_full_capture(monkeypatch):
    # Full capture at step 5 must not overwrite the first capture at step 3.
    monkeypatch.setattr(evaluate, "V2Controller", _ScriptedV2Controller)
    env = _ScriptedCaptureEnv(capture_schedule={3: 0, 5: 1}, max_steps=8)
    record = evaluate.run_episode_baseline(
        env, _scripted_baseline_kwargs(), seed=7, scale=(4, 2), max_steps=8)
    assert record["captured"] == 2 and record["full_capture"] == 1.0
    assert record["first_capture_time"] == 3 * env.time_step


def test_first_capture_time_semantics_match_between_sides(monkeypatch):
    # Same seed and identical capture scripts: both recorders must derive
    # first_capture_time from the first non-empty capture_events step.
    monkeypatch.setattr(evaluate, "V2Controller", _ScriptedV2Controller)
    monkeypatch.setattr(evaluate, "HybridPolicy", _ScriptedDecisionPolicy)
    seed = 11
    baseline_env = _ScriptedCaptureEnv(capture_schedule={3: 0}, max_steps=8)
    baseline_record = evaluate.run_episode_baseline(
        baseline_env, _scripted_baseline_kwargs(), seed=seed, scale=(4, 2),
        max_steps=8)
    v3_env = _ScriptedCaptureEnv(capture_schedule={3: 0}, max_steps=8)
    v3_record = evaluate.run_episode_eval(
        v3_env, {**_scripted_allocator_modules(), "config": {"max_steps": 8}},
        seed=seed, scale=(4, 2))
    assert baseline_record["first_capture_time"] is not None
    assert v3_record["first_capture_time"] is not None
    assert (baseline_record["first_capture_time"]
            == v3_record["first_capture_time"]
            == 3 * baseline_env.time_step)


def _tiny_v3_checkpoint(tmp_path, config, modules, optimizers):
    payload = training.build_checkpoint_payload(
        modules, optimizers, config, episode=1, update_step=1,
        metrics={"4v2": {"full_capture": 0.0}}, metadata={"source": "test"})
    path = tmp_path / "v3_tiny.pth"
    training.atomic_save_checkpoint(payload, path)
    return path


def test_load_v3_system_ignores_training_warm_start_path(tmp_path):
    """A deployed checkpoint must load without its historical source model."""
    config = _tiny_config(tmp_path)
    modules, optimizers, _ = training.build_system(config)
    path = _tiny_v3_checkpoint(tmp_path, config, modules, optimizers)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["init_v3_from"] = str(
        tmp_path / "source-model-that-no-longer-exists.pth"
    )
    torch.save(payload, path)

    loaded, loaded_config = evaluate.load_v3_system(path, device="cpu")

    assert set(("encoder", "demand_head", "value_head", "actor")) <= set(loaded)
    assert loaded_config["init_v3_from"] is None


def _tiny_v2_checkpoint(tmp_path):
    from .continuous_membership import ContinuousDemandHead
    from .stage_networks import SetInvariantCritic, SharedActor, StageFeatureEncoder
    from .v2_controller import v2_obs_dim
    device = torch.device("cpu")
    encoder = StageFeatureEncoder(
        6, 3, 0, 8.0, 8, 2, 10, 9, device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True,
    ).to(device)
    encoder._skip_legacy_plan = True
    demand_head = ContinuousDemandHead(8, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, 8, 2), hidden_dim=16).to(device)
    critic1 = SetInvariantCritic(v2_obs_dim(6, 8, 2), 16).to(device)
    critic2 = SetInvariantCritic(v2_obs_dim(6, 8, 2), 16).to(device)
    payload = {"architecture_version": "continuous_membership_v2", "hidden_dim": 8,
               "future_steps": 2, "seq_len": 10, "map_size": 8.0, "num_obstacles": 0,
               "d_cap": 1.5, "hold_steps": 3, "target_context": "hard",
               "st_hgat": encoder.st_hgat.state_dict(),
               "relation_head": encoder.relation_head.state_dict(),
               "demand_head": demand_head.state_dict(), "actor": actor.state_dict(),
               "critic1": critic1.state_dict(), "critic2": critic2.state_dict(),
               "args": {"demand_source": "fixed", "fixed_demand": 3.0,
                        "evader_speed": 0.2, "actor_hidden": 16}}
    path = tmp_path / "v2_tiny.pth"
    torch.save(payload, path)
    return path


def test_run_dry_run_checks_pipeline_invariants(tmp_path):
    modules, config = evaluate.build_v3_system_from_warm_start(
        _tiny_v2_checkpoint(tmp_path), device="cpu")
    config["max_steps"] = 5
    plan = {(4, 2): [901], (6, 2): [902], (6, 3): [903]}
    result = evaluate.run_dry_run({**modules, "config": config}, plan)
    checks = result["checks"]
    assert checks["episodes"] == 3
    assert checks["finite_actions_rewards"] is True
    assert checks["max_candidate_count"] <= checks["candidate_limit"] == 32
    assert checks["candidates_within_limit"] is True
    assert checks["capacity_sum_ok"] is True
    assert checks["dead_tasks_zero_members"] is True
    assert [row["scale"] for row in result["episodes"]] == ["4v2", "6v2", "6v3"]
    assert all(row["updates_disabled"] for row in result["episodes"])


def test_cli_paired_eval_writes_matched_records_and_summary(tmp_path):
    config = _tiny_config(tmp_path / "unused")
    modules, optimizers, _ = training.build_system(config)
    v3_path = _tiny_v3_checkpoint(tmp_path, config, modules, optimizers)
    v2_path = _tiny_v2_checkpoint(tmp_path)
    out = tmp_path / "paired.json"
    result = evaluate.main([
        "--checkpoint", str(v3_path), "--baseline-checkpoint", str(v2_path),
        "--scales", "4v2,6v2", "--seeds", "101,102", "--max-steps", "4",
        "--out", str(out), "--device", "cpu",
    ])
    assert out.exists()
    saved = json.loads(out.read_text())
    assert saved["paired_summary"] == result["paired_summary"]
    assert set(saved["scales"]) == {"4v2", "6v2"}
    base_keys = {(row["scale"], row["seed"]) for row in saved["baseline_records"]}
    v3_keys = {(row["scale"], row["seed"]) for row in saved["v3_records"]}
    assert base_keys == v3_keys == {("4v2", 101), ("4v2", 102),
                                    ("6v2", 101), ("6v2", 102)}
    scale_summary = saved["paired_summary"]["per_scale"]["4v2"]
    assert scale_summary["episodes"] == 2
    assert scale_summary["wins"] + scale_summary["ties"] + scale_summary["losses"] == 2


def test_cli_warm_start_policy_builds_v3_without_v3_checkpoint(tmp_path):
    v2_path = _tiny_v2_checkpoint(tmp_path)
    out = tmp_path / "warm.json"
    evaluate.main([
        "--warm-start-policy", str(v2_path), "--baseline-checkpoint", str(v2_path),
        "--scales", "4v2", "--seeds", "1", "--max-steps", "3",
        "--out", str(out), "--device", "cpu",
    ])
    saved = json.loads(out.read_text())
    assert saved["source"]["kind"] == "v2_warm_start"
    assert saved["source"]["path"] == str(v2_path)
    assert len(saved["v3_records"]) == 1
    assert saved["v3_records"][0]["max_candidate_count"] <= 32
    assert saved["paired_summary"]["episodes"] == 1
