"""Training contracts for the isolated sparse-demand hybrid trainer."""

import json

import numpy as np
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from .train_v3_hybrid import V3StratifiedReplay, sample_scale
from . import train_v3_hybrid as training


def test_scale_sampler_is_reproducible_and_only_samples_complete_tasks():
    first = np.random.default_rng(17)
    second = np.random.default_rng(17)
    samples = [sample_scale(first) for _ in range(200)]
    assert samples == [sample_scale(second) for _ in range(200)]
    assert set(samples) == {(4, 2), (6, 2), (6, 3)}
    assert sample_scale(first, ((6, 3, 1.0),)) == (6, 3)
    with pytest.raises(ValueError):
        sample_scale(first, ((3, 1, 1.0),))


def _replay():
    replay = V3StratifiedReplay(max_episodes=8, seed=3)
    for count, captured, progress in [(1, 2, 0), (2, 1, 0), (3, 0, 0.8), (4, 0, 0)]:
        replay.add_episode(
            [{"id": (count, i)} for i in range(count)],
            outcome={"captured": captured, "num_tasks": 2},
            geometry_progress=progress,
        )
    return replay


def test_replay_retains_all_outcomes_and_at_least_half_uniform_draws():
    replay = _replay()
    batch = replay.sample(20, {"uniform": 0.1, "full": 0.9}, beta=0.6)
    assert len(replay) == 10
    assert {name: len(indices) for name, indices in replay.strata.items()} == {
        "full": 1, "partial": 2, "progress": 3, "ordinary": 4,
    }
    assert len(batch["transitions"]) == 20
    assert np.count_nonzero(batch["draw_sources"] == "uniform") >= 10
    weights = batch["importance_weights"]
    assert np.all(np.isfinite(weights)) and np.all(weights > 0)
    assert np.max(weights) <= 1.0
    # Half uniform over ten transitions, half over the one full-capture row.
    expected_probability = np.array([
        0.55 if item["id"][0] == 1 else 0.05 for item in batch["transitions"]
    ])
    np.testing.assert_allclose(batch["probabilities"], expected_probability)
    unnormalized = (1 / (10 * expected_probability)) ** 0.6
    np.testing.assert_allclose(weights, unnormalized / unnormalized.max())


def test_empty_strata_fall_back_to_uniform_and_eviction_removes_stale_indices():
    replay = V3StratifiedReplay(max_episodes=1, seed=1)
    with pytest.raises(ValueError):
        replay.sample(2, {}, beta=0.4)
    replay.add_episode([{"id": "old"}], "full", 0.0)
    replay.add_episode([{"id": "new"}], "ordinary", 0.0)
    batch = replay.sample(3, {"full": 1.0}, beta=1.0)
    assert len(replay) == 1 and not replay.strata["full"]
    assert [item["id"] for item in batch["transitions"]] == ["new"] * 3
    np.testing.assert_array_equal(batch["importance_weights"], np.ones(3))
    assert np.all(batch["draw_sources"] == "uniform")


def _checkpoint_objects():
    encoder = nn.Module()
    encoder.add_module("st_hgat", nn.Linear(2, 2))
    encoder.add_module("relation_head", nn.Linear(2, 1))
    modules = {name: nn.Linear(2, 1) for name in (
        "demand_head", "value_head", "actor", "critic1", "critic2",
        "actor_target", "critic1_target", "critic2_target", "value_head_target",
    )}
    modules["encoder"] = encoder
    optimizers = {"actor": torch.optim.Adam(modules["actor"].parameters())}
    config = {
        "max_capacity_modes": 6, "max_alliance_candidates": 32,
        "scale_weights": ((4, 2, 0.4), (6, 2, 0.3), (6, 3, 0.3)),
        "reward_config": {"angular_closure_weight": 0.1},
        "gate_config": {"min_dwell_steps": 10, "improvement_margin": 0.1},
    }
    return modules, optimizers, config


def test_checkpoint_roundtrip_keeps_networks_targets_optimizer_and_allocation_config(tmp_path):
    modules, optimizers, config = _checkpoint_objects()
    loss = modules["actor"](torch.ones(1, 2)).sum()
    loss.backward()
    optimizers["actor"].step()
    payload = training.build_checkpoint_payload(
        modules, optimizers, config, episode=7, update_step=11,
        metrics={"4v2": {"full_capture": 0.2}}, metadata={"warm_start": "test"},
    )
    assert payload["architecture_version"] == "v3_sparse_demand_value_hybrid_v1"
    assert payload["demand_normalizer"] == "masked_sparsemax"
    assert {"st_hgat", "relation_head", "optimizers", "reward_config", "gate_config",
            "scale_weights", "max_capacity_modes", "max_alliance_candidates", "metrics",
            "config", "metadata", *modules} <= payload.keys()
    path = tmp_path / "model.pth"
    training.atomic_save_checkpoint(payload, path)
    restored, restored_optimizers, _ = _checkpoint_objects()
    loaded = training.load_v3_hybrid_checkpoint(path, restored, restored_optimizers)
    assert loaded["episode"] == 7 and loaded["update_step"] == 11
    assert loaded["max_alliance_candidates"] == 32
    for name in modules:
        for key, value in modules[name].state_dict().items():
            torch.testing.assert_close(restored[name].state_dict()[key], value)
    assert restored_optimizers["actor"].state_dict()["state"]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("schema", ["continuous_membership_v2", "normalized_demand_student_v3"])
def test_loader_rejects_old_schema_before_loading_any_weight(tmp_path, schema):
    modules, optimizers, config = _checkpoint_objects()
    payload = training.build_checkpoint_payload(modules, optimizers, config)
    payload["architecture_version"] = schema
    payload["actor"] = {key: torch.full_like(value, 99) for key, value in modules["actor"].state_dict().items()}
    original = {key: value.clone() for key, value in modules["actor"].state_dict().items()}
    path = tmp_path / "incompatible.pth"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        training.load_v3_hybrid_checkpoint(path, modules)
    for key, value in original.items():
        torch.testing.assert_close(modules["actor"].state_dict()[key], value)


def test_loader_rejects_wrong_normalizer_and_missing_targets_before_mutating_weights(tmp_path):
    modules, optimizers, config = _checkpoint_objects()
    payload = training.build_checkpoint_payload(modules, optimizers, config)
    path = tmp_path / "incomplete.pth"
    for invalid in (dict(payload, demand_normalizer="softmax"),
                    {key: value for key, value in payload.items() if key != "critic2_target"}):
        torch.save(invalid, path)
        with pytest.raises(ValueError):
            training.load_v3_hybrid_checkpoint(path, modules)


def _tiny_config(tmp_path, extra=()):
    args = training.build_parser().parse_args([
        "--output", str(tmp_path), "--episodes", "2", "--max-steps", "3",
        "--hidden-dim", "8", "--actor-hidden", "16", "--head-width", "16",
        "--future-steps", "2", "--num-obstacles", "0", "--batch-size", "2",
        "--warmup-transitions", "1", "--eval-interval", "1", "--eval-episodes", "1",
        "--assign-interval", "2", "--device", "cpu", *extra,
    ])
    return training.config_from_args(args)


def test_parser_builds_full_scale_protocol_and_disjoint_fixed_eval_seeds(tmp_path):
    config = _tiny_config(tmp_path)
    assert config["actor_los_weight"] == config["actor_ring_weight"] == 0.0
    assert config["scale_weights"] == ((4, 2, 0.4), (6, 2, 0.3), (6, 3, 0.3))
    first = training.evaluation_seeds(config)
    assert first == training.evaluation_seeds(config)
    all_seeds = [seed for seeds in first.values() for seed in seeds]
    assert len(all_seeds) == len(set(all_seeds)) == 3
    assert not set(all_seeds) & set(range(config["seed"], config["seed"] + config["episodes"]))
    with pytest.raises(ValueError):
        _tiny_config(tmp_path, ("--eval-seed", "37"))


def test_config_from_args_rejects_policy_warm_start_missing_actor_network(tmp_path):
    path = tmp_path / "v2_missing_actor.pth"
    torch.save({"architecture_version": "continuous_membership_v2", "hidden_dim": 8}, path)
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path / "out"), "--init-policy-from", str(path)])
    with pytest.raises(ValueError, match="actor"):
        training.config_from_args(args)


def test_full_episode_stores_accepted_assignments_and_aligned_next_policy_state(tmp_path):
    config = _tiny_config(tmp_path)
    modules, _, metadata = training.build_system(config)
    assert metadata["encoder_frozen"]
    assert not any(p.requires_grad for p in modules["encoder"].parameters())
    rows, metrics = training.collect_episode(modules, config, (4, 2), 43)
    assert len(rows) == 3 and rows[-1]["done"]
    assert metrics["scale"] == "4v2" and metrics["steps"] == 3
    assert metrics["allocation_calls"] == 2
    assert metrics["max_candidate_count"] <= 32
    assert metrics["first_capture_time"] is None
    for i, row in enumerate(rows):
        assert row["assignment"].shape == (4,)
        np.testing.assert_array_equal(row["snapshot"]["assignment"][:4], row["assignment"])
        assert set(row["assignment"].tolist()) <= {0, 1}
        assert torch.isfinite(row["obs"]).all() and np.isfinite(row["actions"]).all()
        assert np.isclose(row["reward"], row["raw_reward"] + sum(row["shaping_components"].values()))
        assert np.isclose(row["capacity"].sum(), 4)
        if i + 1 < len(rows):
            torch.testing.assert_close(row["next_obs"], rows[i + 1]["obs"])
    assert torch.count_nonzero(rows[-1]["next_mask"]) == 0


def test_td3_applies_importance_weights_and_delays_actor_while_training_both_heads(tmp_path):
    config = _tiny_config(tmp_path, ("--target-policy-noise", "0",))
    modules, optimizers, _ = training.build_system(config)
    rows, _ = training.collect_episode(modules, config, (4, 2), 44)
    batch = {"transitions": rows[:2], "importance_weights": np.array([0.2, 1.0])}
    obs = torch.stack([row["obs"] for row in rows[:2]])
    acts = torch.as_tensor(np.stack([row["actions"] for row in rows[:2]]))
    masks = torch.stack([row["mask"] for row in rows[:2]])
    with torch.no_grad():
        target_obs = torch.stack([row["next_obs"] for row in rows[:2]])
        target_masks = torch.stack([row["next_mask"] for row in rows[:2]])
        target_actions = modules["actor_target"](target_obs)
        target_q = torch.minimum(*[
            modules[name](target_obs, target_actions, target_masks)
            for name in ("critic1_target", "critic2_target")
        ])
        rewards = torch.tensor([row["reward"] for row in rows[:2]]) * config["reward_scale"]
        target = rewards + config["gamma"] * target_q
        expected = sum(((modules[name](obs, acts, masks) - target).square()
                        * torch.tensor([0.2, 1.0])).mean()
                       for name in ("critic1", "critic2"))
    before = {name: {k: v.clone() for k, v in modules[name].state_dict().items()}
              for name in ("actor", "demand_head", "value_head", "encoder")}
    first = training.td3_update(modules, optimizers, batch, config, update_step=1)
    assert first["critic_loss"] == pytest.approx(float(expected), rel=1e-5)
    assert not first["actor_updated"]
    for key, value in before["actor"].items():
        torch.testing.assert_close(modules["actor"].state_dict()[key], value)
    second = training.td3_update(modules, optimizers, batch, config, update_step=2)
    assert second["actor_updated"]
    assert all(np.isfinite(value) for value in second.values())
    for name in ("actor", "demand_head", "value_head"):
        assert any(not torch.equal(value, modules[name].state_dict()[key])
                   for key, value in before[name].items())
    for key, value in before["encoder"].items():
        torch.testing.assert_close(modules["encoder"].state_dict()[key], value)


def test_head_distillation_stays_finite_with_extreme_mode_teacher_values(tmp_path, monkeypatch):
    class _SpreadTeacher(nn.Module):
        def forward(self, features, alive):
            values = torch.full((features.shape[0],), -1e4, dtype=features.dtype)
            values[0] = 1e4
            return values

    def _fixed_candidates(relation_costs, modes, alive, current_assignment=None,
                          max_candidates=None):
        return np.array([[0, 0, 0, 0], [0, 1, 1, 1]])

    captured = {}
    original_target = training.best_mode_demand_target

    def _spy_target(values, candidates, num_agents, num_tasks):
        target, mode = original_target(values, candidates, num_agents, num_tasks)
        captured["target"] = target.detach().clone()
        captured["mode"] = mode
        return target, mode

    monkeypatch.setattr(training, "generate_bounded_candidates", _fixed_candidates)
    monkeypatch.setattr(training, "best_mode_demand_target", _spy_target)
    config = _tiny_config(tmp_path)
    modules, _, _ = training.build_system(config)
    rows, _ = training.collect_episode(modules, config, (4, 2), 45)
    modules["value_head_target"] = _SpreadTeacher()
    (demand_loss, value_loss, rank_loss, cf_rank_loss, cf_reg_loss,
     pair_accuracy, distinct_modes) = training._head_losses(
        modules, rows, config, torch.device("cpu"))
    assert torch.isfinite(demand_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(rank_loss)
    assert float(cf_rank_loss) == pytest.approx(0.0)
    assert float(cf_reg_loss) == pytest.approx(0.0)
    assert pair_accuracy == 0.0
    assert distinct_modes == 0.0
    teacher = captured["target"]
    assert captured["mode"] == (4, 0)
    assert torch.equal(teacher, torch.tensor([1.0, 0.0]))


def test_explicit_v2_and_value_warm_starts_preserve_compatible_weights(tmp_path):
    from .coalition_value import CoalitionValueHead
    config = _tiny_config(tmp_path)
    original, _, _ = training.build_system(config)
    policy = {"architecture_version": "continuous_membership_v2", "hidden_dim": 8,
              "st_hgat": original["encoder"].st_hgat.state_dict(),
              "relation_head": original["encoder"].relation_head.state_dict(),
              **{name: original[name].state_dict() for name in ("actor", "critic1", "critic2")}}
    policy_path = tmp_path / "v2.pth"
    torch.save(policy, policy_path)
    proxy = CoalitionValueHead(8, width=16)
    value_path = tmp_path / "proxy.pth"
    torch.save({"schema": "coalition_value_geometric_proxy_v1",
                "supervision": "geometric_proxy_not_rollout_return",
                "hidden_dim": 8, "value_head": proxy.state_dict()}, value_path)
    config.update(init_policy_from=str(policy_path), init_value_from=str(value_path))
    restored, _, metadata = training.build_system(config)
    for key, value in original["actor"].state_dict().items():
        torch.testing.assert_close(restored["actor"].state_dict()[key], value)
    features = torch.randn(2, 2, 23)
    torch.testing.assert_close(proxy(features), restored["value_head"](
        torch.cat([features, torch.randn(2, 2, 6)], dim=-1)))
    assert metadata["value_warm_start"]["warm_start_only"] is True
    assert metadata["policy_warm_start"]["schema"] == "continuous_membership_v2"
    torch.save({"value_head": proxy.state_dict()}, value_path)
    with pytest.raises(ValueError, match="schema|marked"):
        training.build_system(config)


def test_lexicographic_selection_prioritizes_captures_over_raw_reward():
    base = {"full_capture": 0.0, "any_capture": 0.5, "mean_captured": 0.5,
            "first_capture_time": 10.0, "raw_return": 5.0}
    assert training.selection_key(dict(base, full_capture=0.1, raw_return=-1000)) > training.selection_key(base)
    assert training.selection_key(dict(base, first_capture_time=9.0)) > training.selection_key(base)
    assert training.selection_key(dict(base, first_capture_time=None)) < training.selection_key(base)


def test_small_training_run_saves_all_scale_bests_and_finite_metrics(tmp_path):
    import json
    config = _tiny_config(tmp_path, ("--actor-los-weight", "0.01", "--actor-ring-weight", "0.01"))
    result = training.run_training(config)
    assert result["updates"] >= 2
    assert set(result["metrics"]) == {"4v2", "6v2", "6v3", "joint"}
    assert result["last_update"]["actor_updated"]
    assert np.isfinite(result["last_update"]["actor_loss"])
    for scale in ("4v2", "6v2", "6v3", "joint"):
        payload = training.load_v3_hybrid_checkpoint(tmp_path / f"best_{scale}.pth")
        assert payload["metrics"][scale]["episodes"] >= 1
    logs = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in logs].count("train") == 2
    assert [row["kind"] for row in logs].count("evaluation") == 2


_SMOKE_V3_CKPT = Path(__file__).resolve().parents[5] / "checkpoints" / "v3_hybrid_smoke" / "best_4v2.pth"


def test_config_from_args_takes_v3_warm_start_defaults_and_explicit_overrides(tmp_path):
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path), "--init-v3-from", str(_SMOKE_V3_CKPT)])
    config = training.config_from_args(args)
    assert config["d_cap"] == 2.0
    assert config["hidden_dim"] == 32 and config["actor_hidden"] == 64
    assert config["map_size"] == 8.0 and config["future_steps"] == 10
    assert config["num_obstacles"] == 10 and config["hold_steps"] == 5
    assert config["warm_start_v3_from"] == str(_SMOKE_V3_CKPT.resolve())
    # Training-process arguments keep their own defaults, never the checkpoint's.
    assert config["episodes"] == 1000 and config["seed"] == 37 and config["batch_size"] == 64
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path), "--init-v3-from", str(_SMOKE_V3_CKPT),
         "--d-cap", "0.8", "--hidden-dim", "16"])
    config = training.config_from_args(args)
    assert config["d_cap"] == 0.8
    assert config["hidden_dim"] == 16
    assert config["actor_hidden"] == 64
    bad = tmp_path / "not_v3.pth"
    torch.save({"architecture_version": "continuous_membership_v2"}, bad)
    args = training.build_parser().parse_args(["--output", str(tmp_path), "--init-v3-from", str(bad)])
    with pytest.raises(ValueError, match="schema"):
        training.config_from_args(args)


def _tiny_warm_start_source(tmp_path):
    config = _tiny_config(tmp_path / "src")
    modules, optimizers, _ = training.build_system(config)
    with torch.no_grad():
        for parameter in modules["demand_head"].parameters():
            parameter.add_(1.0)
    payload = training.build_checkpoint_payload(
        modules, optimizers, config, episode=3, update_step=5)
    path = tmp_path / "warm_v3.pth"
    training.atomic_save_checkpoint(payload, path)
    return path, modules


def test_v3_warm_start_loads_networks_targets_and_optimizer_state(tmp_path):
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path), "--init-v3-from", str(_SMOKE_V3_CKPT)])
    config = training.config_from_args(args)
    modules, optimizers, metadata = training.build_system(config)
    payload = training.load_v3_hybrid_checkpoint(_SMOKE_V3_CKPT)
    for key, value in payload["demand_head"].items():
        torch.testing.assert_close(modules["demand_head"].state_dict()[key], value)
    for key, value in payload["encoder"].items():
        torch.testing.assert_close(modules["encoder"].state_dict()[key], value)
    # Target networks are snapshots of the loaded online weights.
    for name in ("actor_target", "critic1_target", "critic2_target", "value_head_target"):
        online = name.removesuffix("_target")
        for key, value in payload[online].items():
            torch.testing.assert_close(modules[name].state_dict()[key], value)
    assert metadata["warm_start_v3_from"]["path"] == str(_SMOKE_V3_CKPT.resolve())
    assert metadata["warm_start_v3_from"]["optimizer_state"] == "loaded"
    assert metadata["warm_start_v3_from"]["d_cap"] == config["d_cap"] == 2.0
    assert optimizers["actor"].state_dict()["state"]


def test_v3_warm_start_resets_mismatched_optimizer_and_records_reset(tmp_path):
    path, source = _tiny_warm_start_source(tmp_path)
    payload = training.load_v3_hybrid_checkpoint(path)
    payload["optimizers"]["actor"] = {
        "state": {},
        "param_groups": payload["optimizers"]["actor"]["param_groups"] * 2,
    }
    forged = tmp_path / "forged_optimizers.pth"
    torch.save(payload, forged)
    config = _tiny_config(tmp_path / "out", ("--init-v3-from", str(forged)))
    modules, _, metadata = training.build_system(config)
    for key, value in source["demand_head"].state_dict().items():
        torch.testing.assert_close(modules["demand_head"].state_dict()[key], value)
    assert metadata["warm_start_v3_from"]["optimizer_state"] == "reset"


def test_v3_warm_start_rejects_shape_mismatch_before_mutating_weights(tmp_path):
    config = _tiny_config(tmp_path, ("--init-v3-from", str(_SMOKE_V3_CKPT)))
    modules, _, _ = training.build_system(training.config_from_args(
        training.build_parser().parse_args(["--output", str(tmp_path / "ref")])))
    before = {key: value.clone() for key, value in modules["demand_head"].state_dict().items()}
    with pytest.raises(ValueError, match="incompatible"):
        training.build_system(config)
    for key, value in before.items():
        torch.testing.assert_close(modules["demand_head"].state_dict()[key], value)


def test_small_v3_warm_start_training_run_keeps_loaded_frozen_encoder(tmp_path):
    path, source = _tiny_warm_start_source(tmp_path)
    config = _tiny_config(tmp_path / "out", ("--init-v3-from", str(path)))
    training.run_training(config)
    payload = training.load_v3_hybrid_checkpoint(tmp_path / "out" / "last.pth")
    for key, value in source["encoder"].state_dict().items():
        torch.testing.assert_close(payload["encoder"][key], value)
    assert payload["metadata"]["warm_start_v3_from"]["path"] == str(path.resolve())
    assert payload["metadata"]["warm_start_v3_from"]["optimizer_state"] == "loaded"
    saved_config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert saved_config["warm_start_v3_from"] == str(path.resolve())
    assert saved_config["d_cap"] == config["d_cap"]


def test_d_cap_annealing_midpoint_and_evaluation_uses_final(tmp_path, monkeypatch):
    config = _tiny_config(tmp_path, ("--d-cap", "2.0", "--d-cap-final", "1.0",
                                    "--d-cap-anneal-episodes", "4"))
    assert training.annealed_d_cap(config, 0) == pytest.approx(2.0)
    assert training.annealed_d_cap(config, 2) == pytest.approx(1.5)
    assert training.annealed_d_cap(config, 4) == pytest.approx(1.0)
    assert training.annealed_d_cap(config, 100) == pytest.approx(1.0)
    plain = _tiny_config(tmp_path)
    assert training.annealed_d_cap(plain, 100) == plain["d_cap"]
    seen = []
    original = training.collect_episode

    def _spy(modules, cfg, scale, seed, exploration_rng=None, d_cap=None, counterfact=None):
        seen.append(d_cap)
        return original(modules, cfg, scale, seed, exploration_rng, d_cap=d_cap,
                        counterfact=counterfact)

    monkeypatch.setattr(training, "collect_episode", _spy)
    training.run_training(config)
    # episodes=2 with eval-interval=1: train e0, one eval per scale, train e1,
    # one eval per scale. Evaluation must always see the final d_cap value.
    assert seen[0] == pytest.approx(2.0)
    assert seen[1:4] == [pytest.approx(1.0)] * 3
    assert seen[4] == pytest.approx(1.75)
    assert seen[5:] == [pytest.approx(1.0)] * 3
    payload = training.load_v3_hybrid_checkpoint(tmp_path / "last.pth")
    assert payload["config"]["d_cap"] == 2.0
    assert payload["config"]["d_cap_final"] == 1.0
    assert payload["config"]["d_cap_anneal_episodes"] == 4


def test_d_cap_anneal_rejects_nonpositive_window_and_non_decreasing_final(tmp_path):
    with pytest.raises(ValueError):
        _tiny_config(tmp_path, ("--d-cap-final", "1.0", "--d-cap-anneal-episodes", "0"))
    with pytest.raises(ValueError):
        _tiny_config(tmp_path, ("--d-cap", "1.0", "--d-cap-final", "1.0"))
    with pytest.raises(ValueError):
        _tiny_config(tmp_path, ("--d-cap", "1.0", "--d-cap-final", "2.0"))


def test_d_cap_final_without_explicit_start_uses_default_or_checkpoint_value(tmp_path):
    config = _tiny_config(tmp_path, ("--d-cap-final", "1.5"))
    assert config["d_cap"] == 2.0
    assert config["d_cap_final"] == 1.5
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path), "--init-v3-from", str(_SMOKE_V3_CKPT),
         "--d-cap-final", "1.5"])
    config = training.config_from_args(args)
    assert config["d_cap"] == 2.0 and config["d_cap_final"] == 1.5
    args = training.build_parser().parse_args(
        ["--output", str(tmp_path), "--init-v3-from", str(_SMOKE_V3_CKPT),
         "--d-cap-final", "2.5"])
    with pytest.raises(ValueError):
        training.config_from_args(args)


def test_ring_radius_reaches_reward_config_and_environment(tmp_path):
    config = _tiny_config(tmp_path)
    assert config["reward_config"]["ring_radius"] == 1.5
    assert config["ring_radius"] == 1.5
    config = _tiny_config(tmp_path, ("--ring-radius", "2.5"))
    assert config["reward_config"]["ring_radius"] == 2.5
    assert config["ring_radius"] == 2.5
    assert training.make_environment(config).reward_config.ring_radius == 2.5


def test_policy_does_not_advance_capture_hold_outside_physical_step_and_repairs_dead_target(tmp_path):
    from .v2_env_adapter import apply_membership
    config = _tiny_config(tmp_path, ("--assign-interval", "100",))
    modules, _, _ = training.build_system(config)
    env = training.make_environment(config)
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=51)
    env.positions[:6] = np.array([
        [2.4, 3.0, 1.0], [3.0, 2.4, 1.0], [3.6, 3.0, 1.0], [3.0, 3.6, 1.0],
        [3.0, 3.0, 1.0], [7.0, 7.0, 1.0],
    ])
    env.velocities[:] = 0
    apply_membership(env, [0, 0, 0, 0])
    env.hold_counters[0] = env.hold_steps - 1
    policy = training.HybridPolicy(modules, config)
    policy.decision(env)
    assert not env.captured[0]
    assert env.hold_counters[0] == env.hold_steps - 1
    apply_membership(env, [0, 0, 0, 0])
    _, done, info = env.step(np.zeros((6, 2)), max_steps=10)
    assert not done and info["capture_events"] == [0]
    decision = policy.decision(env)
    assert decision["assignment"].tolist() == [1, 1, 1, 1]
    assert decision["allocation"]["demand"][0] == 0
    assert decision["allocation"]["capacity"].tolist() == [0, 4]


def test_execution_actor_switches_only_after_exactly_one_task_remains_alive():
    base = torch.nn.Identity()
    finisher = torch.nn.Identity()
    modules = {"actor": base, "post_capture_actor": finisher}

    assert training.select_execution_actor(modules, [True, True, False]) is base
    assert training.select_execution_actor(modules, [False, True, False]) is finisher
    assert training.select_execution_actor({"actor": base}, [False, True, False]) is base


def test_execution_actor_rejects_states_without_a_live_task():
    with pytest.raises(ValueError):
        training.select_execution_actor({"actor": torch.nn.Identity()}, [False, False])


def test_post_capture_ring_waits_for_delay_after_first_single_task_step():
    assert not training.should_use_post_capture_ring(
        [True, True], first_single_task_step=None, current_step=100,
        time_step=0.02, delay_seconds=1.0,
    )
    assert not training.should_use_post_capture_ring(
        [False, True], first_single_task_step=100, current_step=149,
        time_step=0.02, delay_seconds=1.0,
    )
    assert training.should_use_post_capture_ring(
        [False, True], first_single_task_step=100, current_step=150,
        time_step=0.02, delay_seconds=1.0,
    )


def test_post_capture_guidance_factory_keeps_slots_default_and_supports_continuous(tmp_path):
    from .traditional_ring_baseline import (
        ContinuousRingGuidanceController, RingGuidanceController,
    )
    env = training.make_environment(_tiny_config(tmp_path))
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=41)
    try:
        assert isinstance(
            training.build_post_capture_guidance(env, "slots"),
            RingGuidanceController,
        )
        assert isinstance(
            training.build_post_capture_guidance(env, "continuous"),
            ContinuousRingGuidanceController,
        )
        with pytest.raises(ValueError):
            training.build_post_capture_guidance(env, "unknown")
    finally:
        env.close()


def test_post_capture_ring_blend_preserves_actor_then_ramps_to_ring():
    actor = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    ring = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    before = training.blend_post_capture_ring_actions(
        actor, ring, elapsed_seconds=0.9, delay_seconds=1.0,
        transition_seconds=0.5,
    )
    midway = training.blend_post_capture_ring_actions(
        actor, ring, elapsed_seconds=1.25, delay_seconds=1.0,
        transition_seconds=0.5,
    )
    after = training.blend_post_capture_ring_actions(
        actor, ring, elapsed_seconds=1.5, delay_seconds=1.0,
        transition_seconds=0.5,
    )

    np.testing.assert_allclose(before, actor)
    np.testing.assert_allclose(midway, np.zeros_like(actor), atol=1e-7)
    np.testing.assert_allclose(after, ring)


def test_counterfact_pairwise_value_loss_matches_hand_computed_hinge_and_huber():
    predictions = torch.tensor([0.5, -0.2, 0.1])
    returns = torch.tensor([2.0, -1.0, 2.0])

    rank, reg = training.counterfact_pairwise_value_loss(predictions, returns, margin=0.1)

    # returns[0] == returns[2] contributes no pair, and the (0, 1) pair is
    # already correctly ordered, so the hinge term vanishes.
    assert rank == pytest.approx(0.0)
    assert reg == pytest.approx(F.smooth_l1_loss(predictions, returns))
    flipped = torch.tensor([-0.2, 0.5, 0.1])
    rank, _ = training.counterfact_pairwise_value_loss(flipped, returns, margin=0.1)
    expected = 0.5 * (max(0.0, 0.1 - (-0.2 - 0.5)) + max(0.0, 0.1 + (0.5 - 0.1)))
    assert rank == pytest.approx(expected)
    with pytest.raises(ValueError):
        training.counterfact_pairwise_value_loss(torch.zeros(2), torch.zeros(3), 0.1)


def test_counterfact_cli_records_group_and_rejects_invalid_values(tmp_path):
    config = _tiny_config(tmp_path)
    assert config["counterfact_h"] == 0
    assert config["counterfact_config"] == {
        "h": 0, "per_episode": 1, "modes": 3, "batch": 4,
        "rank_weight": 1.0, "reg_weight": 0.5, "margin": 0.1,
    }
    assert config["counterfact_batch"] == 4
    config = _tiny_config(tmp_path, (
        "--counterfact-h", "5", "--counterfact-per-episode", "2",
        "--counterfact-modes", "4", "--counterfact-batch", "8",
        "--counterfact-rank-weight", "0.7",
        "--counterfact-reg-weight", "0.3", "--counterfact-margin", "0.2"))
    assert config["counterfact_config"] == {
        "h": 5, "per_episode": 2, "modes": 4, "batch": 8,
        "rank_weight": 0.7, "reg_weight": 0.3, "margin": 0.2,
    }
    for extra in (("--counterfact-per-episode", "0"), ("--counterfact-modes", "0"),
                  ("--counterfact-batch", "0"), ("--counterfact-margin", "-0.1"),
                  ("--counterfact-rank-weight", "-1.0"),
                  ("--counterfact-reg-weight", "-0.5")):
        with pytest.raises(ValueError):
            _tiny_config(tmp_path, extra)


def test_counterfact_collect_episode_attaches_finite_samples_without_touching_off_default(tmp_path):
    config = _tiny_config(tmp_path, ("--counterfact-h", "2", "--counterfact-modes", "3"))
    modules, _, _ = training.build_system(config)
    rows, metrics = training.collect_episode(
        modules, config, (4, 2), 46, counterfact=dict(config["counterfact_config"]))

    cf_rows = [row for row in rows if "counterfact" in row]
    assert len(cf_rows) == 1
    sample = cf_rows[0]["counterfact"]
    assert 2 <= len(sample["samples"]) <= 3
    assignments = [tuple(item["assignment"]) for item in sample["samples"]]
    assert len(assignments) == len(set(assignments))
    for item in sample["samples"]:
        assert item["features"].ndim == 2 and item["features"].shape[0] == 2
        assert torch.isfinite(item["features"]).all()
        assert np.isfinite(item["discounted_return"])
        assert 0 < item["steps"] <= 2
    assert metrics["counterfact_ms"] >= 0.0
    # The branch state copies must not disturb the main episode record.
    assert len(rows) == metrics["steps"]

    off_rows, off_metrics = training.collect_episode(modules, config, (4, 2), 47)
    assert all("counterfact" not in row for row in off_rows)
    assert "counterfact_ms" not in off_metrics


def test_counterfact_branch_samples_cover_distinct_capacity_modes(monkeypatch):
    calls = []

    def _fake_rollout(env, encoder, actor, first_encoded, assignment,
                      horizon_steps, max_steps, gamma):
        calls.append(tuple(int(value) for value in assignment.tolist()))
        return {"discounted_return": float(len(calls)), "new_captures": 0,
                "steps": int(horizon_steps), "outcome": "running"}

    monkeypatch.setattr(training, "_rollout_locked_branch", _fake_rollout)
    encoded = {
        "pursuer_embeddings": torch.randn(6, 4),
        "task_embeddings": torch.randn(3, 4),
        "relation_costs": torch.randn(6, 3),
    }
    snapshot = {
        "num_pursuers": 4, "num_evaders": 2,
        "positions": np.random.default_rng(1).random((7, 3)),
        "task_mask": np.array([True, True, False]), "length": 8.0,
    }
    context = {
        "env": None, "encoded": encoded, "snapshot": snapshot,
        # The three highest scores are all 2+2 member permutations.  A
        # score-only selector therefore never observes whether 3+1 or 1+3
        # has the better real branch return.
        "candidates": np.array([
            [0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0],
            [0, 0, 0, 1], [0, 1, 1, 1],
        ]),
        "scores": np.array([0.95, 0.90, 0.85, 0.40, 0.30]), "row_index": 0,
    }
    spec = {"h": 1, "per_episode": 1, "modes": 3,
            "rank_weight": 1.0, "reg_weight": 0.5, "margin": 0.1}

    samples, alive = training._counterfact_branch_samples(
        {"encoder": None, "actor": None},
        {"capture_size": 3, "max_steps": 5, "gamma": 0.9}, context, spec)

    assert calls == [(0, 0, 1, 1), (0, 0, 0, 1), (0, 1, 1, 1)]
    assert alive == [True, True]
    assert [tuple(np.bincount(item["assignment"], minlength=2)) for item in samples] == [
        (2, 2), (3, 1), (1, 3),
    ]
    assert [item["discounted_return"] for item in samples] == [1.0, 2.0, 3.0]
    assert [item["capacity_mode"] for item in samples] == [[2, 2], [3, 1], [1, 3]]


def test_counterfact_disabled_by_default_makes_no_branch_calls_and_adds_no_keys(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        training, "_rollout_locked_branch", lambda *args, **kwargs: calls.append(args))
    config = _tiny_config(tmp_path)
    training.run_training(config)
    assert calls == []
    logs = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in logs if row["kind"] == "train"]
    assert train_rows and all("counterfact_ms" not in row for row in train_rows)
    assert all("counterfact_rank_loss" not in row["update"] for row in train_rows)


def test_counterfact_training_run_records_group_and_reports_value_head_terms(tmp_path):
    config = _tiny_config(tmp_path, (
        "--counterfact-h", "2", "--counterfact-modes", "2",
        "--counterfact-rank-weight", "0.7", "--counterfact-reg-weight", "0.3"))
    result = training.run_training(config)
    assert result["updates"] >= 2
    for name in ("counterfact_rank_loss", "counterfact_reg_loss"):
        assert name in result["last_update"]
        assert np.isfinite(result["last_update"][name])
    # The replay must feed every update: regression against branched G_H is
    # positive whenever samples were drawn (the dilution bug kept this at 0).
    assert result["last_update"]["counterfact_reg_loss"] > 0.0
    payload = training.load_v3_hybrid_checkpoint(tmp_path / "last.pth")
    assert payload["config"]["counterfact_config"] == config["counterfact_config"]
    assert payload["config"]["counterfact_h"] == 2
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["counterfact_config"]["modes"] == 2
    logs = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train_rows = [row for row in logs if row["kind"] == "train"]
    assert train_rows and all("counterfact_ms" in row for row in train_rows)


def _counterfact_state(tag, num_tasks=2, num_samples=3):
    return {
        "samples": [{
            "features": torch.full((num_tasks, 3), float(tag + index)),
            "discounted_return": float(tag - index),
            "new_captures": 0, "steps": 1, "outcome": "running",
            "assignment": [0, 0, 0, 0],
        } for index in range(num_samples)],
        "alive": [True] * num_tasks,
        "tag": tag,
    }


def test_counterfact_replay_stores_evicts_and_samples_deterministically():
    with pytest.raises(ValueError):
        training.CounterfactualReplay(max_states=0)
    replay = training.CounterfactualReplay(max_states=3, seed=5)
    assert len(replay) == 0
    assert replay.sample(2) == []
    with pytest.raises(ValueError):
        replay.sample(0)

    for tag in range(5):
        replay.add(_counterfact_state(tag))
    assert len(replay) == 3
    retained = sorted(state["tag"] for state in replay.sample(3))
    assert retained == [2, 3, 4]
    # States with no samples are ignored; misaligned masks are rejected.
    replay.add({"samples": [], "alive": [True, True]})
    assert len(replay) == 3
    with pytest.raises(ValueError):
        replay.add({"samples": _counterfact_state(9)["samples"], "alive": [True]})

    first = training.CounterfactualReplay(max_states=10, seed=7)
    second = training.CounterfactualReplay(max_states=10, seed=7)
    for tag in range(6):
        first.add(_counterfact_state(tag))
        second.add(_counterfact_state(tag))
    draw_a = first.sample(4)
    draw_b = second.sample(4)
    assert [state["tag"] for state in draw_a] == [state["tag"] for state in draw_b]
    assert len({state["tag"] for state in draw_a}) == 4


def test_reservoir_offer_fills_every_slot_before_replacement():
    first = []
    second = []
    rng_a = np.random.default_rng(70088)
    rng_b = np.random.default_rng(70088)
    for seen in range(1, 40):
        training._reservoir_offer(first, seen, seen, 2, rng_a)
        training._reservoir_offer(second, seen, seen, 2, rng_b)
        assert len(first) == min(seen, 2)
        assert all(item is not None for item in first)
    assert first == second


def test_td3_update_draws_counterfact_losses_from_replay_every_update(tmp_path):
    config = _tiny_config(tmp_path, ("--counterfact-h", "2"))
    modules, optimizers, _ = training.build_system(config)
    rows, _ = training.collect_episode(
        modules, config, (4, 2), 48, counterfact=dict(config["counterfact_config"]))
    replay = training.CounterfactualReplay(max_states=8, seed=1)
    for row in rows:
        if "counterfact" in row:
            replay.add(row["counterfact"])
    assert len(replay) >= 1
    batch = {"transitions": rows[:2], "importance_weights": np.ones(2)}
    for step in (1, 2, 3):
        metrics = training.td3_update(
            modules, optimizers, batch, config, update_step=step,
            counterfact_replay=replay)
        assert np.isfinite(metrics["counterfact_rank_loss"])
        assert 0.0 <= metrics["counterfact_pair_accuracy"] <= 1.0
        assert metrics["counterfact_distinct_modes"] >= 1.0
        # Huber regression against real branched returns stays positive even
        # as the value head trains; the rank hinge is nonnegative by construction.
        assert metrics["counterfact_reg_loss"] > 0.0
    empty = training.CounterfactualReplay(max_states=8, seed=1)
    metrics = training.td3_update(
        modules, optimizers, batch, config, update_step=4,
        counterfact_replay=empty)
    assert metrics["counterfact_rank_loss"] == 0.0
    assert metrics["counterfact_reg_loss"] == 0.0
    # No replay at all keeps the counterfactual loss at exact zero as well.
    metrics = training.td3_update(modules, optimizers, batch, config, update_step=5)
    assert metrics["counterfact_rank_loss"] == 0.0
    assert metrics["counterfact_reg_loss"] == 0.0


def test_value_only_update_freezes_policy_critics_and_demand(tmp_path):
    config = _tiny_config(tmp_path, (
        "--counterfact-h", "2", "--freeze-policy", "--freeze-demand",
    ))
    assert config["freeze_policy"] is True
    assert config["freeze_demand"] is True
    modules, optimizers, _ = training.build_system(config)
    rows, _ = training.collect_episode(
        modules, config, (4, 2), 58,
        counterfact=dict(config["counterfact_config"]),
    )
    replay = training.CounterfactualReplay(max_states=8, seed=1)
    for row in rows:
        if "counterfact" in row:
            replay.add(row["counterfact"])
    before = {
        name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
        if name in ("actor", "critic1", "critic2", "demand_head", "value_head")
    }
    batch = {"transitions": rows[:2], "importance_weights": np.ones(2)}
    metrics = training.td3_update(
        modules, optimizers, batch, config, update_step=2,
        counterfact_replay=replay,
    )
    for name in ("actor", "critic1", "critic2", "demand_head"):
        assert all(torch.equal(before[name][key], value)
                   for key, value in modules[name].state_dict().items())
    assert any(not torch.equal(before["value_head"][key], value)
               for key, value in modules["value_head"].state_dict().items())
    assert metrics["policy_frozen"] is True
    assert metrics["demand_frozen"] is True


def test_deploy5x5_protocol_installs_deployment_sampler(tmp_path):
    from .scenario_protocol_deploy5x5 import Deploy5x5Sampler

    config = _tiny_config(tmp_path, ("--scenario-protocol", "deploy5x5"))
    env = training.make_environment(config, d_cap=0.8)
    try:
        assert isinstance(env.scenario_sampler, Deploy5x5Sampler)
        assert env.d_cap == pytest.approx(0.8)
        assert env.v_max_e == pytest.approx(0.35)
    finally:
        env.close()
