"""Acceptance-contract tests for the 4v2 d_cap=0.8 experiment."""

import pytest
from pathlib import Path
from . import v3_08_acceptance as acceptance

from .v3_08_acceptance import (
    validate_protocol,
    validate_seed_splits,
    validate_evaluation_seeds,
    summarize_records,
)


def _config(**changes):
    config = {"scenario_protocol": "deploy5x5", "d_cap": 0.8,
              "d_cap_final": None, "map_size": 8.0,
              "num_obstacles": 10, "hold_steps": 5,
              "max_steps": 1000, "seed": 37}
    config.update(changes)
    return config


@pytest.mark.parametrize("change", [
    {"scenario_protocol": "stage"},
    {"d_cap": 2.0},
    {"d_cap_final": 2.0},
])
def test_protocol_rejects_wrong_scene_or_capture_radius(change):
    with pytest.raises(ValueError, match="protocol|d_cap"):
        validate_protocol(_config(**change))


def test_protocol_accepts_frozen_deploy5x5_geometry():
    assert validate_protocol(_config())["d_cap"] == 0.8


def test_seed_splits_are_disjoint_and_use_reserved_ranges():
    report = validate_seed_splits({
        "train": [72000, 72001], "validation": [73000],
        "test": [74000],
    })
    assert report == {"train": [72000, 72001], "validation": [73000],
                      "test": [74000]}
    with pytest.raises(ValueError, match="overlap|range"):
        validate_seed_splits({"train": [72000], "validation": [72000],
                              "test": [74000]})


@pytest.mark.parametrize("entry", [70086, {"seed": 72000, "parent_seed": 70086},
                                      {"seed": 72001, "derived_from": "70086-jitter"}])
def test_sealed_seed_and_derivatives_rejected_before_final(entry):
    with pytest.raises(ValueError, match="sealed"):
        validate_evaluation_seeds([entry], allocation_mode="online", split="train")
    with pytest.raises(ValueError, match="sealed"):
        validate_evaluation_seeds([entry], allocation_mode="oracle", split="validation")


def test_sealed_seed_requires_explicit_final_mode():
    assert validate_evaluation_seeds([70086], allocation_mode="sealed_final") == [70086]
    with pytest.raises(ValueError, match="sealed"):
        validate_evaluation_seeds([70086], allocation_mode="both")


def test_duplicate_and_wrong_range_seeds_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_evaluation_seeds([73000, 73000], "online", "validation")
    with pytest.raises(ValueError, match="range"):
        validate_evaluation_seeds([72000], "online", "validation")


def test_summary_reports_capture_and_geometry_for_each_mode():
    rows = [
        {"seed": 74000, "mode": "online", "captured": 1,
         "first_capture_time": 2.0, "full_capture_time": None,
         "min_max_assigned_member_distance": 0.7,
         "min_angular_gap": 2.8, "switches": 2},
        {"seed": 74001, "mode": "online", "captured": 2,
         "first_capture_time": 1.0, "full_capture_time": 3.0,
         "min_max_assigned_member_distance": 0.6,
         "min_angular_gap": 2.6, "switches": 1},
    ]
    summary = summarize_records(rows)["online"]
    assert summary["any_capture_rate"] == 1.0
    assert summary["full_capture_rate"] == 0.5
    assert summary["mean_switches"] == 1.5
    assert summary["mean_first_capture_time_success_only"] == 1.5
    assert summary["mean_full_capture_time_success_only"] == 3.0
    assert summary["mean_min_max_assigned_member_distance"] == pytest.approx(0.65)


def test_evaluate_checkpoint_reports_deterministic_seeds_and_oracle(monkeypatch):
    monkeypatch.setattr(acceptance, "_load_checkpoint", lambda path: {"config": _config()})
    monkeypatch.setattr(acceptance, "_load_modules", lambda path, payload: object())

    def run_episode(modules, config, seed, mode, assignment=None):
        captured = 2 if assignment == (0, 0, 0, 1) else 1
        return {"seed": seed, "mode": mode, "captured": captured,
                "first_capture_time": 1.0, "full_capture_time": 2.0 if captured == 2 else None,
                "min_max_assigned_member_distance": 0.6,
                "min_angular_gap": 2.0, "switches": 0, "reward": 0.0}

    monkeypatch.setattr(acceptance, "_run_episode", run_episode)
    result = acceptance.evaluate_checkpoint("model.pth", [74001, 74000], "both")
    assert result["seeds"] == [74000, 74001]
    assert len(result["records"]) == 4
    assert result["summary"]["oracle"]["full_capture_rate"] == 1.0
    assert all(row["initial_assignment"] == [0, 0, 0, 1]
               for row in result["records"] if row["mode"] == "oracle")


def test_real_v3_environment_step_contract():
    checkpoint = (Path(__file__).resolve().parents[5] /
                  "checkpoints/v3_cf_value_fix_4v2_20261118/deploy5x5_smoke/best_4v2.pth")
    if not checkpoint.exists():
        pytest.skip("local V3 smoke checkpoint unavailable")
    payload = acceptance._load_checkpoint(checkpoint)
    modules = acceptance._load_modules(checkpoint, payload)
    config = dict(payload["config"], max_steps=2)
    row = acceptance._run_episode(modules, config, 74000, "online")
    assert row["seed"] == 74000
    assert row["steps"] == 2
