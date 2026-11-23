from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from .capture_diagnostics import target_capture_diagnostics
from .main_continuous_v2 import build_env, model_selection_key
from .stage_env import StageCoalitionEnv
from .stage_networks import StageFeatureEncoder
from .v2_training import build_v2_obs_torch, hard_K_from_assignment
from .v2_env_adapter import apply_membership


def test_target_capture_diagnostics_exposes_geometry_and_hold_progress():
    env = StageCoalitionEnv(
        length=8.0,
        num_obstacle=0,
        max_pursuers=6,
        max_evaders=3,
        d_cap=2.0,
        hold_steps=5,
        seed=3,
    )
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=50000)
    assignment = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    apply_membership(env, assignment)

    rows = target_capture_diagnostics(env, assignment)

    assert len(rows) == 2
    assert set(rows[0]) == {
        "target", "active", "captured", "members", "member_count",
        "nearest_three_distances", "max_member_distance", "enclosed",
        "hold_counter", "hold_fraction", "capture_hold_ok",
        "distance_to_goal",
    }
    assert rows[0]["members"] == [0, 1, 2]
    assert rows[1]["members"] == [3, 4, 5]
    assert 0.0 <= rows[0]["hold_fraction"] <= 1.0


def test_training_env_applies_evader_speed_to_potential_field():
    args = SimpleNamespace(
        map_size=8.0,
        num_obstacles=0,
        d_cap=2.0,
        hold_steps=5,
        seed=7,
        evader_speed=0.15,
        closure_weight=0.5,
        closure_band=2.5,
        ring_weight=0.0,
        ring_radius=1.5,
        angular_closure_weight=0.3,
        angular_closure_band=1.5,
    )
    env = build_env(args)
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=50000)

    speeds = [np.linalg.norm(env._evader_velocity(j)) for j in range(2)]

    assert max(speeds) <= 0.150001
    assert env.scenario_config.max_speed == 0.15


def test_training_membership_count_matches_hard_execution_count():
    env = StageCoalitionEnv(
        length=8.0, num_obstacle=0, max_pursuers=6, max_evaders=3,
        d_cap=2.0, hold_steps=5, seed=11,
    )
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=50000)
    assignment = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    apply_membership(env, assignment)
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=0, map_size=8.0,
        hidden_dim=32, future_steps=10, seq_len=10, node_dim=9,
        device="cpu", use_ring_slot_features=False,
        use_stage_embedding=False, use_role_embedding=False,
        use_graph_embedding=True,
    )
    encoder.eval()
    snap = env.snapshot()
    enc_out = encoder.encode(snap, differentiable=False)
    hard_k = hard_K_from_assignment(
        assignment, snap["task_mask"], 6, 3, encoder.device,
    )

    obs = build_v2_obs_torch(enc_out, hard_k, snap, 6, 3)

    membership_count_index = 10 + 3 * 5 + 3 + 1
    expected = np.full(6, 3.0 / 6.0)
    actual = obs[:6, membership_count_index].detach().cpu().numpy()
    assert np.allclose(actual, expected)


def test_model_selection_prefers_single_capture_then_reward_on_full_tie():
    low_single = {
        "capture_rate": 0.0,
        "target_capture_rate": 0.1,
        "partial_capture_rate": 0.1,
        "mean_reward": -50.0,
    }
    high_single = {
        "capture_rate": 0.0,
        "target_capture_rate": 0.3,
        "partial_capture_rate": 0.3,
        "mean_reward": -200.0,
    }
    same_capture_better_reward = dict(low_single, mean_reward=-20.0)

    assert model_selection_key(high_single) > model_selection_key(low_single)
    assert model_selection_key(same_capture_better_reward) > model_selection_key(low_single)
