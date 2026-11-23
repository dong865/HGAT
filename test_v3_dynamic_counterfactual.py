import copy

import numpy as np
import torch

from .stage_env import StageCoalitionEnv
from .v3_dynamic_counterfactual import (
    counterfactual_value_loss, fork_environment, summarize_branch,
)


def test_fork_environment_replays_identically_without_mutating_source():
    env = StageCoalitionEnv(seed=3)
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=41)
    original_positions = env.positions.copy()
    left = fork_environment(env)
    right = fork_environment(env)
    actions = np.zeros((env.max_pursuers, 2), dtype=np.float64)

    left_result = left.step(actions, max_steps=20)
    right_result = right.step(actions, max_steps=20)

    assert left_result[0] == right_result[0]
    assert left_result[1] == right_result[1]
    np.testing.assert_allclose(left.positions, right.positions)
    np.testing.assert_allclose(env.positions, original_positions)
    assert env.episode_step_count == 0


def test_summarize_branch_prioritizes_capture_then_earlier_time_then_return():
    result = summarize_branch(
        rewards=[-1.0, -2.0], captured_before=0, captured_after=1,
        elapsed_steps=2, time_step=0.02, first_capture_step=1,
    )

    assert result["new_captures"] == 1
    assert result["first_capture_delay"] == 0.02
    assert result["discounted_return"] == -3.0
    assert result["ranking_key"] > summarize_branch(
        rewards=[100.0], captured_before=0, captured_after=0,
        elapsed_steps=1, time_step=0.02, first_capture_step=None,
    )["ranking_key"]


def test_counterfactual_value_loss_orders_same_state_candidates():
    predictions = torch.tensor([0.0, 0.0], requires_grad=True)
    branch_results = [
        {"ranking_key": 0.0, "discounted_return": -3.0},
        {"ranking_key": 10.0, "discounted_return": 2.0},
    ]

    loss = counterfactual_value_loss(
        predictions, branch_results, rank_margin=1.0,
    )
    loss.backward()

    assert loss.item() > 0
    assert predictions.grad[1] < 0
    assert predictions.grad[0] > 0
