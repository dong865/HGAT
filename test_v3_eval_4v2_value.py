import numpy as np

from .coalition_value import generate_candidate_assignments
from .v3_eval_4v2_value import (
    choose_balanced_assignment,
    pad_actions_for_env,
    repair_assignment_for_active_tasks,
    summarize_records,
)


def test_balanced_baseline_is_two_plus_two_and_keeps_tasks_nonempty():
    candidates = np.stack(generate_candidate_assignments(4, 2))

    assignment = choose_balanced_assignment(candidates, num_tasks=2)

    assert tuple(np.bincount(assignment, minlength=2)) == (2, 2)


def test_lifecycle_repair_moves_every_agent_to_only_active_task():
    initial = np.array([0, 0, 0, 1], dtype=np.int64)

    repaired = repair_assignment_for_active_tasks(
        initial, np.array([False, True]),
    )

    assert repaired.tolist() == [1, 1, 1, 1]
    assert repair_assignment_for_active_tasks(
        initial, np.array([True, True]),
    ).tolist() == initial.tolist()


def test_four_agent_actions_are_zero_padded_to_environment_capacity():
    actions = np.ones((4, 2), dtype=np.float32)

    padded = pad_actions_for_env(actions, max_pursuers=6)

    assert padded.shape == (6, 2)
    assert np.all(padded[:4] == 1.0)
    assert np.all(padded[4:] == 0.0)


def test_summary_uses_paired_seeds_and_counts_capture_wins():
    records = [
        {"seed": 10, "method": "value_top1", "captured": 1,
         "first_capture_time": 4.0, "full_capture_time": None, "reward": 2.0},
        {"seed": 10, "method": "balanced_2plus2", "captured": 0,
         "first_capture_time": None, "full_capture_time": None, "reward": -2.0},
        {"seed": 11, "method": "value_top1", "captured": 0,
         "first_capture_time": None, "full_capture_time": None, "reward": -3.0},
        {"seed": 11, "method": "balanced_2plus2", "captured": 1,
         "first_capture_time": 5.0, "full_capture_time": None, "reward": 1.0},
        {"seed": 12, "method": "value_top1", "captured": 1,
         "first_capture_time": 3.0, "full_capture_time": None, "reward": 3.0},
        {"seed": 12, "method": "balanced_2plus2", "captured": 1,
         "first_capture_time": 3.0, "full_capture_time": None, "reward": 3.0},
    ]

    summary = summarize_records(records)

    assert summary["methods"]["value_top1"]["single_capture_rate"] == 2 / 3
    assert summary["paired_value_vs_balanced"] == {
        "wins": 1, "ties": 1, "losses": 1, "paired_seeds": 3,
    }


def test_summary_uses_null_when_no_capture_time_exists():
    summary = summarize_records([
        {"seed": 1, "method": "value_top1", "captured": 0,
         "first_capture_time": None, "full_capture_time": None, "reward": 0.0},
    ])

    metrics = summary["methods"]["value_top1"]
    assert metrics["mean_first_capture_time_success_only"] is None
    assert metrics["mean_full_capture_time_success_only"] is None
