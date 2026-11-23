from .v3_ab_scheme2_vs_scheme3_4v2 import compare_paired_records


def test_paired_comparison_prioritizes_capture_then_time_then_reward():
    rows = [
        {"seed": 1, "method": "scheme2", "captured": 1,
         "first_capture_time": 5.0, "reward": 0.0},
        {"seed": 1, "method": "scheme3", "captured": 0,
         "first_capture_time": None, "reward": 100.0},
        {"seed": 2, "method": "scheme2", "captured": 1,
         "first_capture_time": 7.0, "reward": 0.0},
        {"seed": 2, "method": "scheme3", "captured": 1,
         "first_capture_time": 6.0, "reward": -10.0},
        {"seed": 3, "method": "scheme2", "captured": 0,
         "first_capture_time": None, "reward": 1.0},
        {"seed": 3, "method": "scheme3", "captured": 0,
         "first_capture_time": None, "reward": 1.0},
    ]

    result = compare_paired_records(rows)

    assert result == {
        "scheme2_wins": 1,
        "scheme3_wins": 1,
        "ties": 1,
        "paired_seeds": 3,
    }
