from .evaluate_v3_mode_demand import summarize_paired


def test_paired_summary_reports_capture_and_switch_deltas():
    rows = [
        {"seed": 1, "method": "base", "any_capture": 0.0,
         "full_capture": 0.0, "mean_captured": 0.0, "switch_count": 1},
        {"seed": 1, "method": "mode", "any_capture": 1.0,
         "full_capture": 0.0, "mean_captured": 1.0, "switch_count": 2},
        {"seed": 2, "method": "base", "any_capture": 1.0,
         "full_capture": 1.0, "mean_captured": 2.0, "switch_count": 3},
        {"seed": 2, "method": "mode", "any_capture": 1.0,
         "full_capture": 1.0, "mean_captured": 2.0, "switch_count": 2},
    ]

    summary = summarize_paired(rows)

    assert summary["paired_seeds"] == 2
    assert summary["mode_minus_base"]["any_capture_rate"] == 0.5
    assert summary["mode_minus_base"]["mean_captured"] == 0.5
    assert summary["mode_minus_base"]["mean_switch_count"] == 0.0
