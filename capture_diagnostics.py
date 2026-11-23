"""Read-only capture diagnostics shared by evaluation tools."""

from __future__ import annotations

import numpy as np

from .stage_env import evader_enclosed_by_coalition


def target_capture_diagnostics(env, assignment) -> list[dict]:
    assignment = np.asarray(assignment, dtype=np.int64)
    pursuers = env.pursuer_positions()
    evaders = env.evader_positions()
    rows = []
    for target in range(env.num_evaders):
        members = [
            i for i in range(env.num_pursuers)
            if bool(env.pursuer_mask[i])
            and i < len(assignment)
            and int(assignment[i]) == target
        ]
        all_distances = np.linalg.norm(pursuers - evaders[target], axis=1)
        nearest_three = np.sort(all_distances)[:3]
        member_distances = (
            np.linalg.norm(pursuers[members] - evaders[target], axis=1)
            if members else np.asarray([], dtype=np.float64)
        )
        enclosed = bool(
            len(members) >= 3
            and evader_enclosed_by_coalition(pursuers[members], evaders[target])
        )
        hold_counter = int(env.hold_counters[target])
        rows.append({
            "target": target,
            "active": bool(env.task_mask[target]),
            "captured": bool(env.captured[target]),
            "members": members,
            "member_count": len(members),
            "nearest_three_distances": nearest_three.tolist(),
            "max_member_distance": (
                float(member_distances.max()) if len(member_distances) else None
            ),
            "enclosed": enclosed,
            "hold_counter": hold_counter,
            "hold_fraction": min(1.0, hold_counter / max(1, int(env.hold_steps))),
            "capture_hold_ok": bool(env._capture_hold_ok(target)),
            "distance_to_goal": float(
                np.linalg.norm(env.evader_goals[target] - evaders[target])
            ),
        })
    return rows
