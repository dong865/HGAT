"""Locked-assignment V3 runtime contracts (TDD)."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from .config import SixVTwoConfig
from .runtime_v3_locked import V3LockedRuntime
from .stage_env import StageCoalitionEnv


SMOKE_CHECKPOINT = (Path(__file__).resolve().parents[5]
                    / "checkpoints" / "v3_hybrid_smoke" / "best_4v2.pth")


def _runtime(**kwargs):
    return V3LockedRuntime(
        str(SMOKE_CHECKPOINT), config=SixVTwoConfig(num_pursuers=4, num_evaders=2),
        device="cpu", scenario_seed=51, **kwargs,
    )


def _world_state(env):
    positions = env.positions[:6].copy()
    positions[:, :2] -= env.length / 2.0
    return positions, env.velocities[:6].copy()


def test_locked_assignment_is_enforced_without_online_allocation():
    runtime = _runtime()

    def forbidden_allocate(*args, **kwargs):
        raise AssertionError("locked runtime must not call the online allocator")

    runtime.allocator.allocate = forbidden_allocate
    positions, velocities = _world_state(runtime.env)
    for _ in range(3):
        command, info = runtime.act(positions, velocities)
        assert info["assignment"] == [1, 0, 1, 1]
        np.testing.assert_array_equal(runtime.env.assignment[:4], [1, 0, 1, 1])
        assert np.isfinite(command).all()


def test_dead_task_members_retarget_the_remaining_live_task():
    runtime = _runtime(lock_assignment=[1, 0, 1, 1])
    runtime.env.task_mask[:2] = [False, True]
    positions, velocities = _world_state(runtime.env)

    _, info = runtime.act(positions, velocities)

    assert info["assignment"] == [1, 1, 1, 1]
    np.testing.assert_array_equal(runtime.env.assignment[:4], [1, 1, 1, 1])

    runtime.reset()
    runtime.env.task_mask[:2] = [True, False]
    positions, velocities = _world_state(runtime.env)
    _, info = runtime.act(positions, velocities)
    assert info["assignment"] == [0, 0, 0, 0]


def test_act_output_is_finite_speed_bounded_and_acceleration_limited():
    runtime = _runtime(max_acceleration=1.5)
    positions, velocities = _world_state(runtime.env)
    command, _ = runtime.act(positions, velocities)
    next_command, _ = runtime.act(positions, velocities)

    assert command.shape == (6, 2)
    assert np.isfinite(next_command).all()
    assert np.linalg.norm(command[:4], axis=1).max() <= runtime.env.v_max + 1e-6
    delta = np.linalg.norm(next_command[:4] - command[:4], axis=1)
    assert delta.max() <= 1.5 * runtime.env.time_step + 1e-6


def _import_real_flight(monkeypatch):
    stub = types.ModuleType("crazyflie_py")
    stub.Crazyswarm = type("Crazyswarm", (), {})
    monkeypatch.setitem(sys.modules, "crazyflie_py", stub)
    mocap_stub = types.ModuleType("crazyflie_examples.six_v_two_hgat.mocap_bridge")
    mocap_stub.Nokov = type("Nokov", (), {})
    monkeypatch.setitem(sys.modules,
                        "crazyflie_examples.six_v_two_hgat.mocap_bridge", mocap_stub)
    from . import real_flight
    return real_flight


class _RecordingLockedRuntime:
    constructed = None

    def __init__(self, checkpoint, **kwargs):
        type(self).constructed = kwargs
        self.env = StageCoalitionEnv(length=8.0, num_obstacle=0,
                                     max_pursuers=6, max_evaders=3)
        self.env.reset(num_pursuers=4, num_evaders=2, scenario_seed=73016)
        self.num_pursuers = 4
        self.num_evaders = 2
        self.recenter_scene = False

    def reset(self):
        pass


def test_real_flight_locked_runtime_branch_passes_assignment(monkeypatch):
    real_flight = _import_real_flight(monkeypatch)
    monkeypatch.setattr(real_flight, "V3LockedRuntime", _RecordingLockedRuntime)
    monkeypatch.setattr(sys, "argv", [
        "real_flight.py", "dummy.pth", "--scale", "4v2", "--plan-only",
        "--roles", "cf1,cf2,cf3,cf4,cf5,cf6",
        "--runtime", "v3hybrid_locked", "--lock-assignment", "1", "0", "1", "1",
        "--scenario-seed", "73016", "--d-cap", "0.8",
    ])

    real_flight.main()

    kwargs = _RecordingLockedRuntime.constructed
    assert kwargs["lock_assignment"] == [1, 0, 1, 1]
    assert kwargs["scenario_seed"] == 73016
