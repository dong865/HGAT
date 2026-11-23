"""Real-flight runtime adapter contracts for the V3 hybrid allocator (TDD).

parity 说明（与 train_v3_hybrid.HybridPolicy.decision 对齐口径）：
  * 训练侧每个物理步是 ``decision = policy.decision(env)``（读当前快照，
    ``env.episode_step_count == k`` 第 k 步），再用 ``env.step(actions)`` 推物理；
    生命周期在 env.step 内部、动力学之后推进。
  * 实机侧没有 env.step：runtime.act() 每控制周期把动捕状态写进 env、
    推进一次 ``advance_lifecycle()``（与 runtime_stage 相同契约），并把
    ``env.episode_step_count`` 镜像成决策步序号 k（训练侧第 k 次 decision 的计数）。
    因此 encode 的 GRU 缓存、``assign_interval`` 分配触发步、gate 的 dwell 步
    与训练口径**逐步精确对齐**，不存在相位差。
  * 首帧是训练侧 step 0 的决策状态，不推进 lifecycle；后续每个动捕帧为上一
    控制周期后的新物理状态，恰好推进一次。捕获保持计数与冻结门控因此逐步对齐。
"""

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

torch.set_num_threads(1)  # 训练作业在后台跑，测试保持轻量

from .config import SixVTwoConfig
from .stage_env import StageCoalitionEnv, MODE_EXIT
from .safety import filter_velocity
from .train_v3_hybrid import (HybridPolicy, load_v3_hybrid_checkpoint,
                              make_environment)
from .v2_controller import v2_obs_dim
from .v3_demand_distillation import ScaleAwareCoalitionValueHead
from .v3_sparse_demand import SparseDemandHead
from .stage_networks import StageFeatureEncoder, SharedActor
from .runtime_v3_hybrid import V3HybridRuntime
from . import runtime_v3_hybrid

SMOKE_CHECKPOINT = (Path(__file__).resolve().parents[5]
                    / "checkpoints" / "v3_hybrid_smoke" / "best_4v2.pth")
PARITY_SEED = 51
PARITY_STEPS = 12  # 跨过 smoke ckpt 的 assign_interval=10 触发点


def _config_4v2():
    return SixVTwoConfig(num_pursuers=4, num_evaders=2)


def _runtime(obstacles=(), **kwargs):
    kwargs.setdefault("scenario_seed", PARITY_SEED)
    return V3HybridRuntime(str(SMOKE_CHECKPOINT), config=_config_4v2(),
                           device="cpu", obstacles=obstacles, **kwargs)


def _world_state(env, origin):
    """arena 状态 -> 世界系（与 runtime.act 内的换算互逆）。"""
    n = env.num_pursuers + env.num_evaders
    world = env.positions[:n].copy()
    world[:, :2] = world[:, :2] + np.asarray(origin, dtype=np.float32)[None, :2] - env.length / 2.0
    return world, env.velocities[:n].copy()


def test_act_emits_finite_commands_bounded_by_v_max():
    runtime = _runtime()
    origin = np.zeros(2, dtype=np.float32)
    positions_world, velocities_world = _world_state(runtime.env, origin)
    command, info = runtime.act(positions_world, velocities_world, origin)
    # 完整契约与 StageCoalitionRuntime.act 对齐：6 机（4 追捕 + 2 逃逸）
    assert command.shape == (6, 2)
    # 任务书里的 (4,2) 指追捕机指令块
    pursuer_block = command[:4]
    assert pursuer_block.shape == (4, 2)
    assert np.isfinite(command).all()
    speed = np.linalg.norm(pursuer_block, axis=1)
    assert speed.max() <= runtime.env.v_max + 1e-6
    assert len(info["assignment"]) == 4
    assert info["episode_step"] == 0
    assert info["allocated"] is True  # 首步 gate 无 assignment -> 强制分配


def test_act_rejects_bad_shapes_and_nonfinite_state():
    runtime = _runtime()
    origin = np.zeros(2, dtype=np.float32)
    good_p, good_v = _world_state(runtime.env, origin)
    with pytest.raises(ValueError):
        runtime.act(good_p[:5], good_v[:5], origin)
    bad = good_p.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        runtime.act(bad, good_v, origin)


def test_reset_restores_initial_commands_and_allocator_gate():
    runtime = _runtime()
    origin = np.zeros(2, dtype=np.float32)
    first_p, first_v = _world_state(runtime.env, origin)
    cmd_a, _ = runtime.act(first_p, first_v, origin)
    runtime.act(first_p, first_v, origin)
    runtime.reset()
    assert runtime._step == 0
    assert runtime.allocator.gate.assignment is None
    assert np.all(runtime.command_velocity == 0.0)
    cmd_b, info_b = runtime.act(first_p, first_v, origin)
    # reset 后同一切始状态 -> 相同指令（encoder/allocator 门控全部清零）
    np.testing.assert_allclose(cmd_b, cmd_a, atol=1e-6)
    assert info_b["allocated"] is True


def test_runtime_resets_optional_post_capture_ring_state():
    runtime = _runtime(post_capture_ring_delay=1.0)
    assert runtime.post_capture_ring_delay == pytest.approx(1.0)
    runtime.first_single_task_step = 12
    runtime.post_capture_ring = object()

    runtime.reset()

    assert runtime.first_single_task_step is None
    assert runtime.post_capture_ring is None


def test_runtime_loads_post_capture_ring_timing_from_checkpoint_config(tmp_path):
    payload = torch.load(SMOKE_CHECKPOINT, map_location="cpu", weights_only=False)
    payload["config"] = dict(payload["config"])
    payload["config"]["post_capture_ring_delay"] = 1.0
    payload["config"]["post_capture_ring_transition"] = 0.5
    payload["config"]["freeze_distance_ratio"] = 1.25
    payload["config"]["freeze_release_ratio"] = 1.2
    checkpoint = tmp_path / "ring_timing.pth"
    torch.save(payload, checkpoint)

    runtime = V3HybridRuntime(
        str(checkpoint), config=_config_4v2(), device="cpu",
        scenario_seed=PARITY_SEED,
    )

    assert runtime.post_capture_ring_delay == pytest.approx(1.0)
    assert runtime.post_capture_ring_transition == pytest.approx(0.5)
    assert runtime.allocator.freeze_guard.freeze_distance == pytest.approx(
        1.25 * runtime.d_cap)
    assert runtime.allocator.freeze_guard.release_distance == pytest.approx(
        1.2 * 1.25 * runtime.d_cap)


def test_first_mocap_frame_does_not_advance_capture_lifecycle():
    runtime = _runtime()
    target = np.array([4.0, 4.0, 1.0], dtype=np.float32)
    runtime.env.positions[4] = target
    runtime.env.positions[:3] = np.array([
        [5.0, 4.0, 1.0],
        [3.5, 4.8660254, 1.0],
        [3.5, 3.1339746, 1.0],
    ], dtype=np.float32)
    runtime.env.assignment[:4] = np.array([0, 0, 0, 1])
    assert runtime.env._capture_hold_ok(0)
    positions, velocities = _world_state(runtime.env, np.zeros(2, dtype=np.float32))

    runtime.act(positions, velocities, np.zeros(2, dtype=np.float32))

    assert runtime.env.hold_counters[0] == 0


def test_runtime_uses_checkpoint_scenario_protocol_sampler(tmp_path):
    payload = torch.load(SMOKE_CHECKPOINT, map_location="cpu", weights_only=False)
    payload["config"] = dict(payload["config"])
    payload["config"]["scenario_protocol"] = "deploy5x5"
    checkpoint = tmp_path / "deploy_protocol.pth"
    torch.save(payload, checkpoint)
    config = payload["config"]
    expected = make_environment(config)
    expected.reset(num_pursuers=4, num_evaders=2, scenario_seed=PARITY_SEED)

    runtime = V3HybridRuntime(
        str(checkpoint), config=_config_4v2(), device="cpu",
        scenario_seed=PARITY_SEED,
    )

    np.testing.assert_allclose(
        runtime.env.evader_goals[:2], expected.evader_goals[:2], atol=1e-7,
    )


def test_switch_smoother_blends_only_changed_pursuers_for_configured_window():
    """Deleting transition interpolation must expose an immediate command jump."""
    assert hasattr(runtime_v3_hybrid, "AssignmentTransitionSmoother")
    smoother = runtime_v3_hybrid.AssignmentTransitionSmoother(duration=0.4)
    old = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    first_assignment = np.array([0, 1])
    np.testing.assert_allclose(
        smoother.blend(old, first_assignment, old, dt=0.1), old
    )

    desired = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    changed = smoother.blend(desired, np.array([1, 1]), old, dt=0.1)
    np.testing.assert_allclose(changed[0], [0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(changed[1], desired[1], atol=1e-6)

    for expected in ([0.0, 0.0], [-0.5, 0.0], [-1.0, 0.0]):
        blended = smoother.blend(desired, np.array([1, 1]), old, dt=0.1)
        np.testing.assert_allclose(blended[0], expected, atol=1e-6)


def test_vector_acceleration_limit_bounds_norm_instead_of_each_axis():
    """Per-axis clipping would exceed the requested Euclidean acceleration."""
    assert hasattr(runtime_v3_hybrid, "limit_vector_acceleration")
    previous = np.zeros((2, 2), dtype=np.float32)
    desired = np.array([[1.0, 1.0], [0.1, 0.0]], dtype=np.float32)
    limited = runtime_v3_hybrid.limit_vector_acceleration(
        desired, previous, max_acceleration=2.0, dt=0.1,
    )
    assert np.linalg.norm(limited[0] - previous[0]) == pytest.approx(0.2)
    np.testing.assert_allclose(limited[1], desired[1])


def test_strict_loader_loads_only_policy_heads_and_no_critic():
    runtime = _runtime()
    assert tuple(runtime.loaded_modules) == (
        "st_hgat", "relation_head", "demand_head", "value_head", "actor")
    for name in ("critic1", "critic2", "actor_target", "critic1_target",
                 "critic2_target", "value_head_target"):
        assert name not in runtime.__dict__, f"{name} must not be constructed"
    assert "optimizers" not in runtime.__dict__
    assert isinstance(runtime.actor, torch.nn.Module)
    assert isinstance(runtime.env, StageCoalitionEnv)
    assert runtime.env.num_pursuers == 4 and runtime.env.num_evaders == 2
    assert runtime.map_size == pytest.approx(8.0)
    assert runtime.d_cap == pytest.approx(2.0)
    assert runtime.hold_steps == 5


def test_runtime_optionally_loads_external_mode_demand_without_replacing_base(tmp_path):
    reference = _runtime()
    with torch.no_grad():
        for parameter in reference.demand_head.mode_net.parameters():
            parameter.fill_(0.25)
    external = tmp_path / "mode_demand.pth"
    torch.save({
        "architecture_version": "v3_counterfactual_mode_demand_4v2_v1",
        "base_checkpoint": str(SMOKE_CHECKPOINT.resolve()),
        "hidden_dim": reference.demand_head.hidden_dim,
        "head_width": 64,
        "demand_head": reference.demand_head.state_dict(),
    }, external)

    runtime = _runtime(
        mode_demand_checkpoint=str(external), mode_prior_weight=0.2,
    )

    assert runtime.mode_demand_checkpoint == external.resolve()
    assert runtime.allocator.mode_prior_weight == pytest.approx(0.2)
    for parameter in runtime.demand_head.mode_net.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))


def _build_policy_modules(checkpoint_path, device="cpu"):
    """与 train_v3_hybrid.build_system 同构的轻量装载（不碰 warm-start 源）。"""
    payload = load_v3_hybrid_checkpoint(checkpoint_path)
    config = payload["config"]
    hidden = int(config["hidden_dim"])
    encoder = StageFeatureEncoder(
        6, 3, int(config["num_obstacles"]), float(config["map_size"]), hidden,
        int(config["future_steps"]), 10, 9, torch.device(device),
        use_role_embedding=False, use_stage_embedding=False,
        use_graph_embedding=True, use_ring_slot_features=False,
    ).to(device)
    encoder._skip_legacy_plan = True
    modules = {
        "encoder": encoder,
        "demand_head": SparseDemandHead(hidden, width=int(config["head_width"])).to(device),
        "value_head": ScaleAwareCoalitionValueHead(hidden, width=int(config["head_width"])).to(device),
        "actor": SharedActor(v2_obs_dim(6, hidden, int(config["future_steps"])),
                             hidden_dim=int(config["actor_hidden"])).to(device),
    }
    load_v3_hybrid_checkpoint(checkpoint_path, modules=modules)
    for module in modules.values():
        module.eval()
    return modules, config


def _shadow_vector_limit(command, previous, max_acceleration, dt):
    """Hand-derived Euclidean acceleration projection for parity testing."""
    delta = command - previous
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_acceleration * dt / np.maximum(norm, 1e-12))
    return previous + delta * scale


def test_parity_with_training_hybrid_policy_multi_step():
    modules, config = _build_policy_modules(str(SMOKE_CHECKPOINT))
    policy = HybridPolicy(modules, config)

    env_sim = make_environment(config)
    env_sim.reset(num_pursuers=4, num_evaders=2, scenario_seed=PARITY_SEED)

    # 障碍显式同步，消除对场景采样一致性的依赖
    obstacles = [(float(o.position[0]), float(o.position[1]), float(o.radius))
                 for o in env_sim.obstacles]
    runtime = _runtime(obstacles=obstacles)

    origin = np.zeros(2, dtype=np.float32)
    v_max = float(env_sim.v_max)
    shadow = np.zeros((6, 2), dtype=np.float32)
    shadow_assignment = None
    transition_start = np.zeros((4, 2), dtype=np.float32)
    transition_elapsed = np.full(4, 0.4, dtype=np.float32)
    dt = float(env_sim.time_step)

    for step in range(PARITY_STEPS):
        decision = policy.decision(env_sim)  # eval 模式，无探索噪声

        positions_world, velocities_world = _world_state(env_sim, origin)
        command, info = runtime.act(positions_world, velocities_world, origin)

        # ---- 期望指令：同一推理链输出 + 同一组部署投影（filter + rate limit）
        raw = np.zeros((6, 2), dtype=np.float32)
        raw[:4] = np.clip(decision["actions"][:4], -1.0, 1.0) * v_max
        for j in range(2):
            idx = 4 + j
            if env_sim.task_mask[j]:
                raw[idx] = env_sim._evader_velocity(j)
            elif env_sim.stages[j] == MODE_EXIT:
                raw[idx] = env_sim._exit_velocity(j)
        current_assignment = np.asarray(decision["assignment"])
        if shadow_assignment is None:
            shadow_assignment = current_assignment.copy()
        else:
            changed = current_assignment != shadow_assignment
            transition_start[changed] = shadow[:4][changed]
            transition_elapsed[changed] = 0.0
            shadow_assignment = current_assignment.copy()
        active = transition_elapsed < 0.4
        transition_elapsed[active] = np.minimum(0.4, transition_elapsed[active] + dt)
        alpha = transition_elapsed / 0.4
        raw[:4][active] = (
            (1.0 - alpha[active, None]) * transition_start[active]
            + alpha[active, None] * raw[:4][active]
        )
        expected = filter_velocity(positions_world, raw, origin, _config_4v2())
        expected[:4] = _shadow_vector_limit(
            expected[:4], shadow[:4], max_acceleration=1.5, dt=dt,
        )
        shadow[...] = expected

        np.testing.assert_allclose(command, expected, atol=1e-5)
        assert info["assignment"] == decision["assignment"].tolist()
        assert info["allocated"] == decision["allocated"]

        env_sim.step(np.asarray(decision["actions"]), max_steps=1000)

    # 窗口内不得发生捕获（捕获翻转两边最多差 1 个控制周期，见模块 docstring）
    assert not env_sim.captured[:2].any()


def _import_real_flight(monkeypatch):
    stub = types.ModuleType("crazyflie_py")
    stub.Crazyswarm = type("Crazyswarm", (), {})
    monkeypatch.setitem(sys.modules, "crazyflie_py", stub)
    # mocap_bridge 依赖 ROS2 的 rclpy，测试环境没有：用带 Nokov 的桩替代
    mocap_stub = types.ModuleType("crazyflie_examples.six_v_two_hgat.mocap_bridge")
    mocap_stub.Nokov = type("Nokov", (), {})
    monkeypatch.setitem(sys.modules,
                        "crazyflie_examples.six_v_two_hgat.mocap_bridge", mocap_stub)
    from . import real_flight
    return real_flight


class _RecordingRuntime:
    """只记录构造分支的假 Runtime；--plan-only 路径在连机前退出。"""

    constructed = {}

    def __init__(self, checkpoint, **kwargs):
        type(self).constructed["class"] = type(self).__name__
        type(self).constructed["kwargs"] = kwargs
        self.env = StageCoalitionEnv(length=8.0, num_obstacle=0,
                                     max_pursuers=6, max_evaders=3)
        self.env.reset(num_pursuers=4, num_evaders=2, scenario_seed=1)
        self.num_pursuers = 4
        self.num_evaders = 2
        self.recenter_scene = kwargs.get("recenter_scene", False)

    def reset(self):
        pass


class _RecordingStageRuntime(_RecordingRuntime):
    pass


class _RecordingV3Runtime(_RecordingRuntime):
    pass


@pytest.mark.parametrize("flag,expected", [
    (["--runtime", "v3hybrid"], "_RecordingV3Runtime"),
    (["--runtime", "stage"], "_RecordingStageRuntime"),
    ([], "_RecordingStageRuntime"),  # 默认不变
])
def test_real_flight_runtime_flag_selects_runtime_class(monkeypatch, flag, expected):
    real_flight = _import_real_flight(monkeypatch)
    monkeypatch.setattr(real_flight, "StageCoalitionRuntime", _RecordingStageRuntime)
    monkeypatch.setattr(real_flight, "V3HybridRuntime", _RecordingV3Runtime)
    monkeypatch.setattr(sys, "argv", [
        "real_flight.py", "dummy.pth", "--scale", "4v2", "--plan-only",
        "--roles", "cf1,cf2,cf3,cf4,cf5,cf6", *flag,
    ])
    real_flight.main()
    assert _RecordingRuntime.constructed["class"] == expected
    kwargs = _RecordingRuntime.constructed["kwargs"]
    assert kwargs["scenario_seed"] is None  # 未指定时透传 None（场景默认种子）
    assert "device" in kwargs and "obstacles" in kwargs
