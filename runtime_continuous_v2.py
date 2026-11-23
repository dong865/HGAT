"""continuous_membership_v2 runtime：录制快照/仿真回放入口。

与 runtime_stage.py 相同的"checkpoint → 速度指令"角色，但走 V2 统一
controller_step；不连接实机（阶段 C 验收只用录制快照或仿真）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .continuous_membership import ContinuousDemandHead
from .stage_env import StageCoalitionEnv
from .stage_networks import SharedActor, StageFeatureEncoder
from .v2_controller import (
    ALLOCATION_SOURCE,
    ARCHITECTURE_VERSION,
    V2Controller,
    v2_obs_dim,
)

REQUIRED_SCHEMA_KEYS = (
    "architecture_version",
    "observation_schema_version",
    "task_plan_schema_version",
    "membership_schema_version",
    "capture_definition_version",
    "context_pool",
    "target_context",
    "demand_parameterization",
)


class V2Runtime:
    """V2 仿真/回放运行时。加载 V2 checkpoint schema，驱动 controller_step。"""

    def __init__(self, checkpoint: str, device: str = "cpu",
                 scenario_seed: Optional[int] = None, **controller_kwargs):
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint).expanduser().resolve(),
                             map_location=self.device, weights_only=False)
        self._check_schema(payload)
        args = payload.get("args", {}) or {}

        self.hidden_dim = int(payload.get("hidden_dim", 32))
        self.future_steps = int(payload.get("future_steps", 10))
        self.seq_len = int(payload.get("seq_len", 10))
        self.d_cap = float(payload.get("d_cap", 1.5))
        self.hold_steps = int(payload.get("hold_steps", 5))
        self.max_pursuers = int(payload.get("max_pursuers", 6))
        self.max_evaders = int(payload.get("max_evaders", 3))

        self.env = StageCoalitionEnv(
            length=float(payload.get("map_size", 8.0)),
            num_obstacle=int(payload.get("num_obstacles", 10)),
            max_pursuers=self.max_pursuers, max_evaders=self.max_evaders,
            d_cap=self.d_cap, hold_steps=self.hold_steps,
        )
        self.env.reset(num_pursuers=int(payload.get("num_pursuers", 6)),
                       num_evaders=int(payload.get("num_evaders", 2)),
                       scenario_seed=scenario_seed)

        self.encoder = StageFeatureEncoder(
            max_pursuers=self.max_pursuers, max_evaders=self.max_evaders,
            num_obstacles=self.env.num_obstacle, map_size=self.env.length,
            hidden_dim=self.hidden_dim, future_steps=self.future_steps,
            seq_len=self.seq_len, node_dim=9, device=self.device,
            use_ring_slot_features=False, use_stage_embedding=False,
            use_role_embedding=False, use_graph_embedding=True,
        ).to(self.device)
        self.demand_head = ContinuousDemandHead(
            self.hidden_dim,
            demand_init=float(args.get("demand_init", 3.0)),
        ).to(self.device)
        self.actor = SharedActor(
            v2_obs_dim(self.max_pursuers, self.hidden_dim, self.future_steps),
            hidden_dim=64,
        ).to(self.device)

        loaded, missing = [], []
        for name, module in (("st_hgat", self.encoder.st_hgat),
                             ("relation_head", self.encoder.relation_head),
                             ("demand_head", self.demand_head),
                             ("actor", self.actor)):
            state = payload.get(name)
            if state is None:
                missing.append(name)
                continue
            module.load_state_dict(state)
            loaded.append(name)
            module.eval()
        self.migration_report = {"loaded": loaded, "missing": missing,
                                 "reinitialized": ["actor"] if "actor" in missing else []}
        if "actor" in missing:
            raise ValueError("checkpoint 缺少 actor，拒绝作为可用控制器运行")

        self.controller = V2Controller(
            self.encoder, self.demand_head, self.actor, self.device,
            max_pursuers=self.max_pursuers, max_evaders=self.max_evaders,
            demand_source=str(args.get("demand_source", "learned_continuous")),
            **{k: v for k, v in controller_kwargs.items()
               if k in ("epsilon_ot", "sinkhorn_iters", "count_penalty",
                        "switch_penalty", "hysteresis", "min_dwell_steps",
                        "assign_interval", "fixed_demand")},
        )
        self.controller.reset()

    @staticmethod
    def _check_schema(payload: Dict):
        arch = payload.get("architecture_version")
        if arch != ARCHITECTURE_VERSION:
            raise ValueError(
                f"checkpoint architecture_version={arch!r}，本运行时只接受 "
                f"{ARCHITECTURE_VERSION!r}；旧 checkpoint 请走 legacy 入口")
        for key in REQUIRED_SCHEMA_KEYS:
            if key not in payload:
                raise ValueError(f"V2 checkpoint 缺少 schema 字段: {key}")
        if payload.get("allocation_source") not in (None, ALLOCATION_SOURCE):
            raise ValueError("allocation_source 与 V2 主线不符")

    # ------------------------------------------------------------------
    def reset(self, scenario_seed: Optional[int] = None):
        if scenario_seed is not None:
            self.env.reset(num_pursuers=self.env.num_pursuers,
                           num_evaders=self.env.num_evaders,
                           scenario_seed=int(scenario_seed))
        self.controller.reset()

    def step(self, positions_world: np.ndarray,
             velocities_world: np.ndarray,
             origin=(0.0, 0.0)
             ) -> Tuple[np.ndarray, Dict]:
        """与 runtime_stage.act 同签名：写入动捕状态 → controller_step → 指令。"""
        positions = np.asarray(positions_world, dtype=np.float32)
        velocities = np.asarray(velocities_world, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        n_agents = self.env.num_pursuers + self.env.num_evaders
        if positions.shape not in {(n_agents, 2), (n_agents, 3)}:
            raise ValueError("positions/velocities 形状不符")

        local = np.zeros((n_agents, 3), dtype=np.float32)
        local[:, :2] = positions[:, :2] - origin[:2] + self.env.length / 2.0
        local[:, 2] = positions[:, 2] if positions.shape[1] == 3 else self.env.flight_height
        self.env.positions[:n_agents] = local
        self.env.velocities[:n_agents, :2] = velocities[:, :2]
        self.env.pursuer_mask[:self.env.num_pursuers] = True
        # 真机语义：每个 50 Hz 动捕帧 = 一个新的物理时刻。
        # 注意：legacy runtime_stage 未递增该计数器，其编码器时间状态实为首步后冻结
        # （已记录为基线发现，不在本次静默修改）；V2 必须递增。
        self.env.episode_step_count += 1

        actions, info = self.controller.controller_step(self.env)
        command = np.zeros((n_agents, 2), dtype=np.float32)
        command[:self.env.num_pursuers] = (
            np.clip(actions, -1.0, 1.0) * self.config_max_speed()
        )
        return command, info

    def config_max_speed(self) -> float:
        return float(getattr(self.env, "v_max", 0.35))
