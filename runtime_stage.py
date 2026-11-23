"""Real-flight runtime for the paper method (SDGCR) on variable-scale pursuit.

The training checkpoints cover up to 6 pursuers + 3 evaders, so the underlying
env and encoder stay at that maximum scale while reset()/act() use the active
team size configured by the user.  The runtime:

* feeds mocap state into the stage-feasible allocator and ST-HGAT encoder,
* emits normalized planar velocity commands for the six pursuers,
* drives the two evaders with the same potential-field rule used in training,
* advances the task/safety dual-mask lifecycle (capture -> exit -> landed),
* applies boundary, pairwise separation and command-continuity projections
  before returning velocities for ``cmdVelocityWorld``.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
torch.set_num_threads(4)  # CPU perf: avoid thread oversubscription on tiny nets

from .config import SixVTwoConfig
from .stage_env import StageCoalitionEnv, MODE_NAMES
from .stage_coalitions import StageCoalitionAllocator
from .stage_networks import SharedActor, StageFeatureEncoder
from .safety import filter_velocity
from .training_protocol import DEFAULT_D_CAP, DEFAULT_HOLD_STEPS
from .main_stage_feasible import validate_predictor_free_checkpoint


class StageCoalitionRuntime:
    """Checkpoint -> eight planar velocity set-points for the SDGCR method."""

    def __init__(
        self,
        checkpoint: str,
        config: Optional[SixVTwoConfig] = None,
        device: str = "cpu",
        obstacles: Iterable[Tuple[float, float, float]] = (),
        evader_mode: str = "escape",
        perception_radius: Optional[float] = None,
        lock_on_close3: Optional[bool] = None,
        scenario_seed: Optional[int] = None,
        recenter_scene: bool = False,
    ):
        self.config = config or SixVTwoConfig()
        self.num_pursuers = self.config.num_pursuers
        self.num_evaders = self.config.num_evaders
        self.num_agents = self.num_pursuers + self.num_evaders
        self.device = torch.device(device)
        # partial observability radius used at deployment (None = full obs)
        self.perception_radius = perception_radius
        # 固定场景种子。None = 沿用训练默认（scenario_seed = env._root_seed + episode_index，
        # 每次 reset() 都会换场景）。实机要复现 5x5 场景清单里的某个 seed（逃逸机目标点 +
        # 障碍布局都由此决定），必须显式传入，否则飞的是默认种子采样出来的另一个场景。
        self.scenario_seed = None if scenario_seed is None else int(scenario_seed)
        # 部署用：把采样场景整体平移，使出生点质心落在 arena 中心（见 _recenter_scene）。
        # 仿真口径（评测器 / verify_online_decision / 5x5 场景生成）保持默认 False，逐值不变。
        self.recenter_scene = bool(recenter_scene)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        payload = torch.load(
            Path(checkpoint).expanduser().resolve(), map_location=self.device, weights_only=False
        )
        validate_predictor_free_checkpoint(payload)
        method = payload.get("method")
        # 显式白名单：只接受已知可部署的书写器/版本组合
        _accept = {
            "stage_feasible_dgcr": range(2, 10),
            "sdgcr_end_to_end_probe": range(1, 10),
        }
        if method not in _accept:
            raise ValueError(
                "checkpoint is not an SDGCR checkpoint; "
                "retrain with main_stage_feasible (or use --method legacy)"
            )
        version = int(payload.get("version", 1))
        if version not in _accept[method]:
            raise ValueError(
                f"unsupported checkpoint version {version} for method {method!r}; "
                f"accepted: {sorted(_accept[method])}"
            )
        print(f"[runtime] checkpoint method={method} version={version} "
              f"episode={payload.get('episode')}", flush=True)

        args = payload.get("args", {}) or {}
        self.hidden_dim = int(payload.get("hidden_dim") or args.get("hidden_dim", 32))
        self.future_steps = int(payload.get("future_steps") or args.get("future_steps", 10))
        self.seq_len = int(payload.get("seq_len") or args.get("seq_len", 10))
        self.d_cap = float(payload.get("d_cap") or args.get("d_cap", DEFAULT_D_CAP))
        self.hold_steps = int(
            payload.get("hold_steps") or args.get("hold_steps", DEFAULT_HOLD_STEPS)
        )
        self.evader_mode = evader_mode
        # ------------------------------------------------------------------
        # 合围锁定（lock_on_close3）必须与论文评测协议一致。
        #
        # 论文里所有 SDGCR 结果都是用 `--lock-on-close3` 评出来的：只要有 3 个核心
        # 成员同时进入 d_cap，分配器就冻结该联盟的目标/角色，不再每 5 步重分配，
        # 保证封锁-收拢阶段不被打断。而 StageCoalitionAllocator 的默认值是
        # lock_on_close3=False —— 之前 runtime 用默认值构造分配器，于是**部署路径
        # 静默地少了一个阶段可行性机制**：仿真里 4v2 每回合 1.5~2 倍时长且 0 捕获，
        # 而同一检查点同一场景在评测器（带 lock）下 598 步 all_captured。
        #
        # 解析优先级：显式参数 > 检查点里记录的值 > 部署默认 True（= 论文协议）。
        # ------------------------------------------------------------------
        if lock_on_close3 is None:
            saved_lock = payload.get("lock_on_close3")
            if saved_lock is None:
                saved_lock = args.get("lock_on_close3")
            self.lock_on_close3 = True if saved_lock is None else bool(saved_lock)
            source = "部署默认(论文协议 = 开启)" if saved_lock is None else "检查点记录"
        else:
            self.lock_on_close3 = bool(lock_on_close3)
            source = "显式参数"
        print(f"[runtime] 合围锁定 lock_on_close3={self.lock_on_close3}（来源：{source}）",
              flush=True)
        # The end-to-end probe pipeline always trained with ring-slot features;
        # stage_feasible checkpoints store the flag explicitly or fall back to the
        # training CLI default.
        # ------------------------------------------------------------------
        # 架构开关必须与保存时的 actor 输入维度一致，否则 actor 首层会 size mismatch。
        # 历史检查点（main_stage_end_to_end 写出）的 args 里没有 ring/stage/graph 键，
        # 因此这里以**保存的 actor 首层宽度**为唯一判据，在 (ring,stage,graph) 组合中
        # 寻找匹配者；找不到就显式报错，绝不带着错误架构静默飞。
        # 与 evaluate_stage_baseline.py 的反推逻辑保持一致。
        # ------------------------------------------------------------------
        try:
            saved_obs_dim = int(payload["actor"]["net.0.weight"].shape[1])
        except Exception:
            saved_obs_dim = None

        if method == "sdgcr_end_to_end_probe":
            pref = (True, True, True)
        else:
            ring_arg = bool(args.get("ring_slot_interface", False))
            pref = (ring_arg,
                    bool(args.get("use_stage_embedding", True)),
                    bool(args.get("use_graph_embedding", True)))

        def _ring_params(ring: bool):
            legacy = bool(ring) and version < 8
            return (float(payload.get("ring_slot_normalizer", 2.0 if legacy else 4.0)),
                    float(payload.get("ring_slot_phase_hysteresis", 0.0 if legacy else 0.25)))

        def _probe_obs_dim(ring: bool, stage: bool, graph: bool) -> int:
            norm_, hyst = _ring_params(ring)
            enc = StageFeatureEncoder(
                max_pursuers=6, max_evaders=3,
                num_obstacles=self.config.num_obstacles,
                map_size=self.config.arena_size,
                hidden_dim=self.hidden_dim, future_steps=self.future_steps,
                seq_len=self.seq_len, device=self.device,
                use_ring_slot_features=ring,
                use_stage_embedding=stage, use_graph_embedding=graph,
                ring_slot_normalizer=norm_, ring_slot_phase_hysteresis=hyst,
            )
            return int(enc.obs_dim)

        resolved = pref
        if saved_obs_dim is not None and _probe_obs_dim(*pref) != saved_obs_dim:
            found = None
            for r in (pref[0], not pref[0]):
                for st in (pref[1], not pref[1]):
                    for g in (pref[2], not pref[2]):
                        if (r, st, g) == pref:
                            continue
                        if _probe_obs_dim(r, st, g) == saved_obs_dim:
                            found = (r, st, g)
                            break
                    if found:
                        break
                if found:
                    break
            if found is None:
                raise ValueError(
                    f"cannot reconstruct the encoder for this checkpoint: saved actor "
                    f"expects obs_dim={saved_obs_dim}, and no (ring,stage,graph) toggle "
                    f"combination reproduces it (default {pref} gives "
                    f"obs_dim={_probe_obs_dim(*pref)}). Refusing to fly with a mismatched "
                    f"architecture."
                )
            resolved = found
            print(f"[runtime] saved actor expects obs_dim={saved_obs_dim}; "
                  f"using ring={resolved[0]} stage={resolved[1]} graph={resolved[2]} "
                  f"(default was {pref})", flush=True)

        self.use_ring_slot_features = bool(payload.get("use_ring_slot_features", resolved[0]))
        self.use_stage_embedding = bool(resolved[1])
        self.use_graph_embedding = bool(resolved[2])
        self.ring_slot_normalizer, self.ring_slot_phase_hysteresis = _ring_params(resolved[0])
        self.saved_obs_dim = saved_obs_dim

        # The checkpoint was trained at up to 6 pursuers / 3 evaders, so keep the
        # underlying buffers at that maximum scale and only reset to the active
        # team size configured by the user.
        self.env = StageCoalitionEnv(
            length=self.config.arena_size,
            num_obstacle=self.config.num_obstacles,
            max_pursuers=6,
            max_evaders=3,
            d_cap=self.d_cap,
            hold_steps=self.hold_steps,
            perception_radius=self.perception_radius,
        )
        self.env.reset(num_pursuers=self.num_pursuers, num_evaders=self.num_evaders)
        self._set_obstacles(obstacles)

        self.encoder = StageFeatureEncoder(
            max_pursuers=6,
            max_evaders=3,
            num_obstacles=self.config.num_obstacles,
            map_size=self.config.arena_size,
            hidden_dim=self.hidden_dim,
            future_steps=self.future_steps,
            seq_len=self.seq_len,
            device=self.device,
            perception_radius=self.perception_radius,
            use_ring_slot_features=self.use_ring_slot_features,
            use_stage_embedding=self.use_stage_embedding,
            use_graph_embedding=self.use_graph_embedding,
            ring_slot_normalizer=self.ring_slot_normalizer,
            ring_slot_phase_hysteresis=self.ring_slot_phase_hysteresis,
        ).to(self.device)
        if self.saved_obs_dim is not None and int(self.encoder.obs_dim) != int(self.saved_obs_dim):
            raise ValueError(
                f"encoder obs_dim={self.encoder.obs_dim} does not match the saved actor "
                f"obs_dim={self.saved_obs_dim}; refusing to fly"
            )
        self.actor = SharedActor(self.encoder.obs_dim, hidden_dim=64).to(self.device)
        self.encoder.st_hgat.load_state_dict(payload["st_hgat"])
        self.encoder.relation_head.load_state_dict(payload["relation_head"])
        self.actor.load_state_dict(payload["actor"])
        self.encoder.eval()
        self.actor.eval()

        # 与评测器 load_sdgcr 保持同一组参数：use_stage_feasibility / coalition_mode
        # 用默认值（True / "dynamic"，即主线消融档），显式传 lock_on_close3 与 device。
        self.allocator = StageCoalitionAllocator(
            d_cap=self.d_cap,
            device=str(self.device),
            use_stage_feasibility=True,
            coalition_mode="dynamic",
            lock_on_close3=self.lock_on_close3,
        )
        self.assign_interval = 5  # low-frequency reconfiguration (paper protocol)
        self.command_velocity = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._step = 0
        self._prev_pred = None
        self._prev_unc = None
        self._prev_rel = None
        self._alloc = None
        # 生命周期是否由外部（仿真 env.step）推进；真机部署保持 False
        self.external_lifecycle = False
        self.reset()

    def _set_obstacles(self, obstacles: Iterable[Tuple[float, float, float]]) -> None:
        values = list(obstacles)
        if len(values) > self.config.num_obstacles:
            raise ValueError("too many obstacles for the trained graph")
        self._fixed_obstacles = values  # keep so reset() re-applies them
        self._apply_obstacles(values)

    def _apply_obstacles(self, values: List[Tuple[float, float, float]]) -> None:
        self.env.obstacles = []
        for x, y, radius in values:
            self.env.obstacles.append(
                type("O", (), {
                    "position": np.array([x, y], dtype=np.float64),
                    "radius": float(radius),
                    "height": 3.0,  # cylinder covers the flight band (obs_height)
                })()
            )

    def _recenter_scene(self) -> None:
        """把采样场景整体平移，使其出生点质心落在 arena 中心。

        部署映射是 ``local = world - mean(world) + arena/2``（见 :meth:`act`），也就是把
        **现场 6 架的实际质心**当成 arena 中心；而场景采样出来的出生点质心 ``C`` 一般
        不等于 arena 中心。若不做这步，现场按「相对出生点质心」的坐标摆放时，策略内部
        看到的机体相对障碍/目标点会整体偏 ``arena/2 - C``（4v2 seed=9 实测 0.64 m）。
        平移机体 + 障碍 + 目标点（刚性整体平移）后：
          * 现场可直接用文档里「相对出生点质心」的坐标；
          * 策略内部「机体 vs 障碍/目标点」的相对几何与仿真完全一致。
        """
        env = self.env
        n = self.num_agents
        positions = np.array(env.positions[:n, :2], dtype=np.float64)
        if positions.size == 0:
            return
        shift = env.length / 2.0 - positions.mean(axis=0)
        if not np.any(np.abs(shift) > 1e-12):
            return
        env.positions[:n, :2] = positions + shift
        for obstacle in env.obstacles:
            obstacle.position = (np.asarray(obstacle.position, dtype=np.float64).copy()
                                 + shift)
        goals = np.array(env.evader_goals, dtype=np.float64, copy=True)
        goals[:, :2] += shift
        env.evader_goals = goals
        self._scene_recenter_shift = shift

    def reset(self) -> None:
        self.encoder.reset(step=0)
        self.allocator.reset()
        self.command_velocity.fill(0.0)
        # 固定场景种子时，每次 reset 都回到同一个场景（逃逸机目标点 / 障碍布局不变）；
        # 未指定时沿用训练默认（root_seed + episode_index，每次 reset 换场景）。
        seed = getattr(self, "scenario_seed", None)
        if seed is None:
            self.env.reset(num_pursuers=self.num_pursuers, num_evaders=self.num_evaders)
        else:
            self.env.reset(num_pursuers=self.num_pursuers, num_evaders=self.num_evaders,
                           scenario_seed=seed)
        # 部署时把场景整体居中（必须在 _apply_obstacles 之前：显式障碍文件优先级更高）
        if getattr(self, "recenter_scene", False):
            self._recenter_scene()
        # env.reset() regenerates random obstacles; re-apply the user/virtual set
        if getattr(self, "_fixed_obstacles", None):
            self._apply_obstacles(self._fixed_obstacles)
        self._step = 0
        self._prev_pred = None
        self._prev_unc = None
        self._prev_rel = None
        self._alloc = None

    # ------------------------------------------------------------------ act
    def act(
        self,
        positions_world: np.ndarray,
        velocities_world: np.ndarray,
        arena_origin_world: Sequence[float] = (0.0, 0.0),
    ) -> Tuple[np.ndarray, dict]:
        positions = np.asarray(positions_world, dtype=np.float32)
        velocities = np.asarray(velocities_world, dtype=np.float32)
        valid_shapes = {(self.num_agents, 2), (self.num_agents, 3)}
        if positions.shape not in valid_shapes or velocities.shape not in valid_shapes:
            raise ValueError(
                f"positions and velocities must both have shape "
                f"({self.num_agents}, 2) or ({self.num_agents}, 3)"
            )
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("non-finite mocap state")

        origin = np.asarray(arena_origin_world, dtype=np.float32)
        # 3-D support: xy in the arena frame, z from the mocap (or flight height)
        local = np.zeros((self.num_agents, 3), dtype=np.float32)
        local[:, :2] = positions[:, :2] - origin[:2] + self.config.arena_size / 2.0
        if positions.shape[1] == 3:
            local[:, 2] = positions[:, 2]
        else:
            local[:, 2] = self.env.flight_height
        self.env.positions[:self.num_agents] = local
        self.env.velocities[:self.num_agents, :2] = velocities[:, :2]
        if velocities.shape[1] == 3:
            self.env.velocities[:self.num_agents, 2] = velocities[:, 2]
        else:
            self.env.velocities[:self.num_agents, 2] = 0.0
        self.env.pursuer_mask[:self.num_pursuers] = True
        # NOTE: task_mask/safe_mask must NOT be forced True here -- they encode
        # the capture->EXIT->LANDED lifecycle (task 1->0 on capture, safe 1->0 on
        # landing).  Forcing them True every control cycle would re-activate
        # captured targets and break sequential pursuit.  Only reset() initialises
        # them; advance_lifecycle() below advances them on mocap geometry.

        # 1) lifecycle on mocap geometry (uses the previous assignment)
        #    真机：生命周期只由这里推进。仿真闭环验证里 env.step 内部也会推进一次，
        #    那样每个控制周期会推进两遍，故提供 external_lifecycle 开关（默认 False
        #    = 保持真机语义）。
        if not self.external_lifecycle:
            self.env.advance_lifecycle()
        snap = self.env.snapshot()

        # 2) stage-feasible coalition allocation (throttled)
        prev_pred = getattr(self, "_prev_pred", None)
        prev_unc = getattr(self, "_prev_unc", None)
        prev_rel = getattr(self, "_prev_rel", None)
        if self._alloc is None or self._step % self.assign_interval == 0:
            self._alloc = self.allocator.allocate(
                self.env.pursuer_positions(), self.env.pursuer_velocities(),
                self.env.evader_positions(), self.env.evader_velocities(),
                prev_pred, prev_unc,
                [(o.position, o.radius) for o in self.env.obstacles],
                snap["task_mask"][:self.num_evaders],
                np.ones(self.num_pursuers, dtype=bool),
                prev_rel, self._step,
            )
        alloc = self._alloc
        self.env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
                             alloc["roles"], alloc["stages"])

        # 3) features + shared actor
        snap = self.env.snapshot()
        with torch.no_grad():
            features = self.encoder.encode(snap)
        self._prev_pred = features["predictions"][:self.num_evaders]
        self._prev_unc = features["uncertainty"][:self.num_evaders]
        self._prev_rel = features["relation_cost"]
        with torch.no_grad():
            actions = self.actor(features["obs"]).cpu().numpy()

        command = np.zeros((self.num_agents, 2), dtype=np.float32)
        command[:self.num_pursuers] = (
            np.clip(actions[:self.num_pursuers], -1.0, 1.0)
            * self.config.max_pursuer_speed
        )
        for j in range(self.num_evaders):
            idx = self.num_pursuers + j
            if self.env.task_mask[j]:
                command[idx] = self._evader_command(j)
            elif self.env.stages[j] == 3:  # MODE_EXIT
                command[idx] = self.env._exit_velocity(j)
            else:
                command[idx] = 0.0

        # 4) safety + continuity projection
        command = filter_velocity(positions, command, origin, self.config)
        command = self._rate_limit(command, self.config.max_pursuer_speed)
        self.command_velocity = command.copy()
        self._step += 1

        info = {
            "core_assignment": alloc["core_assignment"].tolist(),
            "sup_assignment": alloc["sup_assignment"].tolist(),
            "roles": [["core", "support", "reserve"][int(r)] if 0 <= int(r) <= 2 else "?" for r in alloc["roles"]],
            "stages": [MODE_NAMES.get(int(s), "?") for s in snap["stages"][:self.num_evaders]],
            "task_masks": snap["task_mask"][:self.num_evaders].tolist(),
            "safe_masks": snap["safe_mask"][:self.num_evaders].tolist(),
            "captured": snap["captured"][:self.num_evaders].tolist(),
            "target_switch": alloc["target_switch"],
            "role_switch": alloc["role_switch"],
            "reserve_count": alloc["idle"],
            "soft": None,
        }
        return command, info

    def _evader_command(self, j: int) -> np.ndarray:
        if self.evader_mode == "stationary":
            return np.zeros(2, dtype=np.float32)
        # env._evader_velocity is already in m/s (bounded by v_max_e); the radio
        # command is m/s too, so pass it through unmodified (do NOT divide by
        # max_evader_speed, which would rescale 0.35 m/s up to ~1.0)
        return self.env._evader_velocity(j).astype(np.float32)

    def _rate_limit(self, command: np.ndarray, max_speed: float) -> np.ndarray:
        # limit per-axis step change to a fraction of max speed (continuity)
        max_delta = 0.08 * max_speed
        delta = np.clip(command - self.command_velocity, -max_delta, max_delta)
        return self.command_velocity + delta
