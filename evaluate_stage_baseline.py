"""论文主线 SDGCR 的正式评测（StageCoalitionEnv）。

精简后本文件只支持 ``--method sdgcr``；基线方法（grouped/independent/no-graph MADDPG、
MAPPO、IBR、MADDPG、G2ANet-AC、CBBA-IBR）及其训练/评测代码已从本包移除。

用法（多 seed，无 GUI）：
    python -m crazyflie_examples.six_v_two_hgat.evaluate_stage_baseline \
        --method sdgcr \
        --scale 4v2 \
        --checkpoint checkpoints/six_v_two_4v2/checkpoint_002000.pth \
        --seeds 17 101 211 307 401 \
        --episodes-per-seed 20 \
        --max-steps 1000 \
        --lock-on-close3 \
        --no-gui \
        --output evaluations/4v2_deploy/sdgcr.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(4)  # CPU perf: avoid thread oversubscription on tiny nets

from .training_protocol import DEFAULT_D_CAP, DEFAULT_HOLD_STEPS
try:  # 口径自审计：每个结果 JSON 必须能事后分辨场景协议版本
    from .scenario_protocol import PROTOCOL_VERSION as _SCENARIO_PROTOCOL_VERSION
except Exception:  # pragma: no cover
    _SCENARIO_PROTOCOL_VERSION = "unknown"
from .ablation_config import canonical_ablation_names, resolve_ablation


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["sdgcr"])
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint dir (MADDPG) or .pth file (MAPPO/SDGCR); ignored for IBR")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17],
                        help="test scenario seeds")
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--raw-seed-sequence", action="store_true",
        help="interpret each --seeds value as the first literal scenario_seed and "
             "use seed+episode_idx; this reproduces main_stage_end_to_end's "
             "training scenario stream. Default keeps historical derived test seeds.",
    )
    parser.add_argument(
        "--repeat-same-scenario", action="store_true",
        help="每个 episode 重放 --seeds 给出的同一字面场景种子",
    )
    parser.add_argument("--scale", default="6v2",
                        choices=("4v2", "5v2", "6v2", "6v3"),
                        help="team scale to evaluate (default 6v2). The SDGCR encoder "
                             "was built for max_pursuers=6/max_evaders=3, so 4v2/5v2/6v3 "
                             "exercise the variable-cardinality interface zero-shot.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--d-cap", type=float, default=DEFAULT_D_CAP)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument(
        "--ablation", choices=("none",) + canonical_ablation_names(), default="none",
        help="one canonical SDGCR ablation; ignored by other methods",
    )
    parser.add_argument("--allow-protocol-mismatch", action="store_true",
                        help="diagnostic only: allow checkpoint/evaluation protocol mismatch")
    parser.add_argument("--lock-on-close3", action="store_true",
                        help="allocator: lock nearest-3 core while >=3 pursuers within d_cap")
    parser.add_argument(
        "--allow-legacy-predictor", action="store_true",
        help=("evaluate a legacy (version<9) checkpoint that carries an unused "
              "'predictor' key. The key-presence guard is replaced by the stronger "
              "invariant that the Actor must load strictly against the encoder "
              "rebuilt from its saved input width. Default off = unchanged behaviour. "
              "See logs/LEGACY_CHECKPOINT_UNBLOCK.md"))
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="MAPPO Critic/Actor hidden dim (only used if not in checkpoint)")
    parser.add_argument("--no-gui", action="store_true",
                        help="force headless (default: headless)")
    parser.add_argument("--output", default=None,
                        help="optional JSON file to write summary metrics")
    parser.add_argument(
        "--team-change", choices=("control", "drop1", "drop2", "join1", "hold1"),
        default="control", help="mid-episode pursuer membership intervention",
    )
    parser.add_argument("--team-change-step", type=int, default=150)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------
def make_env(args) -> Any:
    from .stage_env import StageCoalitionEnv
    return StageCoalitionEnv(
        length=args.map_size,
        num_obstacle=args.num_obstacles,
        max_pursuers=6,
        max_evaders=3,
        d_cap=args.d_cap,
        hold_steps=args.hold_steps,
    )


# ---------------------------------------------------------------------------
# Method loaders / actors
# ---------------------------------------------------------------------------
def load_sdgcr(
    checkpoint_path: str,
    device: torch.device,
    ablation: str = "none",
    d_cap: float = DEFAULT_D_CAP,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    allow_protocol_mismatch: bool = False,
    lock_on_close3: bool = False,
    map_size: float = 8.0,
    num_obstacles: int = 10,
    allow_legacy_predictor: bool = False,
):
    spec = resolve_ablation(ablation)
    from .main_stage_feasible import (
        load_checkpoint, validate_checkpoint_ablation, validate_predictor_free_checkpoint,
    )
    from .stage_coalitions import StageCoalitionAllocator
    from .stage_networks import SharedActor, StageFeatureEncoder

    payload = load_checkpoint(checkpoint_path, device)
    # ------------------------------------------------------------------
    # `validate_predictor_free_checkpoint` rejects any payload carrying a
    # "predictor" key.  That tests the *presence of a key*, not whether the
    # Actor actually consumes a learned prediction.  For the legacy (version<9)
    # ablation checkpoints the predictor was an auxiliary head only: the Actor's
    # input is the DETERMINISTIC constant-velocity lookahead
    # (`2*future_steps`, see stage_networks.py:435), so its width is fully
    # accounted for by the encoder and the predictor weights are unused at
    # evaluation time.
    #
    # With --allow-legacy-predictor we therefore replace the key-presence guard
    # with the actual invariant: the Actor must load with strict=True against
    # the encoder reconstructed from the SAVED actor input width (below).  If
    # the predictor were an Actor input, that width would not match and
    # `actor.load_state_dict` would raise a size mismatch -- so the check is
    # strictly stronger than the key test, not weaker.
    #
    # Default stays False: existing results are reproducible byte-for-byte.
    # ------------------------------------------------------------------
    predictor_present = "predictor" in payload
    if not allow_legacy_predictor:
        validate_predictor_free_checkpoint(payload)
    elif predictor_present:
        print(
            "[load_sdgcr] --allow-legacy-predictor：检查点含 'predictor' 键"
            f"（version={payload.get('version')}），但不再据此拒绝；"
            "改为强制校验 actor 输入宽度与重建编码器一致。",
            file=sys.stderr,
        )
    validate_checkpoint_ablation(payload, spec, allow_mismatch=allow_protocol_mismatch)
    hidden_dim = int(payload.get("hidden_dim", 32))
    future_steps = int(payload.get("future_steps", 10))
    seq_len = int(payload.get("seq_len", 10))
    d_cap = float(d_cap)
    hold_steps = int(hold_steps)
    use_ring_slots = bool(payload.get(
        "use_ring_slot_features",
        payload.get("args", {}).get("ring_slot_interface", False),
    ))
    use_stage_embedding = bool(payload.get(
        "use_stage_embedding",
        payload.get("args", {}).get("supplementary_ablation", "").find("no_stage_embedding") < 0,
    ))
    use_graph_embedding = bool(payload.get(
        "use_graph_embedding",
        payload.get("args", {}).get("supplementary_ablation", "").find("no_graph_embedding") < 0,
    ))
    legacy_ring = use_ring_slots and int(payload.get("version", 0)) < 8
    ring_normalizer = float(payload.get("ring_slot_normalizer", 2.0 if legacy_ring else 4.0))
    ring_hysteresis = float(payload.get("ring_slot_phase_hysteresis", 0.0 if legacy_ring else 0.25))
    protocol_expected = {
        "d_cap": d_cap,
        "hold_steps": hold_steps,
        "map_size": float(map_size),
        "num_obstacles": int(num_obstacles),
    }
    if not allow_protocol_mismatch:
        for key, expected in protocol_expected.items():
            if key in payload and payload[key] != expected:
                raise ValueError(
                    f"checkpoint protocol mismatch for {key}: "
                    f"saved={payload[key]!r}, requested={expected!r}",
                )

    # ------------------------------------------------------------------
    # 编码器开关必须与保存时的 obs_dim 一致，否则 actor 首层直接 size mismatch。
    # 历史主线检查点（resume_20260922/*、sdgcr_compare_formal400/*、snapshot_*）
    # 由 main_stage_end_to_end 写出，其 args 里**没有** ring_slot_interface 键，
    # 于是上面的推断得到 use_ring_slots=False，构造出的 obs_dim 与保存的 162 不符
    # → `size mismatch for SharedActor.net`，这些检查点无法被官方评测器加载。
    # 当时的历史脚本靠手工命令行标志（pred=/nograph/...）绕过，评测器没有该回退。
    #
    # 这里改为：以**保存的 actor 输入维度**为准，在 (ring, stage, graph) 组合中
    # 寻找唯一匹配者；找不到则**显式报错**，绝不带着错误架构静默出数。
    # ------------------------------------------------------------------
    try:
        saved_obs_dim = int(payload["actor"]["net.0.weight"].shape[1])
    except Exception:
        saved_obs_dim = None

    def _ring_params(ring: bool):
        """legacy(version<8) 的环位用不同归一化/迟滞，必须随 ring 开关一起重算。"""
        legacy = bool(ring) and int(payload.get("version", 0)) < 8
        norm_ = float(payload.get("ring_slot_normalizer", 2.0 if legacy else 4.0))
        hyst = float(payload.get("ring_slot_phase_hysteresis", 0.0 if legacy else 0.25))
        return norm_, hyst

    def _make_encoder(ring: bool, stage: bool, graph: bool):
        norm_, hyst = _ring_params(ring)
        return StageFeatureEncoder(
            max_pursuers=6, max_evaders=3, num_obstacles=int(num_obstacles),
            map_size=float(map_size), hidden_dim=hidden_dim,
            future_steps=future_steps, seq_len=seq_len, device=device,
            node_encoder=spec.node_encoder,
            coalition_pool=spec.coalition_pool,
            use_ring_slot_features=ring,
            use_stage_embedding=stage,
            use_graph_embedding=graph,
            ring_slot_normalizer=norm_,
            ring_slot_phase_hysteresis=hyst,
        ).to(device)

    encoder = _make_encoder(use_ring_slots, use_stage_embedding, use_graph_embedding)
    inferred_obs_dim = encoder.obs_dim
    if saved_obs_dim is not None and encoder.obs_dim != saved_obs_dim:
        match = None
        for _r in (use_ring_slots, not use_ring_slots):
            for _s in (use_stage_embedding, not use_stage_embedding):
                for _g in (use_graph_embedding, not use_graph_embedding):
                    if (_r, _s, _g) == (use_ring_slots, use_stage_embedding,
                                        use_graph_embedding):
                        continue
                    cand = _make_encoder(_r, _s, _g)
                    if cand.obs_dim == saved_obs_dim:
                        match = (_r, _s, _g, cand)
                        break
                if match:
                    break
            if match:
                break
        if match is None:
            raise ValueError(
                "cannot reconstruct the encoder for this checkpoint: saved actor "
                f"expects obs_dim={saved_obs_dim}, but no (ring,stage,graph) toggle "
                f"combination reproduces it (inferred combo gives "
                f"obs_dim={inferred_obs_dim}). Refusing to evaluate with a mismatched "
                "architecture — record the missing flags in the checkpoint instead.",
            )
        _r, _s, _g, encoder = match
        print(
            f"[load_sdgcr] 保存的 actor 期望 obs_dim={saved_obs_dim}，与按 args 推断的 "
            f"{inferred_obs_dim} 不符；已改用 ring={_r} stage={_s} graph={_g} "
            f"（原推断 ring={use_ring_slots} stage={use_stage_embedding} "
            f"graph={use_graph_embedding}）",
            file=sys.stderr,
        )
        use_ring_slots, use_stage_embedding, use_graph_embedding = _r, _s, _g
    actor = SharedActor(encoder.obs_dim, hidden_dim=64).to(device)
    encoder.st_hgat.load_state_dict(payload["st_hgat"])
    encoder.relation_head.load_state_dict(payload["relation_head"])
    if encoder.fixed_concat_projector is not None:
        encoder.fixed_concat_projector.load_state_dict(payload["fixed_concat_projector"])
    # Explicit strict load: this is the invariant that makes it safe to ignore a
    # present-but-unused `predictor`.  A size mismatch here means the Actor DOES
    # consume something the encoder does not produce, and we must refuse.
    try:
        actor.load_state_dict(payload["actor"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "actor could not be loaded strictly against the reconstructed encoder "
            f"(obs_dim={encoder.obs_dim}, saved={saved_obs_dim}); the checkpoint's "
            "Actor consumes an input this pipeline does not build, so the predictor "
            "IS in the Actor's input path and this checkpoint cannot be evaluated "
            f"predictor-free. Original error: {exc}"
        ) from exc
    encoder.eval()
    actor.eval()
    allocator = StageCoalitionAllocator(
        d_cap=d_cap, device=str(device),
        use_stage_feasibility=spec.use_stage_feasibility,
        coalition_mode=spec.coalition_mode,
        lock_on_close3=bool(lock_on_close3),
    )
    return {
        "encoder": encoder,
        "actor": actor,
        "allocator": allocator,
        "d_cap": d_cap,
        "hold_steps": hold_steps,
        "ablation_spec": spec,
        # provenance for the output JSON: was a legacy predictor key present, and
        # did the Actor depend on it (answer is provably "no" when we get here)
        "legacy_predictor_present": bool(predictor_present),
        "predictor_used_by_actor": False,
        "loaded_obs_dim": int(encoder.obs_dim),
        "loaded_ring_slots": bool(use_ring_slots),
        "loaded_stage_embedding": bool(use_stage_embedding),
        "loaded_graph_embedding": bool(use_graph_embedding),
    }


def select_action_sdgcr(env, model, device: torch.device, step: int,
                        prev_pred, prev_unc, prev_rel, alloc):
    snap = env.snapshot()
    if alloc is None or step % 5 == 0:
        alloc = model["allocator"].allocate(
            env.pursuer_positions(), env.pursuer_velocities(),
            env.evader_positions(), env.evader_velocities(),
            prev_pred, prev_unc,
            [(o.position, o.radius) for o in env.obstacles],
            snap["task_mask"][: env.num_evaders],
            env.pursuer_mask[: env.num_pursuers],
            prev_rel, step,
        )
    env.apply_roles(
        alloc["core_assignment"], alloc["sup_assignment"],
        alloc["roles"], alloc["stages"],
    )
    snap2 = env.snapshot()
    feats = model["encoder"].encode(snap2)
    with torch.no_grad():
        actions = model["actor"](feats["obs"]).cpu().numpy()
    new_prev = (
        feats["predictions"][: env.num_evaders],
        feats["uncertainty"][: env.num_evaders],
        feats["relation_cost"],
    )
    return actions, new_prev, alloc


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
def resolve_episode_scenario_seed(
    seed: int, episode_idx: int, scale, raw_sequence: bool = False,
) -> int:
    """Resolve a scenario seed with explicit, auditable stream semantics."""
    if raw_sequence:
        return int(seed) + int(episode_idx)
    from .scenario_protocol import derive_scenario_seed
    return derive_scenario_seed(seed, "test", episode_idx, scale)


def run_episode(env, method: str, model, device: torch.device,
                seed: int, episode_idx: int, max_steps: int,
                gui: bool, renderer=None, team_change_mode: str = "control",
                team_change_step: int = 150,
                num_pursuers: int = 6, num_evaders: int = 2,
                raw_seed_sequence: bool = False) -> Dict[str, Any]:
    from .stage_coalitions import StageCoalitionAllocator
    from .stage_env import ROLE_CORE, mode_capacity
    from .team_size_change import TeamSizeChange

    # num_pursuers/num_evaders default to the historical 6v2 so every existing
    # caller is unaffected; passing them enables the 4-scale protocol
    # (4v2 / 5v2 / 6v2 / 6v3). The scale also feeds seed derivation, so the same
    # (seed, episode) pair yields a different scenario per scale by design.
    scale = (int(num_pursuers), int(num_evaders))
    scenario_seed = resolve_episode_scenario_seed(
        seed, episode_idx, scale, raw_sequence=raw_seed_sequence,
    )
    env.reset(num_pursuers=scale[0], num_evaders=scale[1],
              scenario_seed=scenario_seed)
    team_change = TeamSizeChange(team_change_mode, scenario_seed, team_change_step)
    team_change.prepare(env)
    if gui and renderer is not None:
        renderer.load_scene(env)

    # reset method-specific state
    model["encoder"].reset()
    model["allocator"].reset()

    # size from the requested scale, not a hardcoded 6+2, so 4v2/5v2/6v3 work
    trajectories: List[List[np.ndarray]] = [[] for _ in range(scale[0] + scale[1])]
    total_reward = 0.0
    step = 0
    done = False
    strict_any = False
    event_applied = False
    capacity_recovery = None
    strict_recovery = None
    post_invalid = []
    target_switches = 0
    role_switches = 0
    prev_assignment = env.assignment[:env.num_pursuers].copy()
    prev_roles = env.roles[:env.num_pursuers].copy()
    demand_probe = StageCoalitionAllocator(d_cap=env.d_cap, device="cpu")

    sdgcr_state = {"prev_pred": None, "prev_unc": None, "prev_rel": None, "alloc": None}

    while not done and step < max_steps:
        if not event_applied and step >= team_change_step:
            if team_change.apply_if_due(env, step):
                event_applied = True
                sdgcr_state["alloc"] = None
        for i in range(scale[0] + scale[1]):
            trajectories[i].append(env.multi_current_pos[i].copy())

        if method == "sdgcr":
            actions, new_prev, sdgcr_state["alloc"] = select_action_sdgcr(
                env, model, device, step,
                sdgcr_state["prev_pred"], sdgcr_state["prev_unc"],
                sdgcr_state["prev_rel"], sdgcr_state["alloc"],
            )
            sdgcr_state["prev_pred"], sdgcr_state["prev_unc"], sdgcr_state["prev_rel"] = new_prev
        else:
            raise ValueError(f"unknown method {method}")

        active_mask = env.pursuer_mask[: env.num_pursuers]
        actions = np.asarray(actions).copy()
        actions[: env.num_pursuers][~active_mask] = 0.0

        reward, done, info = env.step(actions, max_steps=max_steps)
        total_reward += reward
        step += 1

        strict_now = any(
            env.task_mask[j] and not env.captured[j] and env._capture_hold_ok(j)
            for j in range(env.num_evaders)
        )
        strict_any = strict_any or strict_now
        if event_applied and strict_recovery is None and strict_now:
            strict_recovery = step - team_change_step

        if step % 5 == 0:
            current_assignment = env.assignment[:env.num_pursuers].copy()
            current_roles = env.roles[:env.num_pursuers].copy()
            target_switches += int(np.count_nonzero(current_assignment != prev_assignment))
            role_switches += int(np.count_nonzero(current_roles != prev_roles))
            prev_assignment, prev_roles = current_assignment, current_roles
            if event_applied:
                stages = demand_probe._stage_demand(
                    env.pursuer_positions(), env.evader_positions(),
                    env.evader_velocities(), env.task_mask[:env.num_evaders],
                    active_mask,
                )
                kappa = np.asarray([mode_capacity(int(x)) for x in stages])
                counts = np.asarray([
                    np.count_nonzero(
                        active_mask
                        & (env.roles[:env.num_pursuers] == ROLE_CORE)
                        & (env.assignment[:env.num_pursuers] == j)
                    )
                    for j in range(env.num_evaders)
                ])
                active_tasks = env.task_mask[:env.num_evaders].astype(bool)
                invalid = int(np.any(counts[active_tasks] < kappa[active_tasks]))
                post_invalid.append(invalid)
                if capacity_recovery is None and not invalid:
                    capacity_recovery = step - team_change_step

        if gui and renderer is not None and step % 2 == 0:
            renderer.update(env)

    if gui and renderer is not None:
        renderer.update(env)

    return {
        "reward": total_reward,
        "steps": step,
        "outcome": env.outcome,
        "captured": 1.0 if env.outcome == "all_captured" else 0.0,
        "n_captured": int(np.count_nonzero(env.captured[:env.num_evaders])),
        "strict": bool(strict_any),
        "event_applied": bool(event_applied),
        "capacity_recovery_steps": capacity_recovery,
        "strict_recovery_steps": strict_recovery,
        "post_invalid_frac": float(np.mean(post_invalid)) if post_invalid else None,
        "target_switches": int(target_switches),
        "role_switches": int(role_switches),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _arguments()
    # scale -> (num_pursuers, num_evaders); feeds run_episode and seed derivation
    _scale = tuple(int(x) for x in str(args.scale).split("v"))
    if len(_scale) != 2:
        raise SystemExit(f"invalid --scale {args.scale!r}; expected e.g. 6v2")
    gui = False
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    renderer = None

    env = make_env(args)

    model = load_sdgcr(
        args.checkpoint, device, args.ablation, args.d_cap, args.hold_steps,
        allow_protocol_mismatch=args.allow_protocol_mismatch,
        map_size=args.map_size, num_obstacles=args.num_obstacles,
        lock_on_close3=bool(getattr(args, "lock_on_close3", False)),
        allow_legacy_predictor=bool(getattr(args, "allow_legacy_predictor", False)),
    )

    outcomes = {"all_captured": 0, "evader_goal_success": 0, "timeout": 0}
    rewards = []
    captures = []
    all_results = []
    per_seed = {}

    for seed in args.seeds:
        seed_outcomes = {"all_captured": 0, "evader_goal_success": 0, "timeout": 0}
        seed_rewards = []
        seed_captures = []
        for ep in range(args.episodes_per_seed):
            episode_index = 0 if args.repeat_same_scenario else ep
            result = run_episode(env, args.method, model, device, seed, episode_index,
                                 args.max_steps, gui, renderer,
                                 args.team_change, args.team_change_step,
                                 num_pursuers=_scale[0], num_evaders=_scale[1],
                                 raw_seed_sequence=args.raw_seed_sequence)
            all_results.append(result)
            outcomes[result["outcome"]] += 1
            seed_outcomes[result["outcome"]] += 1
            rewards.append(result["reward"])
            seed_rewards.append(result["reward"])
            captures.append(result["captured"])
            seed_captures.append(result["captured"])
        per_seed[str(seed)] = {
            "capture_rate": float(np.mean(seed_captures)),
            "mean_reward": float(np.mean(seed_rewards)),
            "outcomes": seed_outcomes,
        }

    n_total = len(args.seeds) * args.episodes_per_seed
    summary = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "ablation": args.ablation,
        "architecture": (
            model["ablation_spec"].architecture() if args.method == "sdgcr" else None
        ),
        "protocol": {
            "map_size": args.map_size,
            "num_obstacles": args.num_obstacles,
            "d_cap": args.d_cap,
            "hold_steps": args.hold_steps,
            "max_steps": args.max_steps,
        },
        "scales": args.scale,
        # record the allocator switch that training hardcodes to True for the
        # SDGCR lines; without it the summary cannot self-certify its config
        "lock_on_close3": bool(getattr(args, "lock_on_close3", False)),
        # provenance for legacy (version<9) checkpoints: the 'predictor' key was
        # present but is NOT in the Actor's input path (proved by the strict actor
        # load); recorded so a reader can tell these results from predictor-free ones
        "allow_legacy_predictor": bool(getattr(args, "allow_legacy_predictor", False)),
        "legacy_predictor_present": model.get("legacy_predictor_present"),
        "predictor_used_by_actor": model.get("predictor_used_by_actor"),
        "loaded_obs_dim": model.get("loaded_obs_dim"),
        "loaded_ring_slots": model.get("loaded_ring_slots"),
        "loaded_stage_embedding": model.get("loaded_stage_embedding"),
        "loaded_graph_embedding": model.get("loaded_graph_embedding"),
        "seeds": args.seeds,
        "seed_mode": "raw_sequence" if args.raw_seed_sequence else "derived_test",
        "repeat_same_scenario": args.repeat_same_scenario,
        # 场景协议版本决定同一 (seed, episode, scale) 生成哪个场景；
        # v2 goal_directed_v2_goal_margin = 论文主表口径，
        # v4 goal_directed_v4_dispersed_nontrivial = 更难的 dispersed 实验线。
        # 跨版本的数字不可并列比较。
        "scenario_protocol_version": _SCENARIO_PROTOCOL_VERSION,
        "episodes_per_seed": args.episodes_per_seed,
        "total_episodes": n_total,
        "capture_rate": outcomes["all_captured"] / max(1, n_total),
        "target_capture_rate": float(np.mean([
            r["n_captured"] >= 1 for r in all_results
        ])),
        "strict_formation_rate": float(np.mean([r["strict"] for r in all_results])),
        "team_change": args.team_change,
        "team_change_step": args.team_change_step,
        "event_application_rate": float(np.mean([r["event_applied"] for r in all_results])),
        "capacity_recovery_rate": float(np.mean([
            r["capacity_recovery_steps"] is not None for r in all_results
        ])),
        "mean_capacity_recovery_steps": float(np.mean([
            r["capacity_recovery_steps"] for r in all_results
            if r["capacity_recovery_steps"] is not None
        ])) if any(r["capacity_recovery_steps"] is not None for r in all_results) else None,
        "strict_recovery_rate": float(np.mean([
            r["strict_recovery_steps"] is not None for r in all_results
        ])),
        "mean_strict_recovery_steps": float(np.mean([
            r["strict_recovery_steps"] for r in all_results
            if r["strict_recovery_steps"] is not None
        ])) if any(r["strict_recovery_steps"] is not None for r in all_results) else None,
        "mean_post_invalid_frac": float(np.mean([
            r["post_invalid_frac"] for r in all_results
            if r["post_invalid_frac"] is not None
        ])),
        "median_target_switches": float(np.median([r["target_switches"] for r in all_results])),
        "median_role_switches": float(np.median([r["role_switches"] for r in all_results])),
        "escape_rate": outcomes["evader_goal_success"] / max(1, n_total),
        "timeout_rate": outcomes["timeout"] / max(1, n_total),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "per_seed": per_seed,
        "episodes": all_results,
    }

    print(f"\n== {args.method.upper()} {args.scale} evaluation ==")
    print(f"seeds: {args.seeds} | episodes: {n_total}")
    print(f"capture={summary['capture_rate']:.3f} "
          f"escape={summary['escape_rate']:.3f} "
          f"timeout={summary['timeout_rate']:.3f} "
          f"reward={summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"summary written to {args.output}")

    if gui:
        print("close the PyBullet window to exit")
        try:
            input("press Enter to close the PyBullet window...")
        except EOFError:
            pass
    if renderer is not None:
        renderer.close()


if __name__ == "__main__":
    main()
