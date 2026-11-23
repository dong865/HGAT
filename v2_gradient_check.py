"""V2 三 pass 纯价值梯度检查（交接文档第 15 节 B 组）。

仅使用 L_value = -mean(Q1(O, actor(O)))，不混入动作惩罚、关系监督或需求正则。
Pass B 的干预作用于"统一上下文函数消费 K 之前"（monkey-patch 点即
build_membership_context 的调用处，P 真正参与 forward 的张量）。
输出 JSON：v2_deliverables/V2_GRADIENT_CHECK.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if (_HERE / "v2_training.py").exists():
    sys.path.insert(0, str(_HERE.parent))

import crazyflie_examples.six_v_two_hgat.v2_training as VT
from crazyflie_examples.six_v_two_hgat.continuous_membership import ContinuousDemandHead
from crazyflie_examples.six_v_two_hgat.stage_env import StageCoalitionEnv
from crazyflie_examples.six_v_two_hgat.stage_networks import (
    SetInvariantCritic, SharedActor, StageFeatureEncoder,
)
from crazyflie_examples.six_v_two_hgat.v2_controller import v2_obs_dim
from crazyflie_examples.six_v_two_hgat.v2_training import (
    compute_demand_and_plan, encode_segment_v2,
)

OUT = _HERE / "v2_deliverables" / "V2_GRADIENT_CHECK.json"


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().square().sum())
    return total ** 0.5


def param_l1_delta(before, module):
    total = 0.0
    state = module.state_dict()
    for k, v in state.items():
        total += float((v.detach().cpu() - before[k]).abs().sum())
    return total


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = StageCoalitionEnv(length=8.0, num_obstacle=10, max_pursuers=6,
                            max_evaders=3, d_cap=1.5, hold_steps=5, seed=3)
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=11)
    encoder = StageFeatureEncoder(
        max_pursuers=6, max_evaders=3, num_obstacles=10, map_size=8.0,
        hidden_dim=32, future_steps=10, seq_len=10, node_dim=9, device=device,
        use_ring_slot_features=False, use_stage_embedding=False,
        use_role_embedding=False, use_graph_embedding=True).to(device)
    demand_head = ContinuousDemandHead(32, demand_init=3.0).to(device)
    actor = SharedActor(v2_obs_dim(6, 32, 10), hidden_dim=64).to(device)
    critic1 = SetInvariantCritic(v2_obs_dim(6, 32, 10), hidden_dim=64).to(device)
    # 注意：encoder 必须 train() 才能接梯度；固定种子使 dropout 模式可复现
    encoder.train()
    torch.manual_seed(123)

    # 构造一条 6 步片段（raw snapshot + assignment + action）
    assignment = [0, 0, 0, 1, 1, -1]
    segment = []
    for t in range(7):
        snap = env.snapshot()
        a = np.random.uniform(-1, 1, (6, 2)).astype(np.float32)
        segment.append({"snapshot": snap, "assignment": assignment,
                        "action": a, "reward": float(t), "done": False})
        env.step(a, max_steps=1000)

    def value_loss():
        batch = encode_segment_v2(
            encoder, demand_head, segment, plan_length=6,
            context_mode="soft", grad_enabled=True,
            demand_source="learned_continuous", fixed_demand=3.0,
            epsilon_ot=0.5, sinkhorn_iters=80,
            max_pursuers=6, max_evaders=3)
        with torch.enable_grad():
            acts = actor(batch["obs"])
            q = critic1(batch["obs"], acts, batch["mask"]).mean()
        return -q

    results = {}

    # ---------- Pass A：无干预 ----------
    for m in (encoder, demand_head, actor, critic1):
        m.zero_grad(set_to_none=True)
    loss_a = value_loss()
    loss_a.backward()
    results["pass_A"] = {
        "loss": float(loss_a.detach()),
        "grad_relation_head": grad_norm(encoder.relation_head),
        "grad_demand_head": grad_norm(demand_head),
        "grad_st_hgat": grad_norm(encoder.st_hgat),
        "grad_actor": grad_norm(actor),
        "grad_critic1": grad_norm(critic1),
    }

    # ---------- Pass B：统一上下文消费 K 之前 detach P ----------
    orig_ctx = VT.build_membership_context
    def detached_ctx(K, *a, **k):
        return orig_ctx(K.detach(), *a, **k)
    VT.build_membership_context = detached_ctx
    try:
        for m in (encoder, demand_head, actor, critic1):
            m.zero_grad(set_to_none=True)
        loss_b = value_loss()
        loss_b.backward()
        results["pass_B"] = {
            "loss": float(loss_b.detach()),
            "grad_relation_head": grad_norm(encoder.relation_head),
            "grad_demand_head": grad_norm(demand_head),
            "grad_st_hgat": grad_norm(encoder.st_hgat),
            "grad_actor": grad_norm(actor),
        }
    finally:
        VT.build_membership_context = orig_ctx

    # ---------- Pass C：整份 O detach（编码器侧归零，Actor 仍有梯度）----------
    orig_builder = VT.build_v2_obs_torch
    def detached_builder(*a, **k):
        return orig_builder(*a, **k).detach()
    VT.build_v2_obs_torch = detached_builder
    try:
        for m in (encoder, demand_head, actor, critic1):
            m.zero_grad(set_to_none=True)
        loss_c = value_loss()
        loss_c.backward()
        results["pass_C"] = {
            "loss": float(loss_c.detach()),
            "grad_relation_head": grad_norm(encoder.relation_head),
            "grad_demand_head": grad_norm(demand_head),
            "grad_st_hgat": grad_norm(encoder.st_hgat),
            "grad_actor": grad_norm(actor),
        }
    finally:
        VT.build_v2_obs_torch = orig_builder

    # ---------- 判定 ----------
    a, b, c = results["pass_A"], results["pass_B"], results["pass_C"]
    checks = {
        "A_relation_nonzero": a["grad_relation_head"] > 1e-10,
        "A_demand_nonzero": a["grad_demand_head"] > 1e-10,
        "B_relation_zero": b["grad_relation_head"] <= 1e-12,
        "B_demand_zero": b["grad_demand_head"] <= 1e-12,
        "B_sthgat_retained": b["grad_st_hgat"] > 1e-10,
        "C_encoder_zero": (c["grad_relation_head"] <= 1e-12
                           and c["grad_demand_head"] <= 1e-12
                           and c["grad_st_hgat"] <= 1e-12),
        "C_actor_retained": c["grad_actor"] > 1e-10,
    }
    results["checks"] = checks
    results["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    results["note"] = ("纯价值损失连通性检查（随机初始化、单片段）；"
                       "Sinkhorn 对 costs/demand 的有限差分见 V2_ACCEPTANCE_TESTS.log "
                       "（B finite-diff 两项）。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
