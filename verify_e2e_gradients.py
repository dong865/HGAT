"""Standalone verification that the pure policy loss reaches the upper heads.

Forward under test:

    snap -> encoder(differentiable=True) -> {obs(P), relation_costs, demand}
    -> actor(obs) -> actions -> critic1(obs, actions) -> Q
    -> actor_loss = -Q.mean()  -> backward

Three passes:

A. Full end-to-end: nothing is detached.
B. detach(P) INSIDE the forward: ``differentiable_task_plan`` is monkey-patched
   to return a detached plan before ``soft_coalition_context`` consumes it.
   Node embeddings (h_i, task embeddings) keep their gradient paths, so this
   isolates the plan-specific contribution.  relation/demand heads only reach
   the loss through the plan, so their gradients here must vanish if and only
   if the plan path is the sole route -- which the architecture guarantees and
   this pass empirically confirms.
C. Full obs detach (legacy control): obs is detached after encoding; every
   encoder gradient vanishes.  This shows what "no graph path at all" looks
   like, and is NOT a plan-specific intervention.

Scope note: random init, single snapshot -- this checks computational-graph
connectivity only; it is not a training-effectiveness result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if (_HERE / "stage_env.py").exists():
    sys.path.insert(0, str(_HERE.parent))

import crazyflie_examples.six_v_two_hgat.stage_networks as sn
from crazyflie_examples.six_v_two_hgat.stage_env import StageCoalitionEnv
from crazyflie_examples.six_v_two_hgat.stage_networks import (
    SharedActor,
    SetInvariantCritic,
    StageFeatureEncoder,
)


def grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().square().sum())
    return total ** 0.5


def reset_grads(*modules: torch.nn.Module) -> None:
    for m in modules:
        for p in m.parameters():
            p.grad = None


def build_modules(args: argparse.Namespace, device: torch.device):
    env = StageCoalitionEnv(
        length=args.map_size,
        num_obstacle=args.num_obstacles,
        max_pursuers=6,
        max_evaders=3,
        d_cap=args.d_cap,
    )
    encoder = StageFeatureEncoder(
        hidden_dim=args.hidden_dim,
        node_dim=9,
        max_pursuers=6,
        max_evaders=3,
        future_steps=args.future_steps,
        seq_len=args.seq_len,
        map_size=args.map_size,
        num_obstacles=args.num_obstacles,
        device=device,
    )
    actor = SharedActor(encoder.obs_dim, hidden_dim=args.hidden_dim).to(device)
    critic = SetInvariantCritic(encoder.obs_dim, hidden_dim=args.hidden_dim).to(device)
    return env, encoder, actor, critic


def actor_loss_backward(encoder, actor, critic, obs, masks, detach_obs: bool):
    if detach_obs:
        obs = obs.detach()
    actions = actor(obs)
    q1 = critic(obs, actions, masks)
    loss = -q1.mean()
    reset_grads(encoder, actor, critic)
    loss.backward()
    return float(loss.detach())


def report(name, loss, encoder, actor, critic):
    grads = {
        "actor_loss": loss,
        "grad_st_hgat": grad_norm(encoder.st_hgat),
        "grad_relation_head": grad_norm(encoder.relation_head),
        "grad_demand_head": grad_norm(encoder.demand_head),
        "grad_actor": grad_norm(actor),
        "grad_critic": grad_norm(critic),
    }
    print(f"\n=== {name} ===")
    for k, v in grads.items():
        print(f"{k}: {v:.6f}")
    return grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-size", type=float, default=8.0)
    parser.add_argument("--num-obstacles", type=int, default=10)
    parser.add_argument("--d-cap", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env, encoder, actor, critic = build_modules(args, device)
    env.reset(num_pursuers=6, num_evaders=2, scenario_seed=args.seed)
    for _ in range(3):
        env.step(np.random.uniform(-1.0, 1.0, size=(6, 2)).astype(np.float32))

    snap = env.snapshot()

    # -------- Pass A: full end-to-end --------
    encoder.reset(step=0)
    enc_a = encoder.encode(snap, differentiable=True)
    obs_a = enc_a["obs"].unsqueeze(0)
    masks_a = enc_a["masks"].unsqueeze(0)
    loss_a = actor_loss_backward(encoder, actor, critic, obs_a, masks_a, detach_obs=False)
    grads_a = report("Pass A: full end-to-end", loss_a, encoder, actor, critic)

    # -------- Pass B: detach(P) inside the forward --------
    # Monkey-patch differentiable_task_plan so the plan is detached BEFORE
    # soft_coalition_context consumes it.  Values are unchanged; only the
    # plan->context gradient edge is removed.
    original_fn = sn.differentiable_task_plan

    def detached_plan_fn(*fn_args, **fn_kwargs):
        return original_fn(*fn_args, **fn_kwargs).detach()

    sn.differentiable_task_plan = detached_plan_fn
    try:
        encoder.reset(step=0)
        enc_b = encoder.encode(snap, differentiable=True)
        obs_b = enc_b["obs"].unsqueeze(0)
        masks_b = enc_b["masks"].unsqueeze(0)
        loss_b = actor_loss_backward(encoder, actor, critic, obs_b, masks_b, detach_obs=False)
    finally:
        sn.differentiable_task_plan = original_fn
    grads_b = report("Pass B: detach(P_t) before soft context", loss_b, encoder, actor, critic)

    # -------- Pass C: full obs detach (legacy control) --------
    encoder.reset(step=0)
    enc_c = encoder.encode(snap, differentiable=True)
    obs_c = enc_c["obs"].unsqueeze(0)
    masks_c = enc_c["masks"].unsqueeze(0)
    loss_c = actor_loss_backward(encoder, actor, critic, obs_c, masks_c, detach_obs=True)
    grads_c = report("Pass C: full obs detach (control)", loss_c, encoder, actor, critic)

    # -------- Checks --------
    print("\n=== Checks ===")
    ok = True
    if grads_a["grad_relation_head"] <= 1e-8 or grads_a["grad_demand_head"] <= 1e-8:
        print("FAIL: relation/demand heads got no gradient in Pass A")
        ok = False
    else:
        print(f"PASS A: relation={grads_a['grad_relation_head']:.6f}, "
              f"demand={grads_a['grad_demand_head']:.6f} (nonzero)")

    if grads_b["grad_relation_head"] > 1e-10 or grads_b["grad_demand_head"] > 1e-10:
        print("FAIL: relation/demand gradients survived detach(P) -- plan is not their only path")
        ok = False
    else:
        print("PASS B: detach(P) zeroes relation/demand gradients "
              "(plan is their sole route to the loss)")

    if grads_b["grad_st_hgat"] <= 1e-8:
        print("WARN: ST-HGAT also lost all gradient under detach(P); "
              "expected nonzero via direct h_i paths -- check encoder wiring")
    else:
        print(f"PASS B: ST-HGAT retains gradient via non-plan paths "
              f"({grads_b['grad_st_hgat']:.6f}), as expected")

    if grads_c["grad_st_hgat"] > 1e-10 or grads_c["grad_relation_head"] > 1e-10:
        print("FAIL: full obs detach should zero every encoder gradient")
        ok = False
    else:
        print("PASS C: full obs detach zeros all encoder gradients (sanity)")

    print("\nInterpretation:")
    print("- Pass A vs Pass B isolates the plan-specific gradient: relation and")
    print("  demand heads receive the pure value objective ONLY through P_t.")
    print("- Pass B ST-HGAT > 0 confirms its remaining gradient flows through")
    print("  node/task embeddings, not the plan.  Do NOT cite ST-HGAT as")
    print("  plan-verified.")
    print("- Random init, single snapshot: graph connectivity check only.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
