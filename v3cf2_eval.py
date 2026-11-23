"""v3cf2 在线闭环验收：V3cf2Policy（81 维价值头）与基线 HybridPolicy 同种子对比。

只读基线 checkpoint，不修改共享文件。指标：any/full 捕获、首捕时刻、切换数、非有限动作。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_v3_hybrid import build_evaluation_env
from .scenario_protocol_deploy5x5 import Deploy5x5Sampler
from .train_v3_hybrid import HybridPolicy, build_system, load_v3_hybrid_checkpoint
from .v3cf2_policy import ValueHead81, V3cf2Policy


def _load_system_clean(path, device):
    """同 load_v3_system，但清除 ckpt 里记录的 warm-start 回放路径（跨树部署必备）。"""
    payload = load_v3_hybrid_checkpoint(path)
    config = dict(payload["config"])
    config["init_policy_from"] = None
    config["init_value_from"] = None
    config["init_v3_from"] = None
    config.pop("warm_start_v3_from", None)
    config["device"] = str(torch.device(device))
    modules, _, _ = build_system(config)
    load_v3_hybrid_checkpoint(path, modules)
    return modules, config


def load_policy_with_value81(base_checkpoint, trained_value_path, device):
    modules, config = _load_system_clean(base_checkpoint, device)
    if trained_value_path:
        trained = torch.load(trained_value_path, map_location=device, weights_only=False)
        state = trained.get("value_head", trained) if isinstance(trained, dict) else trained
        head = ValueHead81(int(config["hidden_dim"])).to(device)
        head.load_state_dict(state)
        head.eval()
        modules["value_head"] = head
        policy = V3cf2Policy(modules, config)
    else:
        policy = HybridPolicy(modules, config)
    return policy, config


def run_episode(policy, env, seed, max_steps=700):
    env.reset(num_pursuers=4, num_evaders=2, scenario_seed=int(seed))
    policy.reset()
    decision = policy.decision(env)
    first = full = None
    switches = 0
    prev = None
    t = 0
    while t < max_steps:
        cur = np.asarray(decision["assignment"][:4], dtype=int)
        if prev is not None and not np.array_equal(cur, prev):
            switches += 1
        prev = cur
        actions = decision["actions"]
        if not np.isfinite(actions).all():
            raise FloatingPointError("nonfinite action")
        _, done, _ = env.step(actions, max_steps=max_steps)
        t += 1
        captured = int(np.count_nonzero(env.captured[:2]))
        if captured >= 1 and first is None:
            first = t * env.time_step
        if captured == 2 and full is None:
            full = t * env.time_step
        if done:
            break
        decision = policy.decision(env)
    return {"seed": int(seed), "captured": int(np.count_nonzero(env.captured[:2])),
            "first_capture_time": first, "full_capture_time": full,
            "switches": switches, "outcome": env.outcome}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--value-head", default=None)
    p.add_argument("--seeds", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, config = load_policy_with_value81(args.base_checkpoint, args.value_head, device)
    env = build_evaluation_env(config, v_max_e=config.get("v_max_e", 0.35), seed=0)
    env.scenario_sampler = Deploy5x5Sampler()
    env.d_cap = 0.8
    records = [run_episode(policy, env, s, int(config.get("max_steps", 700)))
               for s in args.seeds.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"records": records}, ensure_ascii=False))
    tmp.replace(out)
    n = len(records)
    print(f"{out.name}: n={n} any={sum(r['captured']>=1 for r in records)/n:.2f} "
          f"full={sum(r['captured']==2 for r in records)/n:.2f}")


if __name__ == "__main__":
    main()
