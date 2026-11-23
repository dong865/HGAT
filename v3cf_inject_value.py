"""把离线训练好的价值头注入 V3 checkpoint（生成新的续训起点）。

只替换 value_head 与 value_head_target 的权重，其余网络/优化器原样继承；
schema 不变，下游 --init-v3-from / runtime 加载器无需改动。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="基础 V3 checkpoint（如 dcap08c/best_4v2.pth）")
    p.add_argument("--value-head", required=True, help="离线训练产物 value_head_trained.pth")
    p.add_argument("--out", required=True, help="输出 checkpoint 路径")
    args = p.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    if base.get("architecture_version") != "v3_sparse_demand_value_hybrid_v1":
        raise ValueError("base 必须是 V3 hybrid checkpoint")

    trained = torch.load(args.value_head, map_location="cpu", weights_only=False)
    # 离线产物可能是裸 state_dict，也可能带包装键
    state = trained.get("value_head") if isinstance(trained, dict) and "value_head" in trained else trained
    if not isinstance(state, dict):
        raise ValueError("无法从 --value-head 提取 value_head state_dict")
    # 形状校验：与 base 的 value_head 同键同形状
    ref = base["value_head"]
    missing = set(ref) - set(state)
    if missing:
        raise ValueError(f"训练后的价值头缺键: {sorted(missing)[:5]}")
    for k in ref:
        if tuple(ref[k].shape) != tuple(state[k].shape):
            raise ValueError(f"形状不符 {k}: {ref[k].shape} vs {state[k].shape}")

    base["value_head"] = {k: state[k].clone() for k in ref}
    base["value_head_target"] = {k: v.clone() for k, v in base["value_head"].items()}
    base.setdefault("metadata", {})["v3cf_value_injection"] = {
        "source": str(Path(args.value_head).resolve()),
        "base": str(Path(args.base).resolve()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(base, tmp)
    tmp.replace(out)
    print(f"injected -> {out}", flush=True)


if __name__ == "__main__":
    main()
