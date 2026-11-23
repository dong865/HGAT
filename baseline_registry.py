"""Canonical names and availability for the formal 6V2 experiment suite."""

from .ablation_config import canonical_ablation_names


COMPARISON_BASELINES = {
    "ibr": {
        "label": "Finite-Horizon IBR",
        "implemented": True,
        "train_entry": "main_ibr_baseline",
        "eval_method": "ibr",
    },
    "mappo": {
        "label": "MAPPO",
        "implemented": True,
        "train_entry": "main_mappo_baseline",
        "eval_method": "mappo",
    },
    "maddpg": {
        "label": "MADDPG",
        "implemented": True,
        "train_entry": "main_maddpg_baseline",
        "eval_method": "maddpg",
    },
    "g2anet_ac": {
        "label": "G2ANet Actor-Critic",
        "implemented": True,
        "train_entry": "main_g2anet_ac_baseline",
        "eval_method": "g2anet_ac",
    },
    "cbba_ibr": {
        "label": "CBBA + IBR",
        "implemented": True,
        "train_entry": "main_cbba_ibr",
        "eval_method": "cbba_ibr",
    },
}

ABLATIONS = {
    name: f"SDGCR/{name}" for name in canonical_ablation_names()
}

# Compatibility registry for old scripts/checkpoints. These are structural
# variants, not the independent baselines in the revised paper.
BASELINES = {
    "grouped_maddpg": "Grouped MADDPG",
    "independent_maddpg": "Independent MADDPG",
    "no_graph_maddpg": "No-Graph MADDPG",
    "role_based_maddpg": "Role-Based MADDPG",
    "unified_group_maddpg": "Unified-Group MADDPG",
}

# 本文方法单列，避免在论文和实验表中误称为基线。
PROPOSED_METHOD = {
    "stage_feasible_dgcr": "Stage-Feasible Dynamic Graph Coalition Reconfiguration (SDGCR)"
}

# 旧路线方法（ST-HGAT + 自适应组间策略共识），论文修订后不再作为本文方法，
# 仅保留名称供旧检查点/脚本兼容。
LEGACY_METHODS = {
    "st_hgat_acpc_maddpg": "ST-HGAT + Adaptive Cross-group Policy Consensus MADDPG (legacy route)"
}

ALL_METHODS = {**BASELINES, **PROPOSED_METHOD}

# Formal paper methods are kept separate from compatibility names.
FORMAL_METHODS = {**COMPARISON_BASELINES, **PROPOSED_METHOD}


def require_runnable_baseline(name: str):
    if name not in COMPARISON_BASELINES:
        raise KeyError(f"unknown formal baseline {name!r}")
    metadata = COMPARISON_BASELINES[name]
    if not metadata["implemented"]:
        raise NotImplementedError(
            f"formal baseline {name!r} is registered but its faithful "
            "reproduction is not implemented yet",
        )
    return metadata
