"""Canonical one-factor ablation definitions for formal SDGCR experiments."""

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class AblationSpec:
    name: str = "none"
    node_encoder: str = "st_hgat"
    coalition_pool: str = "attention"
    coalition_mode: str = "dynamic"
    use_stage_feasibility: bool = True

    def architecture(self) -> Dict[str, object]:
        values = asdict(self)
        values.pop("name")
        return values


_SPECS = {
    "none": AblationSpec(),
    "no_graph": AblationSpec("no_graph", node_encoder="shared_mlp"),
    "mean_pool": AblationSpec("mean_pool", coalition_pool="mean"),
    "fixed_concat": AblationSpec("fixed_concat", coalition_pool="fixed_concat"),
    "no_pool": AblationSpec("no_pool", coalition_pool="none"),
    "fixed_coalition": AblationSpec(
        "fixed_coalition", coalition_mode="fixed_membership",
    ),
    "no_feasibility": AblationSpec(
        "no_feasibility", use_stage_feasibility=False,
    ),
}


def canonical_ablation_names() -> Tuple[str, ...]:
    return tuple(name for name in _SPECS if name != "none")


def resolve_ablation(name: str) -> AblationSpec:
    normalized = str(name).strip()
    if normalized not in _SPECS:
        choices = ", ".join(_SPECS)
        raise ValueError(
            f"invalid formal ablation {name!r}; choose exactly one of: {choices}",
        )
    return _SPECS[normalized]
