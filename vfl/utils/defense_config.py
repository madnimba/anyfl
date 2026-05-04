"""YAML config for ``scripts/run_attack_defense.py`` (attack + RGAR defense).

Extends the attack experiment schema with an optional top-level ``defense:`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from dataclasses import replace as dc_replace
from typing import Any, Dict, Optional

import yaml

from server_rgar_defense import RGARConfig
from vfl.utils.attack_config import AttackExperimentConfig, attack_experiment_config_from_dict


def _norm_dataset(name: str) -> str:
    return str(name).strip().upper().replace("-", "")


def _is_rgb_embedding_defense_dataset(dataset_name: str) -> bool:
    """Datasets that use ``KPartyEmbeddingFusion`` in the defense runner."""
    d = _norm_dataset(dataset_name)
    return d in {"CIFAR10", "CIFAR100", "STL10"}


# Flatten-VFL defaults in ``RGARConfig`` use low ``tau_pair`` + aggressive recon floors
# tuned for ~392-dim ReLU flats. RGB embeddings (256+256) + RandAug inflate ``s_pair``,
# which otherwise marks ~100% of batches “suspicious” and replaces most of ``h_A`` with
# a still-noisy ``g(h_B,y)`` — worse than naked poisoned training. These defaults apply
# only when ``use_rgar_vision_defaults`` is true and the dataset is CIFAR/STL-like;
# YAML ``defense.rgar:`` entries still override any key below.
_RGAR_VISION_DEFAULTS: Dict[str, Any] = {
    "tau_pair": 0.58,
    "tau_global": 0.07,
    "watch_window_epochs": 5,
    "min_w_recon_when_suspicious": 0.28,
    "suspicion_recon_strength": 0.42,
    "global_recon_boost": 0.30,
    "tau_recon_lo": 0.20,
    "tau_recon_hi": 0.58,
    "modality_dropout_p": 0.0,
    "proto_snap_weight": 0.10,
    "pair_w_joint": 0.40,
    "rho_decay_on_attrib": 0.45,
}


@dataclass(frozen=True)
class DefensePipelineConfig:
    """Which baselines to run after building the poisoned attacker view."""

    # Standard end-to-end training on poisoned data (no mitigation).
    run_naked_poisoned: bool = True
    # RGAR full (reconstruction + trust fusion).
    rgar_full: bool = True
    # RGAR ablation: downweight / trust only (no soft recon blend).
    rgar_downweight: bool = False
    # Key-value overrides for :class:`server_rgar_defense.RGARConfig` (e.g. ``recon_epochs``).
    rgar: Dict[str, Any] = None  # type: ignore[assignment]
    # When true and dataset is CIFAR-10/100 or STL-10, merge ``_RGAR_VISION_DEFAULTS`` before
    # applying ``rgar`` YAML (so embedding + RandAug runs do not use flatten-tuned thresholds).
    use_rgar_vision_defaults: bool = True

    def __post_init__(self):
        if self.rgar is None:
            object.__setattr__(self, "rgar", {})


@dataclass(frozen=True)
class DefenseExperimentBundle:
    """Attack settings + defense toggles loaded from one YAML file."""

    attack: AttackExperimentConfig
    defense: DefensePipelineConfig


def load_defense_experiment_bundle(path: str) -> DefenseExperimentBundle:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    def_raw = dict(raw.get("defense") or {})
    attack_raw = {k: v for k, v in raw.items() if k != "defense"}
    atk = attack_experiment_config_from_dict(attack_raw)
    dpl = DefensePipelineConfig(
        run_naked_poisoned=bool(def_raw.get("run_naked_poisoned", True)),
        rgar_full=bool(def_raw.get("rgar_full", True)),
        rgar_downweight=bool(def_raw.get("rgar_downweight", False)),
        rgar={str(k): v for k, v in (def_raw.get("rgar") or {}).items()},
        use_rgar_vision_defaults=bool(def_raw.get("use_rgar_vision_defaults", True)),
    )
    return DefenseExperimentBundle(attack=atk, defense=dpl)


def rgar_config_from_defense_block(
    dpl: DefensePipelineConfig,
    *,
    dataset_name: Optional[str] = None,
) -> RGARConfig:
    base = RGARConfig()
    merged: Dict[str, Any] = {}
    if (
        bool(dpl.use_rgar_vision_defaults)
        and dataset_name is not None
        and _is_rgb_embedding_defense_dataset(dataset_name)
    ):
        merged.update(_RGAR_VISION_DEFAULTS)
    merged.update(dict(dpl.rgar))
    valid = {f.name for f in fields(RGARConfig)}
    kwargs: Dict[str, Any] = {}
    for k, v in merged.items():
        if k not in valid:
            raise ValueError(f"Unknown RGARConfig key in defense.rgar: {k!r}")
        kwargs[k] = v
    return dc_replace(base, **kwargs) if kwargs else base
