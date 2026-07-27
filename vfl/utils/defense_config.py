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


def _is_tabular_party_mlp_defense_dataset(dataset_name: str) -> bool:
    """UCI-HAR / Mushroom / UCI-BANK tabular stacks (``KPartyTabularMLP`` family)."""
    d = _norm_dataset(dataset_name)
    return d in {"UCIHAR", "HAR", "UCIMUSHROOM", "MUSHROOM", "UCIBANK", "BANK"}


def _is_ucihar_family(dataset_name: str) -> bool:
    d = _norm_dataset(dataset_name)
    return d in {"UCIHAR", "HAR"}


def _is_ucibank_family(dataset_name: str) -> bool:
    d = _norm_dataset(dataset_name)
    return d in {"UCIBANK", "BANK"}


# Tabular party embeddings are stable (no RandAug); defaults emphasize reference + recon
# budget and moderate blend floors. HAR asymmetric (96 vs 8 dims) gets an extra overlay
# (larger ``recon_hidden``, slightly higher ``tau_pair``) — override via ``defense.rgar:``.
_RGAR_TABULAR_DEFAULTS: Dict[str, Any] = {
    "ref_frac": 0.12,
    "ref_warmup_epochs": 12,
    "recon_epochs": 260,
    "recon_batch_size": 64,
    "recon_lr": 1.4e-3,
    "tau_pair": 0.28,
    "tau_global": 0.04,
    "watch_window_epochs": 3,
    "min_w_recon_when_suspicious": 0.70,
    "suspicion_recon_strength": 0.88,
    "global_recon_boost": 0.55,
    "tau_recon_lo": 0.08,
    "tau_recon_hi": 0.42,
    "proto_snap_weight": 0.20,
    "pair_w_joint": 0.50,
    "modality_dropout_p": 0.0,
    "recon_hidden": 320,
}

# Asymmetric HAR: ``g(h_B,y) -> h_A`` is ill-posed (8-D passive vs 96-D attacker). Blending
# toward a wrong ``h_hat`` *hurts* vs naked. Use ``soft_recon_h_hat_mode: proto_a`` so
# suspicious rows pull toward **reference class prototypes** ``p_A[y]`` (honest attacker
# geometry keyed by the still-correct label), not the unreliable honest-view MLP.
_RGAR_UCIHAR_TABULAR_OVERLAY: Dict[str, Any] = {
    # Repair target: blend suspicious h_A toward p_A[y] (class prototype from clean ref).
    # p_A[y] is reliable because y is still correct under cluster-swap; avoids the ill-posed
    # g(h_B,y)->h_A map (8-D passive cannot uniquely determine 96-D attacker embedding).
    "soft_recon_h_hat_mode": "proto_a",
    # Keep prototypes aligned with the evolving encoder (one cheap ref forward pass / epoch).
    "refit_ref_every_epoch": True,
    # After global attribution fires, freeze attacker encoder: only passive + server update.
    # Eliminates the (1-w)*h_A gradient path that let the poisoned encoder keep learning.
    "freeze_attacker_on_attribution": True,
    # Suspicion scoring: reduce joint-cosine weight to avoid false alarms on noisy 96||8 joint.
    "pair_w_joint": 0.10,
    "pair_w_proto": 1.0,
    "tau_pair": 0.38,
    "tau_global": 0.045,
    "watch_window_epochs": 3,
    # High blend weights: p_A[y] is trustworthy → aggressively replace corrupted h_A.
    # Previous values (0.14 / 0.20) left 68%+ of poisoned embedding reaching the server.
    "min_w_recon_when_suspicious": 0.85,
    "suspicion_recon_strength": 0.94,
    # global_recon_boost is now applied directly (not scaled by 1-rho_a) for proto_a mode.
    "global_recon_boost": 0.92,
    "proto_snap_weight": 0.0,
    "tau_recon_lo": 0.05,
    "tau_recon_hi": 0.38,
    "recon_hidden": 384,
    "recon_label_emb_dim": 48,
    "recon_cosine_weight": 0.0,
    "modality_dropout_p": 0.0,
}


# UCI-BANK asymmetric: dim(h_A)=96, dim(h_B)=2 — even more extreme than HAR.
# The passive 2-D linear projection is intentionally weak (attack design), making
# g(h_B,y)->h_A completely hopeless. class_flip gives donor_label_flip~0.90, so
# 90% of suspicious rows have a genuine opposite-class donor — p_A[y] is the
# exact right repair direction (y stays correct under cluster-swap).
# Same refit+freeze strategy as HAR; lower joint weight since 2-D h_B dominates nothing.
_RGAR_BANK_TABULAR_OVERLAY: Dict[str, Any] = {
    # Repair target: class prototype (binary: only 2 prototypes, both well-estimated on ref).
    "soft_recon_h_hat_mode": "proto_a",
    "refit_ref_every_epoch": True,
    "freeze_attacker_on_attribution": True,
    # Scoring: h_B is 2-D linear → joint [h_A(96), h_B(2)] cosine is dominated by h_A.
    # Reduce joint weight to near-zero; rely on Mahalanobis d_A to drive suspicion.
    "pair_w_joint": 0.06,
    "pair_w_proto": 1.0,
    # Lower tau_pair: class_flip creates strong h_A inconsistency (donor is opposite class),
    # so even a moderate d_A signal should exceed the threshold quickly.
    "tau_pair": 0.28,
    "tau_global": 0.045,
    "watch_window_epochs": 3,
    # High blend: p_A[y] is reliable (binary, balanced ref, donor_label_flip~0.90).
    "min_w_recon_when_suspicious": 0.88,
    "suspicion_recon_strength": 0.96,
    # proto_a mode: global_recon_boost applied directly (not scaled by 1-rho_A).
    "global_recon_boost": 0.94,
    "proto_snap_weight": 0.0,
    "tau_recon_lo": 0.04,
    "tau_recon_hi": 0.28,
    # Aggressive rho decay: passive (2-D) cannot compensate → push attacker trust to floor fast.
    "rho_floor": 0.08,
    "rho_decay_on_attrib": 0.40,
    "recon_hidden": 192,
    "recon_label_emb_dim": 16,   # binary: only 2 classes
    "recon_cosine_weight": 0.0,
    "modality_dropout_p": 0.0,
}


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
    # When true and dataset is UCI-HAR / Mushroom, merge ``_RGAR_TABULAR_DEFAULTS`` (and HAR
    # overlay) before ``defense.rgar:`` — mutually exclusive with vision presets by dataset.
    use_rgar_tabular_defaults: bool = True
    # Adaptive attacker: an attacker who knows RGAR is deployed and knows which rows
    # form the reference set R will simply not poison them, so R stays clean *and*
    # uninformative about the attack. Excludes R from the victim set before donor
    # assignment. False (default) = the non-adaptive attacker used in all submitted
    # numbers. This is a strictly stronger threat model, not a different attack.
    adaptive_exclude_reference: bool = False

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
        use_rgar_tabular_defaults=bool(def_raw.get("use_rgar_tabular_defaults", True)),
        adaptive_exclude_reference=bool(def_raw.get("adaptive_exclude_reference", False)),
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
    elif (
        bool(dpl.use_rgar_tabular_defaults)
        and dataset_name is not None
        and _is_tabular_party_mlp_defense_dataset(dataset_name)
    ):
        merged.update(_RGAR_TABULAR_DEFAULTS)
        if _is_ucihar_family(dataset_name):
            merged.update(_RGAR_UCIHAR_TABULAR_OVERLAY)
        elif _is_ucibank_family(dataset_name):
            merged.update(_RGAR_BANK_TABULAR_OVERLAY)
    merged.update(dict(dpl.rgar))
    valid = {f.name for f in fields(RGARConfig)}
    kwargs: Dict[str, Any] = {}
    for k, v in merged.items():
        if k not in valid:
            raise ValueError(f"Unknown RGARConfig key in defense.rgar: {k!r}")
        kwargs[k] = v
    return dc_replace(base, **kwargs) if kwargs else base
