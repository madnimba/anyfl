from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional

import yaml

from vfl.data.types import DataConfig, NUSWIDEConfig
from vfl.train.loop import TrainConfig
from vfl.utils.config import ExperimentConfig


@dataclass(frozen=True)
class AttackSwapConfig:
    """
    Swap-only config for Phase II (attack).

    `strategies` are names accepted by `vfl.attack.swap.apply_cluster_swap_to_part`.
    """

    strategies: List[str] = None  # type: ignore[assignment]
    attacker_client_idx: int = 0
    # If True, override `attacker_client_idx` at runtime with the client whose
    # input is most informative on its own (largest leave-one-out drop on the
    # clean baseline). The probe choice is recorded in `swap_meta.json`.
    attacker_probe: bool = False
    topk: int = 3
    core_q: float = 0.60
    cluster_dir: str = "./clusters"
    protect_test: bool = True
    use_signature_cache: bool = False
    # If True, do not pass Phase-1 confidences into swap (full cluster = donor pool).
    # For UCI-BANK, high-conf "cores" are often label-aligned; ignoring conf mixes
    # harder donors and raises swap harm without using y.
    ignore_cluster_conf: bool = False
    # ``class_flip`` strategy only: stratified aux fraction used to estimate per-cluster
    # majority class. Aligns with the Phase-I aux budget (consistent threat model).
    class_flip_aux_frac: float = 0.05
    # RGB / SplitLeNet only: after clean baseline training, run the attacker client
    # encoder once and use L2 geometry in swap (optimal_topk / derangement / class_flip)
    # while still exchanging raw ``X_part`` pixels. Aligns donor choice with fusion
    # features (Phase-I vision clusters are encoder-based). Default False preserves
    # all existing tabular + legacy MNIST runs.
    use_clean_encoder_geometry: bool = False
    # When ``use_clean_encoder_geometry``, write ``clean/attacker_train_embeddings.npy``.
    save_attacker_embeddings: bool = True

    def __post_init__(self):
        if self.strategies is None:
            object.__setattr__(
                self,
                "strategies",
                [
                    "optimal_topk",
                    "derangement",
                    "paired_clusters",
                    "round_robin",
                    "random_clusters",
                    "random_per_sample",
                ],
            )


@dataclass(frozen=True)
class AttackExperimentConfig:
    """
    Attack experiment config = clean ExperimentConfig + swap block.

    We keep the same schema (dataset/k/seed/data/train/nuswide) as clean runs,
    and add `swap:` for the poisoning parameters.
    """

    dataset: str
    k_clients: int
    seed: int = 0
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    nuswide: Optional[NUSWIDEConfig] = None
    run_name: Optional[str] = None
    swap: AttackSwapConfig = AttackSwapConfig()
    # When True (MNIST / Fashion-MNIST only), use ``KPartySplitLeNet`` instead of
    # the default ``KPartyLegacyFlattenVFL`` (flatten locals + small MLP server).
    use_split_lenet: bool = False
    # If True, pass train labels into stealth metrics (donor_label_flip_rate). False =
    # label-free evaluation consistent with “only server sees y” for poisoning craft.
    # True: include label-based stealth diagnostics (swap never uses y).
    stealth_oracle_labels: bool = True
    # UCI-BANK only: ``balanced`` (default, round-robin numeric+one-hot) or
    # ``skewed_attacker`` (route the top ``bank_attack_share`` of MI-ranked
    # features to ``swap.attacker_client_idx`` so the passive side is weak).
    bank_attack_split: str = "balanced"
    bank_attack_share: float = 0.75
    # UCI-BANK only: ``compact`` (default) or ``asymmetric`` (wider attacker bottom
    # + larger attacker emb share at the server head, intended for ``skewed_attacker``).
    bank_attack_model: str = "compact"
    # UCI-HAR only: ``mi_ranked`` (default) matches Phase-I MI partition; ``even`` =
    # legacy sequential blocks (misaligned with default clustering artifacts).
    har_attack_split: Optional[str] = None
    har_attack_share: Optional[float] = None
    har_attack_model: str = "compact"
    # MI partition aux budget (align with clustering ``aux_labeled_frac``).
    har_aux_labeled_frac: float = 0.03


def _maybe_dataclass(cls, obj: Any):
    if obj is None:
        return None
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        return cls(**obj)
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(obj)}")


def attack_experiment_config_from_dict(raw: dict) -> AttackExperimentConfig:
    """Build :class:`AttackExperimentConfig` from a plain YAML-loaded dict."""
    data = _maybe_dataclass(DataConfig, raw.get("data")) or DataConfig()
    train = _maybe_dataclass(TrainConfig, raw.get("train")) or TrainConfig()
    nuswide = _maybe_dataclass(NUSWIDEConfig, raw.get("nuswide"))
    swap = _maybe_dataclass(AttackSwapConfig, raw.get("swap")) or AttackSwapConfig()

    return AttackExperimentConfig(
        dataset=str(raw.get("dataset")),
        k_clients=int(raw.get("k_clients")),
        seed=int(raw.get("seed", 0)),
        data=data,
        train=train,
        nuswide=nuswide,
        run_name=raw.get("run_name"),
        swap=swap,
        use_split_lenet=bool(raw.get("use_split_lenet", False)),
        stealth_oracle_labels=bool(raw.get("stealth_oracle_labels", True)),
        bank_attack_split=str(raw.get("bank_attack_split", "balanced")).strip().lower(),
        bank_attack_share=float(raw.get("bank_attack_share", 0.75)),
        bank_attack_model=str(raw.get("bank_attack_model", "compact")).strip().lower(),
        har_attack_split=(
            None
            if raw.get("har_attack_split") is None
            else str(raw.get("har_attack_split")).strip().lower()
        ),
        har_attack_share=(
            float(raw["har_attack_share"])
            if raw.get("har_attack_share") is not None
            else None
        ),
        har_attack_model=str(raw.get("har_attack_model", "compact")).strip().lower(),
        har_aux_labeled_frac=float(raw.get("har_aux_labeled_frac", 0.03)),
    )


def load_attack_config(path: str) -> AttackExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return attack_experiment_config_from_dict(raw)


def dump_attack_config_yaml(path: str, cfg: AttackExperimentConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def as_clean_experiment_config(cfg: AttackExperimentConfig) -> ExperimentConfig:
    """
    Drop swap fields and convert to `ExperimentConfig` so we can reuse shared build/train code.
    """
    return ExperimentConfig(
        dataset=cfg.dataset,
        k_clients=cfg.k_clients,
        seed=cfg.seed,
        data=cfg.data,
        train=cfg.train,
        model=None,
        nuswide=cfg.nuswide,
        run_name=cfg.run_name,
    )

