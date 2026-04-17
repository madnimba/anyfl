from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import yaml

from vfl.data.types import DataConfig, NUSWIDEConfig
from vfl.models.registry import ModelConfig


@dataclass(frozen=True)
class ClusteringTrainConfig:
    """Hyperparameters for contrastive pretrain + SupCon + GMM/KMeans."""

    pretrain_epochs: int = 40
    supcon_epochs: int = 20
    batch_size: int = 256
    lr_pretrain: float = 1e-3
    lr_supcon: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.2
    supcon_temperature: float = 0.17
    backend: Literal["gmm", "kmeans"] = "gmm"
    gmm_covariance: Literal["full", "diag"] = "diag"
    augment_cifar_style: bool = False
    device: str = "cuda"
    # Optional projector for contrastive (same dim as emb if 0 = disabled)
    proj_dim: int = 0
    # CIFAR-10/100 client-0 RGB pipeline (run_clustering_cifar_custom); ignored elsewhere
    simclr_pretrain_epochs: Optional[int] = None  # None = built-in default per dataset; 0 = skip SimCLR
    cifar_linear_probe_epochs: int = 25
    rgb_encoder_width: int = 0   # 0 = built-in default per dataset
    rgb_feat_dim: int = 0        # 0 = built-in default per dataset
    fixmatch_tau: float = 0.0    # 0 = built-in default per dataset
    fixmatch_mu: int = 0         # 0 = use 7
    gmm_merge_n_components: int = 0  # 0 = auto (min(2*K, 120))


@dataclass(frozen=True)
class ClusteringExperimentConfig:
    dataset: str
    k_clients: int
    seed: int = 0
    aux_labeled_frac: float = 0.03
    data: DataConfig = DataConfig()
    clustering: ClusteringTrainConfig = ClusteringTrainConfig()
    model: Optional[ModelConfig] = None
    nuswide: Optional[NUSWIDEConfig] = None
    out_base: str = "experiments/clustering/runs"
    export_cluster_dir: str = "./clusters"
    run_name: Optional[str] = None


def _maybe_dataclass(cls, obj: Any):
    if obj is None:
        return None
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        return cls(**obj)
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(obj)}")


def load_clustering_config(path: str) -> ClusteringExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data = _maybe_dataclass(DataConfig, raw.get("data")) or DataConfig()
    clustering = _maybe_dataclass(ClusteringTrainConfig, raw.get("clustering")) or ClusteringTrainConfig()
    nuswide = _maybe_dataclass(NUSWIDEConfig, raw.get("nuswide"))

    model = None
    model_raw = raw.get("model")
    if model_raw is not None:
        from vfl.models.encoders import EncoderSpec
        from vfl.models.server import ServerSpec

        enc = _maybe_dataclass(EncoderSpec, model_raw.get("encoder"))
        srv = _maybe_dataclass(ServerSpec, model_raw.get("server")) or ServerSpec()
        model = ModelConfig(
            dataset=str(model_raw.get("dataset", raw.get("dataset"))),
            task=str(model_raw.get("task", "multiclass")),
            k_clients=int(model_raw.get("k_clients", raw.get("k_clients"))),
            encoder=enc,
            server=srv,
        )

    return ClusteringExperimentConfig(
        dataset=str(raw.get("dataset")),
        k_clients=int(raw.get("k_clients")),
        seed=int(raw.get("seed", 0)),
        aux_labeled_frac=float(raw.get("aux_labeled_frac", 0.03)),
        data=data,
        clustering=clustering,
        model=model,
        nuswide=nuswide,
        out_base=str(raw.get("out_base", "experiments/clustering/runs")),
        export_cluster_dir=str(raw.get("export_cluster_dir", "./clusters")),
        run_name=raw.get("run_name"),
    )


def dump_clustering_config_yaml(path: str, cfg: ClusteringExperimentConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
