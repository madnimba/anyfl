from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import yaml

from vfl.data.types import DataConfig, NUSWIDEConfig
from vfl.models.encoders import EncoderSpec
from vfl.models.registry import ModelConfig
from vfl.models.server import ServerSpec
from vfl.train.loop import TrainConfig


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    k_clients: int
    seed: int = 0
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: Optional[ModelConfig] = None
    nuswide: Optional[NUSWIDEConfig] = None
    run_name: Optional[str] = None


def _maybe_dataclass(cls, obj: Any):
    if obj is None:
        return None
    if isinstance(obj, cls):
        return obj
    if isinstance(obj, dict):
        return cls(**obj)
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(obj)}")


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data = _maybe_dataclass(DataConfig, raw.get("data")) or DataConfig()
    train = _maybe_dataclass(TrainConfig, raw.get("train")) or TrainConfig()
    nuswide = _maybe_dataclass(NUSWIDEConfig, raw.get("nuswide"))

    # model config can be explicit or omitted (defaults computed later)
    model_raw = raw.get("model")
    model = None
    if model_raw is not None:
        enc = _maybe_dataclass(EncoderSpec, model_raw.get("encoder"))
        srv = _maybe_dataclass(ServerSpec, model_raw.get("server")) or ServerSpec()
        model = ModelConfig(
            dataset=str(model_raw.get("dataset", raw.get("dataset"))),
            task=str(model_raw.get("task", "multiclass")),
            k_clients=int(model_raw.get("k_clients", raw.get("k_clients"))),
            encoder=enc,
            server=srv,
        )

    return ExperimentConfig(
        dataset=str(raw.get("dataset")),
        k_clients=int(raw.get("k_clients")),
        seed=int(raw.get("seed", 0)),
        data=data,
        train=train,
        model=model,
        nuswide=nuswide,
        run_name=raw.get("run_name"),
    )


def dump_config_yaml(path: str, cfg: ExperimentConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

