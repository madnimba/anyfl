"""Defense modules (RGAR + VFL model adapters)."""

from vfl.defense.rgar_embedding_fusion import train_rgar_embedding_fusion
from vfl.defense.rgar_flat_vfl import train_rgar_flatten_vfl
from vfl.defense.rgar_tabular_mlp import train_rgar_tabular_mlp

__all__ = [
    "train_rgar_embedding_fusion",
    "train_rgar_flatten_vfl",
    "train_rgar_tabular_mlp",
]
