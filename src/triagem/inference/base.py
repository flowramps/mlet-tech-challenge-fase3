"""Contrato de inferência.

A API depende deste Protocol, não de scikit-learn nem de qualquer runtime específico. É o
que permite trocar o motor de inferência por variável de ambiente e comparar latência
entre backends sem alterar uma linha da camada HTTP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """Condição prevista para um laudo, com a confiança do modelo."""

    condition: str
    confidence: float


@runtime_checkable
class Classifier(Protocol):
    """Motor de inferência intercambiável."""

    name: str
    version: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classifica cada texto, preservando a ordem de entrada."""
        ...
