"""Seleção do backend de inferência a partir da configuração."""

from __future__ import annotations

from triagem.config import Settings
from triagem.inference.base import Classifier
from triagem.inference.sklearn_backend import SklearnClassifier

SUPPORTED_BACKENDS = ("sklearn",)


def load_classifier(settings: Settings) -> Classifier:
    """Instancia o backend indicado por ``settings.model_backend``."""
    if settings.model_backend == "sklearn":
        return SklearnClassifier.load(settings.model_path)

    raise ValueError(
        f"backend não suportado: {settings.model_backend!r}. "
        f"Disponíveis: {', '.join(SUPPORTED_BACKENDS)}"
    )
