"""Seleção do backend de inferência a partir da configuração."""

from __future__ import annotations

from triagem.config import Settings
from triagem.inference.base import Classifier
from triagem.inference.sklearn_backend import SklearnClassifier

SUPPORTED_BACKENDS = ("sklearn", "onnx")


def load_classifier(settings: Settings) -> Classifier:
    """Instancia o backend indicado por ``settings.model_backend``.

    O import do backend ONNX é local para que o ``onnxruntime`` só seja carregado por quem
    de fato o usa — quem serve o backend scikit-learn não paga o tempo de importação dele
    na subida do processo.
    """
    if settings.model_backend == "sklearn":
        return SklearnClassifier.load(settings.model_path)

    if settings.model_backend == "onnx":
        from triagem.inference.onnx_backend import OnnxClassifier

        return OnnxClassifier.load(settings.onnx_path)

    raise ValueError(
        f"backend não suportado: {settings.model_backend!r}. "
        f"Disponíveis: {', '.join(SUPPORTED_BACKENDS)}"
    )
