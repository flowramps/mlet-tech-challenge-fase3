"""Backend de inferência baseado no pipeline scikit-learn serializado."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sklearn.pipeline import Pipeline

from triagem.inference.base import Prediction
from triagem.model.train import load_bundle


class SklearnClassifier:
    """Serve o pipeline treinado direto do artefato joblib."""

    name = "sklearn"

    def __init__(
        self,
        pipeline: Pipeline,
        labels_by_id: dict[int, str],
        version: str,
        model_type: str = "desconhecido",
    ) -> None:
        self._pipeline = pipeline
        self._labels_by_id = labels_by_id
        self.version = version
        self.model_type = model_type

    @classmethod
    def load(cls, path: Path) -> SklearnClassifier:
        """Carrega o bundle gravado por ``triagem.model.train.save_model``."""
        bundle = load_bundle(path)
        return cls(
            bundle["pipeline"],
            bundle["labels"],
            bundle["version"],
            bundle.get("model_type", "desconhecido"),
        )

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        entradas = list(texts)
        if not entradas:
            return []

        probabilidades = self._pipeline.predict_proba(entradas)
        classes = self._pipeline.classes_

        predicoes: list[Prediction] = []
        for linha in probabilidades:
            indice = int(linha.argmax())
            predicoes.append(
                Prediction(
                    condition=self._labels_by_id[int(classes[indice])],
                    confidence=float(linha[indice]),
                )
            )
        return predicoes
