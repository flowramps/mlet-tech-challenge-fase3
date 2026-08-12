"""Avaliação do classificador.

O f1-macro é a métrica principal: com classes de 1.195 a 3.844 amostras, a acurácia
premia acertar a classe majoritária e esconde falha nas minoritárias.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    texts: Sequence[str],
    labels: Sequence[int],
    labels_by_id: dict[int, str],
) -> dict[str, Any]:
    """Calcula as métricas de classificação sobre uma partição."""
    predicoes = pipeline.predict(list(texts))
    relatorio = classification_report(labels, predicoes, output_dict=True, zero_division=0)

    per_class = {
        labels_by_id[int(chave)]: valor
        for chave, valor in relatorio.items()
        if chave.isdigit() and int(chave) in labels_by_id
    }

    return {
        "accuracy": float(accuracy_score(labels, predicoes)),
        "f1_macro": float(f1_score(labels, predicoes, average="macro")),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predicoes).tolist(),
        "support": len(labels),
    }


def save_metrics(metrics: dict[str, Any], destination: Path) -> Path:
    """Grava as métricas como JSON legível."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return destination


def main() -> None:
    from triagem.config import get_settings
    from triagem.data.download import download_dataset
    from triagem.data.prepare import CONDITION_NAMES, LABEL_COLUMN, TEXT_COLUMN, load_split
    from triagem.model.train import load_bundle

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    bundle = load_bundle(settings.model_path)
    paths = download_dataset(settings.data_dir / "raw")
    teste = load_split(paths["test"])

    metricas = evaluate_model(
        bundle["pipeline"], teste[TEXT_COLUMN], teste[LABEL_COLUMN], CONDITION_NAMES
    )
    save_metrics(metricas, settings.metrics_dir / "metrics.json")

    logger.info(
        "accuracy=%.4f f1_macro=%.4f (n=%d)",
        metricas["accuracy"],
        metricas["f1_macro"],
        metricas["support"],
    )


if __name__ == "__main__":
    main()
