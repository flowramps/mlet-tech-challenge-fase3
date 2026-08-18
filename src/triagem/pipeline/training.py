"""Execução local do pipeline de treino.

Compõe as mesmas etapas que a DAG do Airflow encadeia. Um único caminho de código para os
dois modos de execução, então não existe o risco de o treino local e o orquestrado
divergirem com o tempo.
"""

from __future__ import annotations

import logging

from triagem.config import Settings, get_settings
from triagem.pipeline.steps import (
    evaluate_incumbent,
    ingest,
    prepare,
    promote,
    select_and_evaluate,
    train_candidates,
)

logger = logging.getLogger(__name__)


def run_pipeline(settings: Settings) -> dict[str, object]:
    """Executa ingestão, preparo, treino, seleção e publicação em sequência."""
    corpus = ingest(settings.data_dir / "raw")

    particoes = prepare(
        corpus["train"],
        settings.data_dir / "interim",
        validation_size=settings.validation_size,
        seed=settings.random_seed,
    )

    scores = train_candidates(
        particoes["train"],
        particoes["validation"],
        settings.models_dir / "candidates",
        seed=settings.random_seed,
        version=settings.model_version,
    )

    resumo = select_and_evaluate(
        scores,
        settings.models_dir / "candidates",
        corpus["test"],
        settings.metrics_dir,
    )

    incumbente = evaluate_incumbent(settings.model_path, corpus["test"])

    publicado = promote(
        resumo,
        incumbente,
        settings.model_path,
        settings.metrics_dir,
        min_f1_macro=settings.min_f1_macro,
        min_priority_recall_alta=settings.min_priority_recall_alta,
    )

    return {**resumo, "published_path": publicado, "baseline": incumbente}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resumo = run_pipeline(get_settings())
    logger.info("pipeline concluído: %s", resumo)


if __name__ == "__main__":
    main()
