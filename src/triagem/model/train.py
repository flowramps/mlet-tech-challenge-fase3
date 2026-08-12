"""Treino e seleção do classificador de laudos.

Dois candidatos disputam a vaga de modelo servido, sobre a **mesma** vetorização TF-IDF —
só o classificador muda, o que isola a variável em comparação:

- ``random_forest``: ensemble de árvores, robusto a interações não lineares.
- ``logistic_regression``: modelo linear, o padrão histórico para texto esparso.

O campeão é escolhido pelo f1-macro na partição de validação. O conjunto de teste não
participa da seleção — ele é gasto uma única vez, em ``triagem.model.evaluate``, para
reportar o desempenho final sem contaminação.

``class_weight`` balanceado compensa o desbalanceamento de 3,2x entre a classe mais e a
menos frequente do corpus, e é por isso que a métrica de decisão é f1-macro e não acurácia.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MAX_FEATURES = 5_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 3
MAX_ITER = 1_000

MODEL_TYPES: tuple[str, ...] = ("random_forest", "logistic_regression")


def _build_vectorizer() -> TfidfVectorizer:
    """Vetorização compartilhada pelos candidatos."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def _build_classifier(model_type: str, seed: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=MAX_ITER,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"modelo desconhecido: {model_type!r}. Disponíveis: {', '.join(MODEL_TYPES)}")


def build_pipeline(model_type: str, seed: int = 42) -> Pipeline:
    """Monta o pipeline de vetorização e classificação para um candidato."""
    return Pipeline(
        [
            ("tfidf", _build_vectorizer()),
            ("clf", _build_classifier(model_type, seed)),
        ]
    )


def train_model(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    model_type: str,
    seed: int = 42,
) -> Pipeline:
    """Treina um candidato sobre os textos e rótulos fornecidos."""
    pipeline = build_pipeline(model_type, seed=seed)
    pipeline.fit(list(texts), list(labels))
    return pipeline


def select_champion(scores: Mapping[str, float]) -> str:
    """Devolve o candidato de maior score.

    Empates são resolvidos pela ordem de ``MODEL_TYPES``, para que a seleção seja
    reprodutível em vez de depender da ordem de iteração do dicionário.
    """
    if not scores:
        raise ValueError("nenhum candidato para selecionar")

    return max(scores, key=lambda nome: (scores[nome], -MODEL_TYPES.index(nome)))


def save_model(
    pipeline: Pipeline,
    labels_by_id: dict[int, str],
    *,
    version: str,
    model_type: str,
    destination: Path,
) -> Path:
    """Serializa o pipeline junto dos rótulos, da versão e do tipo de modelo."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "labels": labels_by_id,
        "version": version,
        "model_type": model_type,
    }
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: Path) -> dict[str, Any]:
    """Lê o bundle serializado por :func:`save_model`."""
    return joblib.load(path)


def main() -> None:
    from sklearn.metrics import f1_score

    from triagem.config import get_settings
    from triagem.data.download import download_dataset
    from triagem.data.prepare import (
        CONDITION_NAMES,
        LABEL_COLUMN,
        TEXT_COLUMN,
        load_split,
        split_train_validation,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    paths = download_dataset(settings.data_dir / "raw")
    frame = load_split(paths["train"])
    treino, validacao = split_train_validation(
        frame, validation_size=settings.validation_size, seed=settings.random_seed
    )
    logger.info("treino=%d validação=%d", len(treino), len(validacao))

    candidatos: dict[str, Pipeline] = {}
    scores: dict[str, float] = {}

    for model_type in MODEL_TYPES:
        pipeline = train_model(
            treino[TEXT_COLUMN],
            treino[LABEL_COLUMN],
            model_type=model_type,
            seed=settings.random_seed,
        )
        f1 = float(
            f1_score(
                validacao[LABEL_COLUMN],
                pipeline.predict(validacao[TEXT_COLUMN]),
                average="macro",
            )
        )
        candidatos[model_type] = pipeline
        scores[model_type] = f1
        logger.info("candidato %-20s f1_macro(validação)=%.4f", model_type, f1)

    campeao = select_champion(scores)
    logger.info("campeão: %s", campeao)

    destino = save_model(
        candidatos[campeao],
        CONDITION_NAMES,
        version=settings.model_version,
        model_type=campeao,
        destination=settings.model_path,
    )

    tamanho_mb = destino.stat().st_size / 1_048_576
    logger.info("modelo salvo em %s (%.1f MB)", destino, tamanho_mb)

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    (settings.metrics_dir / "model_selection.json").write_text(
        json.dumps(
            {
                "champion": campeao,
                "selection_metric": "f1_macro",
                "selection_split": "validation",
                "candidates": {nome: round(valor, 4) for nome, valor in scores.items()},
                "artifact_size_mb": round(tamanho_mb, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
