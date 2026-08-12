"""Treino do classificador de laudos.

TF-IDF com bigramas alimenta uma Random Forest. A floresta é intencionalmente contida
(``min_samples_leaf=3``, 5.000 features) por dois motivos: manter o artefato pequeno o
bastante para caber na imagem de inferência, e limitar o número de nós, que é o que
determina o tamanho do grafo exportado para ONNX mais adiante.

``class_weight="balanced_subsample"`` compensa o desbalanceamento de 3,2x entre a classe
mais e a menos frequente do corpus.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MAX_FEATURES = 5_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 3


def build_pipeline(seed: int = 42) -> Pipeline:
    """Monta o pipeline de vetorização e classificação."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    min_df=MIN_DF,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_model(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    seed: int = 42,
) -> Pipeline:
    """Treina o pipeline sobre os textos e rótulos fornecidos."""
    pipeline = build_pipeline(seed=seed)
    pipeline.fit(list(texts), list(labels))
    return pipeline


def save_model(
    pipeline: Pipeline,
    labels_by_id: dict[int, str],
    version: str,
    destination: Path,
) -> Path:
    """Serializa o pipeline junto dos nomes de rótulo e da versão."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "pipeline": pipeline,
        "labels": labels_by_id,
        "version": version,
    }
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: Path) -> dict[str, Any]:
    """Lê o bundle serializado por :func:`save_model`."""
    return joblib.load(path)


def main() -> None:
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

    pipeline = train_model(treino[TEXT_COLUMN], treino[LABEL_COLUMN], seed=settings.random_seed)
    destino = save_model(pipeline, CONDITION_NAMES, settings.model_version, settings.model_path)

    tamanho_mb = destino.stat().st_size / 1_048_576
    logger.info("modelo salvo em %s (%.1f MB)", destino, tamanho_mb)

    settings.metrics_dir.mkdir(parents=True, exist_ok=True)
    (settings.metrics_dir / "model_size.json").write_text(
        json.dumps({"path": str(destino), "size_mb": round(tamanho_mb, 2)}, indent=2)
    )


if __name__ == "__main__":
    main()
