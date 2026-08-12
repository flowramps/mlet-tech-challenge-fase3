"""Etapas do pipeline de treino.

Cada etapa é uma função independente que recebe e devolve apenas tipos serializáveis
(``str``, ``float``, ``dict``). Isso não é preciosismo: é o que permite encadeá-las como
tarefas de Airflow, onde o valor trafega por XCom e um ``Path`` não sobreviveria à
serialização, e ao mesmo tempo testá-las sem Airflow instalado.

Os artefatos intermediários vão para disco em vez de trafegarem entre as tarefas. Um
dataset não cabe em XCom, e materializá-lo permite inspecionar o que cada etapa produziu
quando algo dá errado.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from triagem.data.download import download_dataset
from triagem.data.prepare import (
    CONDITION_NAMES,
    LABEL_COLUMN,
    TEXT_COLUMN,
    load_split,
    split_train_validation,
)
from triagem.model.evaluate import evaluate_model, save_metrics
from triagem.model.train import (
    MODEL_TYPES,
    load_bundle,
    save_model,
    select_champion,
    train_model,
)

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Modelo recém-treinado reprovado no piso de qualidade."""


def ingest(data_dir: Path | str, *, force: bool = False) -> dict[str, str]:
    """Baixa o corpus e devolve os caminhos como texto."""
    paths = download_dataset(Path(data_dir), force=force)
    return {chave: str(valor) for chave, valor in paths.items()}


def prepare(
    train_csv: str,
    interim_dir: Path | str,
    *,
    validation_size: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Valida o schema, separa validação estratificada e materializa as partições."""
    destino = Path(interim_dir)
    destino.mkdir(parents=True, exist_ok=True)

    frame = load_split(Path(train_csv))
    treino, validacao = split_train_validation(frame, validation_size=validation_size, seed=seed)

    caminho_treino = destino / "train.csv"
    caminho_validacao = destino / "validation.csv"
    treino.to_csv(caminho_treino, index=False)
    validacao.to_csv(caminho_validacao, index=False)

    logger.info("partições: treino=%d validação=%d", len(treino), len(validacao))
    return {"train": str(caminho_treino), "validation": str(caminho_validacao)}


def train_candidates(
    train_csv: str,
    validation_csv: str,
    candidates_dir: Path | str,
    *,
    seed: int = 42,
    version: str = "1.0.0",
) -> dict[str, float]:
    """Treina cada candidato e devolve o f1-macro de validação de cada um."""
    from sklearn.metrics import f1_score

    destino = Path(candidates_dir)
    destino.mkdir(parents=True, exist_ok=True)

    treino = load_split(Path(train_csv))
    validacao = load_split(Path(validation_csv))

    scores: dict[str, float] = {}
    for model_type in MODEL_TYPES:
        pipeline = train_model(
            treino[TEXT_COLUMN], treino[LABEL_COLUMN], model_type=model_type, seed=seed
        )
        f1 = float(
            f1_score(
                validacao[LABEL_COLUMN],
                pipeline.predict(validacao[TEXT_COLUMN]),
                average="macro",
            )
        )
        save_model(
            pipeline,
            CONDITION_NAMES,
            version=version,
            model_type=model_type,
            destination=destino / f"{model_type}.joblib",
        )
        scores[model_type] = f1
        logger.info("candidato %-20s f1_macro(validação)=%.4f", model_type, f1)

    return scores


def select_and_evaluate(
    scores: dict[str, float],
    candidates_dir: Path | str,
    test_csv: str,
    metrics_dir: Path | str,
) -> dict[str, object]:
    """Escolhe o campeão pela validação e o avalia no conjunto de teste."""
    origem = Path(candidates_dir)
    destino_metricas = Path(metrics_dir)

    campeao = select_champion(scores)
    caminho_campeao = origem / f"{campeao}.joblib"
    bundle = load_bundle(caminho_campeao)

    teste = load_split(Path(test_csv))
    metricas = evaluate_model(
        bundle["pipeline"], teste[TEXT_COLUMN], teste[LABEL_COLUMN], CONDITION_NAMES
    )
    save_metrics(metricas, destino_metricas / "metrics.json")

    save_metrics(
        {
            "champion": campeao,
            "selection_metric": "f1_macro",
            "selection_split": "validation",
            "candidates": {nome: round(valor, 4) for nome, valor in scores.items()},
        },
        destino_metricas / "model_selection.json",
    )

    logger.info(
        "campeão=%s f1_macro(teste)=%.4f accuracy(teste)=%.4f",
        campeao,
        metricas["f1_macro"],
        metricas["accuracy"],
    )

    return {
        "champion": campeao,
        "f1_macro": float(metricas["f1_macro"]),
        "accuracy": float(metricas["accuracy"]),
        "candidate_path": str(caminho_campeao),
    }


def should_publish(f1_macro: float, minimum: float) -> bool:
    """Decide se o modelo atinge o piso de qualidade."""
    return f1_macro >= minimum


def publish(
    candidate_path: str,
    model_path: Path | str,
    f1_macro: float,
    minimum: float,
) -> str:
    """Promove o campeão a modelo servido, se ele passar no gate.

    Reprovando, levanta :class:`QualityGateError` **sem tocar no artefato publicado**: um
    retreino ruim não pode derrubar o modelo que já está atendendo em produção.
    """
    if not should_publish(f1_macro, minimum):
        raise QualityGateError(
            f"f1_macro {f1_macro:.4f} abaixo do mínimo {minimum:.4f} — "
            "modelo não promovido, o anterior segue em uso"
        )

    destino = Path(model_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, destino)

    logger.info("modelo promovido: %s (f1_macro=%.4f)", destino, f1_macro)
    return str(destino)
