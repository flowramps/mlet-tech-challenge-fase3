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

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from triagem.data.download import download_dataset
from triagem.data.prepare import (
    CONDITION_NAMES,
    LABEL_COLUMN,
    TEXT_COLUMN,
    load_split,
    split_train_validation,
)
from triagem.model.evaluate import evaluate_model, priority_recall, save_metrics
from triagem.model.train import (
    MODEL_TYPES,
    load_bundle,
    save_model,
    select_champion,
    train_model,
)

logger = logging.getLogger(__name__)


class QualityGateError(RuntimeError):
    """Modelo recém-treinado reprovado num piso absoluto de qualidade.

    Sinaliza que algo está errado com o treino: o candidato não é utilizável. O run do
    pipeline deve **falhar** — alguém precisa investigar.
    """


# O sufixo `Error` que a N818 pede é justamente o que esta classe não pode ter: ela sinaliza
# um desfecho normal do pipeline, e o nome é metade da distinção que o resto do módulo faz
# questão de manter. Regra dispensada aqui de propósito, não por descuido.
class ModelNotPromoted(RuntimeError):  # noqa: N818
    """Candidato utilizável, porém não melhor que o modelo já em produção.

    Deliberadamente **não** é subclasse de :class:`QualityGateError`: é um desfecho normal
    do pipeline, não um defeito. Sobre um corpus estático o retreino reproduz o incumbente,
    então esta é a saída esperada de toda execução periódica depois da primeira — tratá-la
    como falha deixaria a DAG semanal vermelha para sempre e faria o alarme perder o sentido.
    A orquestração converte isto em ``skip``.
    """


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
    predicoes = bundle["pipeline"].predict(teste[TEXT_COLUMN])
    metricas = evaluate_model(
        bundle["pipeline"], teste[TEXT_COLUMN], teste[LABEL_COLUMN], CONDITION_NAMES
    )
    recall_alta = priority_recall(teste[LABEL_COLUMN], predicoes, CONDITION_NAMES, priority="alta")
    save_metrics(
        {**metricas, "priority_recall_alta": recall_alta}, destino_metricas / "metrics.json"
    )

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
        "campeão=%s f1_macro(teste)=%.4f accuracy(teste)=%.4f priority_recall_alta(teste)=%.4f",
        campeao,
        metricas["f1_macro"],
        metricas["accuracy"],
        recall_alta,
    )

    return {
        "champion": campeao,
        "f1_macro": float(metricas["f1_macro"]),
        "accuracy": float(metricas["accuracy"]),
        "priority_recall_alta": float(recall_alta),
        "candidate_path": str(caminho_campeao),
    }


def evaluate_incumbent(model_path: Path | str, test_csv: str) -> dict[str, float] | None:
    """Mede o desempenho do modelo atualmente publicado, como baseline da promoção.

    ``None`` sem modelo publicado ainda — é o caso de bootstrap, sem incumbente para
    comparar. Reavalia o incumbente do zero em vez de ler ``metrics/metrics.json``: esse
    arquivo é sobrescrito por :func:`select_and_evaluate` com os números do *candidato*
    antes da decisão de promoção, então não serve de registro do que está em produção.
    """
    caminho = Path(model_path)
    if not caminho.exists():
        return None

    bundle = load_bundle(caminho)
    teste = load_split(Path(test_csv))
    predicoes = bundle["pipeline"].predict(teste[TEXT_COLUMN])
    metricas = evaluate_model(
        bundle["pipeline"], teste[TEXT_COLUMN], teste[LABEL_COLUMN], CONDITION_NAMES
    )
    recall_alta = priority_recall(teste[LABEL_COLUMN], predicoes, CONDITION_NAMES, priority="alta")
    return {"f1_macro": float(metricas["f1_macro"]), "priority_recall_alta": float(recall_alta)}


def _floor_failures(
    candidate_f1_macro: float,
    candidate_priority_recall_alta: float,
    *,
    min_f1_macro: float,
    min_priority_recall_alta: float,
) -> list[str]:
    """Pisos absolutos violados — cada um significa candidato inutilizável."""
    motivos: list[str] = []
    if candidate_f1_macro < min_f1_macro:
        motivos.append(f"f1_macro {candidate_f1_macro:.4f} abaixo do mínimo {min_f1_macro:.4f}")
    if candidate_priority_recall_alta < min_priority_recall_alta:
        motivos.append(
            f"recall de prioridade alta {candidate_priority_recall_alta:.4f} abaixo do "
            f"mínimo {min_priority_recall_alta:.4f}"
        )
    return motivos


def _regression_failures(
    candidate_f1_macro: float,
    candidate_priority_recall_alta: float,
    baseline_f1_macro: float | None,
    baseline_priority_recall_alta: float | None,
) -> list[str]:
    """Critérios de não regressão contra o incumbente — candidato bom, só não melhor.

    Vazia quando não há incumbente: no primeiro treino só os pisos absolutos se aplicam.
    """
    if baseline_f1_macro is None or baseline_priority_recall_alta is None:
        return []

    motivos: list[str] = []
    if candidate_f1_macro <= baseline_f1_macro:
        motivos.append(
            f"f1_macro {candidate_f1_macro:.4f} não supera o modelo em produção "
            f"({baseline_f1_macro:.4f})"
        )
    if candidate_priority_recall_alta < baseline_priority_recall_alta:
        motivos.append(
            f"recall de prioridade alta {candidate_priority_recall_alta:.4f} regride em "
            f"relação ao modelo em produção ({baseline_priority_recall_alta:.4f})"
        )
    return motivos


def _gate_failures(
    candidate_f1_macro: float,
    candidate_priority_recall_alta: float,
    baseline_f1_macro: float | None,
    baseline_priority_recall_alta: float | None,
    *,
    min_f1_macro: float,
    min_priority_recall_alta: float,
) -> list[str]:
    """Todos os critérios do gate que o candidato não cumpre — vazia se ele deve ser promovido.

    Não distingue a severidade dos motivos; para isso existem :func:`_floor_failures` e
    :func:`_regression_failures`. Serve a quem só precisa da decisão final: o predicado
    :func:`should_promote` e o registro no histórico.
    """
    return _floor_failures(
        candidate_f1_macro,
        candidate_priority_recall_alta,
        min_f1_macro=min_f1_macro,
        min_priority_recall_alta=min_priority_recall_alta,
    ) + _regression_failures(
        candidate_f1_macro,
        candidate_priority_recall_alta,
        baseline_f1_macro,
        baseline_priority_recall_alta,
    )


def should_promote(
    candidate_f1_macro: float,
    candidate_priority_recall_alta: float,
    baseline_f1_macro: float | None,
    baseline_priority_recall_alta: float | None,
    *,
    min_f1_macro: float,
    min_priority_recall_alta: float,
) -> bool:
    """Decide se o candidato deve substituir o modelo em produção.

    Dois pisos absolutos — f1-macro e recall da prioridade "alta" — e, quando já existe um
    incumbente, dois critérios de não regressão: superar seu f1-macro estritamente, e não
    piorar o recall de prioridade alta. O recall de prioridade alta é a métrica de negócio da
    triagem: rebaixar um caso realmente urgente (cardiovascular/nervous system) pesa mais do
    que confundir duas condições que já dariam na mesma prioridade — f1-macro sozinho não
    enxerga essa assimetria.
    """
    return not _gate_failures(
        candidate_f1_macro,
        candidate_priority_recall_alta,
        baseline_f1_macro,
        baseline_priority_recall_alta,
        min_f1_macro=min_f1_macro,
        min_priority_recall_alta=min_priority_recall_alta,
    )


def publish(
    candidate_path: str,
    model_path: Path | str,
    f1_macro: float,
    priority_recall_alta: float,
    *,
    min_f1_macro: float,
    min_priority_recall_alta: float,
    baseline_f1_macro: float | None,
    baseline_priority_recall_alta: float | None,
) -> str:
    """Promove o campeão a modelo servido, se ele passar no gate.

    Em qualquer desfecho negativo o artefato publicado **não é tocado**: um retreino ruim não
    derruba o modelo que já está atendendo. O que muda é a severidade sinalizada:

    - :class:`QualityGateError` se um piso absoluto foi violado — o candidato é inutilizável
      e o run deve falhar.
    - :class:`ModelNotPromoted` se ele só não superou o incumbente — desfecho normal.

    O piso tem precedência: um candidato que viola os dois é um defeito, não um empate.
    """
    pisos = _floor_failures(
        f1_macro,
        priority_recall_alta,
        min_f1_macro=min_f1_macro,
        min_priority_recall_alta=min_priority_recall_alta,
    )
    if pisos:
        raise QualityGateError(
            "; ".join(pisos) + " — modelo não promovido, o anterior segue em uso"
        )

    regressoes = _regression_failures(
        f1_macro, priority_recall_alta, baseline_f1_macro, baseline_priority_recall_alta
    )
    if regressoes:
        raise ModelNotPromoted(
            "; ".join(regressoes) + " — modelo não promovido, o anterior segue em uso"
        )

    destino = Path(model_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, destino)

    logger.info(
        "modelo promovido: %s (f1_macro=%.4f, priority_recall_alta=%.4f)",
        destino,
        f1_macro,
        priority_recall_alta,
    )
    return str(destino)


def _append_history(
    history_path: Path | str,
    *,
    champion: str,
    f1_macro: float,
    priority_recall_alta: float,
    baseline_f1_macro: float | None,
    baseline_priority_recall_alta: float | None,
    promoted: bool,
    rejection_reasons: list[str],
) -> None:
    """Acrescenta uma linha ao histórico de execuções do pipeline, em JSON Lines.

    Uma linha por execução, nunca reescrita: cada run só precisa acrescentar, nunca reler o
    que já existe, e uma escrita interrompida no meio derruba só a última linha, não o
    histórico inteiro — ao contrário de reescrever um único JSON a cada run.
    """
    caminho = Path(history_path)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    entrada = {
        "timestamp": datetime.now(UTC).isoformat(),
        "champion": champion,
        "f1_macro": round(f1_macro, 4),
        "priority_recall_alta": round(priority_recall_alta, 4),
        "baseline_f1_macro": (
            round(baseline_f1_macro, 4) if baseline_f1_macro is not None else None
        ),
        "baseline_priority_recall_alta": (
            round(baseline_priority_recall_alta, 4)
            if baseline_priority_recall_alta is not None
            else None
        ),
        "promoted": promoted,
        "rejection_reasons": rejection_reasons,
    }
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")


def promote(
    resumo: dict[str, object],
    incumbente: dict[str, float] | None,
    model_path: Path | str,
    metrics_dir: Path | str,
    *,
    min_f1_macro: float,
    min_priority_recall_alta: float,
) -> str:
    """Decide a promoção do campeão (via :func:`publish`) e registra o resultado — promovido
    ou não, e por quê — em ``metrics_dir/training_history.jsonl``.

    Ponto de entrada recomendado para as orquestrações (execução local e DAG): chamar
    :func:`publish` diretamente decide a promoção mas deixa o run de fora do histórico.
    """
    f1_macro = float(resumo["f1_macro"])
    priority_recall_alta = float(resumo["priority_recall_alta"])
    baseline_f1_macro = incumbente["f1_macro"] if incumbente else None
    baseline_priority_recall_alta = incumbente["priority_recall_alta"] if incumbente else None

    motivos = _gate_failures(
        f1_macro,
        priority_recall_alta,
        baseline_f1_macro,
        baseline_priority_recall_alta,
        min_f1_macro=min_f1_macro,
        min_priority_recall_alta=min_priority_recall_alta,
    )

    _append_history(
        Path(metrics_dir) / "training_history.jsonl",
        champion=str(resumo["champion"]),
        f1_macro=f1_macro,
        priority_recall_alta=priority_recall_alta,
        baseline_f1_macro=baseline_f1_macro,
        baseline_priority_recall_alta=baseline_priority_recall_alta,
        promoted=not motivos,
        rejection_reasons=motivos,
    )

    return publish(
        str(resumo["candidate_path"]),
        model_path,
        f1_macro,
        priority_recall_alta,
        min_f1_macro=min_f1_macro,
        min_priority_recall_alta=min_priority_recall_alta,
        baseline_f1_macro=baseline_f1_macro,
        baseline_priority_recall_alta=baseline_priority_recall_alta,
    )
