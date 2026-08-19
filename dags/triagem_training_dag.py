"""Pipeline de treino e retreino do classificador de laudos.

A DAG é deliberadamente fina: cada tarefa delega para uma função de
``triagem.pipeline.steps``, que é testada de forma isolada na suíte do projeto. O que se
declara aqui é a topologia — ordem, agendamento e política de retentativa — não a lógica.

Encadeamento: ingestão -> preparo -> treino dos candidatos -> seleção e avaliação ->
publicação com gate de qualidade -> exportação ONNX. A avaliação do incumbente (modelo hoje
em produção) roda em paralelo, pois só depende da ingestão, não do treino dos candidatos.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

from triagem.pipeline.steps import (
    ModelNotPromoted,
    evaluate_incumbent,
    export_onnx,
    ingest,
    prepare,
    promote,
    select_and_evaluate,
    train_candidates,
)

DATA_DIR = Path(os.environ.get("TRIAGEM_DATA_DIR", "/opt/airflow/data"))
MODELS_DIR = Path(os.environ.get("TRIAGEM_MODELS_DIR", "/opt/airflow/models"))
METRICS_DIR = Path(os.environ.get("TRIAGEM_METRICS_DIR", "/opt/airflow/metrics"))

MIN_F1_MACRO = float(os.environ.get("TRIAGEM_MIN_F1_MACRO", "0.53"))
MIN_PRIORITY_RECALL_ALTA = float(os.environ.get("TRIAGEM_MIN_PRIORITY_RECALL_ALTA", "0.72"))
RANDOM_SEED = int(os.environ.get("TRIAGEM_RANDOM_SEED", "42"))
VALIDATION_SIZE = float(os.environ.get("TRIAGEM_VALIDATION_SIZE", "0.2"))
MODEL_VERSION = os.environ.get("TRIAGEM_MODEL_VERSION", "1.0.0")


@dag(
    dag_id="triagem_training",
    description="Treina, avalia e promove o classificador de triagem de laudos",
    # Retreino semanal: o corpus é estático, então a cadência existe para exercitar o
    # caminho de retreino, não para perseguir dados novos.
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "treino", "triagem"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
)
def triagem_training():
    @task
    def ingestao() -> dict[str, str]:
        """Baixa o corpus público, reaproveitando o que já estiver em disco."""
        return ingest(DATA_DIR / "raw")

    @task
    def preparo(corpus: dict[str, str]) -> dict[str, str]:
        """Valida o schema e separa a partição de validação, estratificada."""
        return prepare(
            corpus["train"],
            DATA_DIR / "interim",
            validation_size=VALIDATION_SIZE,
            seed=RANDOM_SEED,
        )

    @task
    def treino(particoes: dict[str, str]) -> dict[str, float]:
        """Treina os candidatos e devolve o f1-macro de validação de cada um."""
        return train_candidates(
            particoes["train"],
            particoes["validation"],
            MODELS_DIR / "candidates",
            seed=RANDOM_SEED,
            version=MODEL_VERSION,
        )

    @task
    def selecao(scores: dict[str, float], corpus: dict[str, str]) -> dict[str, object]:
        """Escolhe o campeão pela validação e mede o desempenho no teste."""
        return select_and_evaluate(scores, MODELS_DIR / "candidates", corpus["test"], METRICS_DIR)

    @task
    def avaliar_incumbente(corpus: dict[str, str]) -> dict[str, float] | None:
        """Mede o desempenho do modelo hoje em produção — baseline da decisão de promoção.

        Independente do treino dos candidatos: só depende do corpus, então roda em
        paralelo com ``treino``/``selecao`` em vez de esperá-los.
        """
        return evaluate_incumbent(MODELS_DIR / "model.joblib", corpus["test"])

    @task
    def publicacao(resumo: dict[str, object], incumbente: dict[str, float] | None) -> str:
        """Promove o campeão e registra o run em ``metrics/training_history.jsonl``.

        Os dois desfechos negativos são deliberadamente distintos na interface do Airflow:

        - **falha** (``QualityGateError``) se o candidato violar um piso absoluto — o modelo
          é inutilizável e alguém precisa investigar;
        - **skip** (``AirflowSkipException``) se ele apenas não superar o incumbente — o
          resultado esperado de todo retreino sobre um corpus que não mudou.

        Sem essa distinção o retreino semanal marcaria o run como falho toda semana, e um
        alarme que dispara sempre não é um alarme.
        """
        from airflow.exceptions import AirflowSkipException

        try:
            return promote(
                resumo,
                incumbente,
                MODELS_DIR / "model.joblib",
                METRICS_DIR,
                min_f1_macro=MIN_F1_MACRO,
                min_priority_recall_alta=MIN_PRIORITY_RECALL_ALTA,
            )
        except ModelNotPromoted as motivo:
            raise AirflowSkipException(str(motivo)) from motivo

    @task
    def exportacao(published_path: str) -> dict[str, str]:
        """Converte o modelo recém-promovido para ONNX, com a variante INT8.

        Depende de ``publicacao`` pelo valor, não só pela ordem: recebe o caminho que ela
        devolve. Assim, quando a promoção é pulada por não haver nada a promover, o Airflow
        pula esta tarefa junto — e o ``.onnx`` publicado continua correspondendo ao
        ``model.joblib`` que está servindo, em vez de ser reescrito a partir de um candidato
        que nunca entrou no ar.
        """
        return export_onnx(
            published_path,
            MODELS_DIR / "model.onnx",
            MODELS_DIR / "model.int8.onnx",
        )

    corpus = ingestao()
    particoes = preparo(corpus)
    scores = treino(particoes)
    resumo = selecao(scores, corpus)
    incumbente = avaliar_incumbente(corpus)
    exportacao(publicacao(resumo, incumbente))


triagem_training()
