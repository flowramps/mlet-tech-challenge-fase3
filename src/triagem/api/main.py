"""API REST de triagem de laudos médicos."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_client import CollectorRegistry

from triagem.api.metrics import instrument
from triagem.api.schemas import HealthResponse, PredictRequest, PredictResponse
from triagem.config import get_settings
from triagem.inference.base import Classifier
from triagem.inference.factory import load_classifier
from triagem.inference.priority import priority_for

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _configure_logging() -> None:
    """Instala um handler de raiz, se ainda não houver.

    O uvicorn só configura os loggers dele, então mensagens da aplicação cairiam num
    root sem handler e sumiriam. Como a API é o processo raiz, configurar o logging é
    responsabilidade dela.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def create_app(classifier: Classifier | None = None) -> FastAPI:
    """Monta a aplicação.

    Recebendo um ``classifier`` pronto, a API fica testável sem tocar em disco; sem ele,
    o modelo é carregado uma única vez no startup, e não a cada requisição.
    """
    _configure_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.classifier = (
            classifier if classifier is not None else load_classifier(get_settings())
        )
        # O mesmo registro do log vai para o /metrics: no Grafana dá para saber qual
        # modelo respondia em cada janela de tempo sem sair do dashboard.
        metricas.model_info.info(
            {
                "version": application.state.classifier.version,
                "backend": application.state.classifier.name,
            }
        )
        logger.info(
            "modelo carregado: backend=%s versão=%s",
            application.state.classifier.name,
            application.state.classifier.version,
        )
        yield

    application = FastAPI(
        title="API de Triagem de Laudos Médicos",
        description=(
            "Classifica o texto de um laudo em uma condição clínica e devolve a "
            "prioridade de atendimento correspondente."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Registry próprio por aplicação, em vez do global da biblioteca: a suíte de testes
    # cria dezenas de apps no mesmo processo, e no registry global a segunda criação
    # colidiria com a primeira — além de uma vazar contadores para dentro da outra.
    metricas = instrument(application, CollectorRegistry())
    application.state.metrics = metricas

    @application.get("/health", response_model=HealthResponse, tags=["operação"])
    def health(request: Request) -> HealthResponse:
        """Liveness do serviço e identificação do modelo em uso."""
        engine: Classifier = request.app.state.classifier
        return HealthResponse(status="ok", model_version=engine.version, backend=engine.name)

    @application.post("/predict", response_model=PredictResponse, tags=["triagem"])
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        """Classifica um laudo e devolve condição, confiança e prioridade."""
        engine: Classifier = request.app.state.classifier

        inicio = time.perf_counter()
        predicao = engine.predict([payload.text])[0]
        decorrido = time.perf_counter() - inicio
        decorrido_ms = decorrido * 1_000

        # A mesma medição vai para o histograma rotulado por backend: é o que permite
        # comparar motores de inferência em produção, não só no benchmark local.
        metricas.inference_duration.labels(backend=engine.name).observe(decorrido)

        prioridade = priority_for(predicao.condition)

        # Saúde do modelo, não do serviço: a distribuição do que ele prevê e com quanta
        # confiança é o que denuncia drift enquanto o HTTP segue respondendo 200.
        metricas.predictions_total.labels(
            condition=predicao.condition, priority=prioridade
        ).inc()
        metricas.prediction_confidence.observe(predicao.confidence)

        return PredictResponse(
            condition=predicao.condition,
            confidence=predicao.confidence,
            priority=prioridade,
            model_version=engine.version,
            backend=engine.name,
            inference_ms=round(decorrido_ms, 3),
        )

    return application


app = create_app()
