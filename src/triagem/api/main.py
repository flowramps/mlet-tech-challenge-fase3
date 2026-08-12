"""API REST de triagem de laudos médicos."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

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
        decorrido_ms = (time.perf_counter() - inicio) * 1_000

        return PredictResponse(
            condition=predicao.condition,
            confidence=predicao.confidence,
            priority=priority_for(predicao.condition),
            model_version=engine.version,
            backend=engine.name,
            inference_ms=round(decorrido_ms, 3),
        )

    return application


app = create_app()
