"""Contratos de entrada e saída da API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 20_000


class PredictRequest(BaseModel):
    """Laudo submetido para triagem."""

    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="Texto livre do laudo médico.",
    )


class PredictResponse(BaseModel):
    """Resultado da triagem."""

    model_config = ConfigDict(protected_namespaces=())

    condition: str = Field(description="Condição clínica prevista pelo modelo.")
    confidence: float = Field(description="Confiança do modelo na condição prevista.")
    priority: str = Field(description="Prioridade de atendimento.")
    priority_source: str = Field(
        default="regra_de_negocio",
        description="Origem da prioridade: regra determinística, não predição do modelo.",
    )
    model_version: str = Field(description="Versão do modelo que respondeu.")
    backend: str = Field(description="Motor de inferência em uso.")
    inference_ms: float = Field(description="Tempo de inferência do modelo, em ms.")


class HealthResponse(BaseModel):
    """Estado do serviço e do modelo carregado."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    backend: str
