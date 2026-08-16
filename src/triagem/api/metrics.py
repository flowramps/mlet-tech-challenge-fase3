"""Instrumentação Prometheus da API.

Duas métricas de tráfego cobrem o essencial de um serviço síncrono: quantas requisições
chegaram (e com qual resultado) e quanto tempo cada uma levou. Throughput e taxa de erro
são derivados delas na consulta — menos estado no processo e nenhuma chance de as séries
divergirem entre si.

A terceira métrica isola a inferência do modelo, sem HTTP, rotulada por backend. É a
mesma separação do benchmark local: ela permite atribuir um ganho de latência ao modelo
ou ao servidor, e comparar backends de inferência em produção.

As demais observam o **modelo**, não o serviço: o que ele prevê e com quanta confiança.
HTTP saudável não diz nada sobre predição saudável — um modelo pode responder rápido e
200 enquanto desliza para uma única classe ou perde confiança. São essas séries que
denunciam drift antes de alguém reclamar da fila de triagem.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    generate_latest,
)

# Os buckets padrão da biblioteca começam em 5 ms: a distribuição inteira desta API
# (p99 ~3 ms) cairia no primeiro bucket e os percentis sairiam sem resolução. Esta régua
# começa em 1 ms e ainda alcança 2,5 s, onde uma degradação real ficaria visível.
HTTP_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

# A inferência pura é ~3x mais rápida que o HTTP ponta a ponta, e um backend otimizado
# deve baixar isso ainda mais — a régua começa em 0,1 ms para a comparação entre
# backends não sair achatada no primeiro bucket.
INFERENCE_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)

# Com 5 classes, a confiança de um chute uniforme é 0,2 — abaixo disso não existe. A
# régua começa ali e sobe em passos de 0,05: fino o bastante para um deslocamento da
# distribuição (drift) aparecer no gráfico antes de virar incidente.
CONFIDENCE_BUCKETS = tuple(round(0.2 + i * 0.05, 2) for i in range(16))


@dataclass(frozen=True)
class Metrics:
    """Instrumentos da aplicação, agrupados para viajar em ``app.state``."""

    requests_total: Counter
    request_duration: Histogram
    inference_duration: Histogram
    predictions_total: Counter
    prediction_confidence: Histogram
    model_info: Info


def instrument(application: FastAPI, registry: CollectorRegistry) -> Metrics:
    """Instala o middleware de medição e a rota ``/metrics``.

    Contagem e duração usam a rota declarada (``/predict``) como rótulo, nunca o path
    bruto: um scanner varrendo URLs aleatórias criaria uma série temporal nova por URL
    e esgotaria a memória do processo. Cardinalidade controlada é o que faz a métrica
    sobreviver exposta à rede.
    """
    metrics = Metrics(
        requests_total=Counter(
            "triagem_http_requests_total",
            "Total de requisições HTTP, por método, rota e status da resposta.",
            labelnames=("method", "route", "status"),
            registry=registry,
        ),
        request_duration=Histogram(
            "triagem_http_request_duration_seconds",
            "Duração da requisição HTTP, do recebimento ao envio da resposta.",
            labelnames=("method", "route"),
            buckets=HTTP_BUCKETS,
            registry=registry,
        ),
        inference_duration=Histogram(
            "triagem_inference_duration_seconds",
            "Duração da inferência do modelo, sem HTTP, por backend.",
            labelnames=("backend",),
            buckets=INFERENCE_BUCKETS,
            registry=registry,
        ),
        # Cardinalidade limitada por construção: os rótulos vêm do conjunto fechado de
        # classes do modelo e da tabela de prioridades — nunca da entrada do usuário.
        predictions_total=Counter(
            "triagem_predictions_total",
            "Predições servidas, por condição prevista e prioridade atribuída.",
            labelnames=("condition", "priority"),
            registry=registry,
        ),
        prediction_confidence=Histogram(
            "triagem_prediction_confidence",
            "Distribuição da confiança do modelo na condição prevista.",
            buckets=CONFIDENCE_BUCKETS,
            registry=registry,
        ),
        model_info=Info(
            "triagem_model",
            "Metadados do modelo servido.",
            registry=registry,
        ),
    )

    @application.middleware("http")
    async def medir_requisicao(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # O scrape do Prometheus não é tráfego de triagem: medi-lo somaria uma requisição
        # a cada 5 s no painel de total mesmo com a API parada.
        if request.url.path == "/metrics":
            return await call_next(request)

        inicio = time.perf_counter()
        status = 500  # se o handler estourar antes de responder, o cliente recebe 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            rota = request.scope.get("route")
            caminho = rota.path if rota is not None else "sem_rota"
            metrics.requests_total.labels(
                method=request.method, route=caminho, status=str(status)
            ).inc()
            metrics.request_duration.labels(method=request.method, route=caminho).observe(
                time.perf_counter() - inicio
            )

    @application.get("/metrics", tags=["operação"], summary="Métricas no formato Prometheus")
    def exposicao() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return metrics
