"""Medição de latência de inferência.

Mede uma requisição por vez, que é o padrão de uso da triagem — um laudo chega, uma
resposta sai. Percentis importam mais que a média: p99 é o que a equipe na ponta sente
quando a fila aperta, e é o número que uma média baixa esconde.

O aquecimento não é detalhe: a primeira inferência de um processo paga importações
tardias e alocação de buffers, e chega a ser cinquenta vezes mais lenta que as seguintes.
Incluí-la na amostra mediria o startup, não o regime permanente.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path

from triagem.inference.base import Classifier

logger = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20


@dataclass(frozen=True)
class LatencyReport:
    """Resumo estatístico de uma rodada de medição."""

    backend: str
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float


def _percentile(values: list[float], percentile: float) -> float:
    ordenados = sorted(values)
    posicao = (percentile / 100) * (len(ordenados) - 1)
    return ordenados[min(int(round(posicao)), len(ordenados) - 1)]


def measure_latency(
    classifier: Classifier,
    texts: Sequence[str],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> LatencyReport:
    """Executa ``iterations`` inferências unitárias e resume a distribuição."""
    if not texts:
        raise ValueError("é preciso ao menos um texto para medir latência")

    amostras = list(islice(cycle(texts), warmup + iterations))

    for texto in amostras[:warmup]:
        classifier.predict([texto])

    duracoes_ms: list[float] = []
    for texto in amostras[warmup:]:
        inicio = time.perf_counter()
        classifier.predict([texto])
        duracoes_ms.append((time.perf_counter() - inicio) * 1_000)

    media = sum(duracoes_ms) / len(duracoes_ms)

    return LatencyReport(
        backend=classifier.name,
        samples=len(duracoes_ms),
        mean_ms=round(media, 3),
        p50_ms=round(_percentile(duracoes_ms, 50), 3),
        p95_ms=round(_percentile(duracoes_ms, 95), 3),
        p99_ms=round(_percentile(duracoes_ms, 99), 3),
        throughput_rps=round(1_000 / media, 1),
    )


def compare_backends(
    classifiers: Sequence[Classifier],
    texts: Sequence[str],
    *,
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> list[LatencyReport]:
    """Mede vários backends na mesma rodada, sobre a mesma amostra.

    No mesmo processo de propósito. Medir cada backend numa execução separada faria a
    comparação absorver o estado da máquina entre elas — frequência da CPU, pressão de cache,
    o que mais estiver rodando — e o número publicado pela entrega é justamente a razão entre
    os backends, não o valor absoluto de cada um.
    """
    return [
        measure_latency(classifier, texts, iterations=iterations, warmup=warmup)
        for classifier in classifiers
    ]


def save_report(report: LatencyReport, destination: Path) -> Path:
    """Grava o relatório como JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2))
    return destination


def save_comparison(reports: Sequence[LatencyReport], destination: Path) -> Path:
    """Grava o comparativo com o ganho relativo já calculado.

    O primeiro relatório da sequência é a referência. Deixar a razão pronta no artefato evita
    que ela seja recalculada à mão para o README e para o vídeo — e divirja entre os dois.
    """
    if not reports:
        raise ValueError("é preciso ao menos um relatório para comparar")

    referencia = reports[0]

    def _speedup(base: float, medido: float) -> float | None:
        # Os percentis são arredondados ao microssegundo: um backend mais rápido que isso
        # zera o denominador. `None` diz "rápido demais para esta régua", que é honesto —
        # inventar um número aqui seria publicar ruído de arredondamento como ganho.
        return round(base / medido, 2) if medido > 0 else None

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "baseline": referencia.backend,
                "samples": referencia.samples,
                "backends": {
                    relatorio.backend: {
                        **asdict(relatorio),
                        "speedup_p50": _speedup(referencia.p50_ms, relatorio.p50_ms),
                        "speedup_p99": _speedup(referencia.p99_ms, relatorio.p99_ms),
                    }
                    for relatorio in reports
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return destination


def main() -> None:
    """Mede todos os backends disponíveis e grava os relatórios individuais e o comparativo.

    O backend scikit-learn vem primeiro por ser a referência da comparação — é o que a
    Etapa 1 mediu e o que o README publica como baseline. Backends cujo artefato ainda não
    foi gerado são simplesmente pulados, com aviso: quem ainda não rodou `make train` depois
    da exportação continua conseguindo medir o que tem.
    """
    from triagem.config import get_settings
    from triagem.data.download import download_dataset
    from triagem.data.prepare import TEXT_COLUMN, load_split
    from triagem.inference.factory import SUPPORTED_BACKENDS, load_classifier

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    paths = download_dataset(settings.data_dir / "raw")
    textos = load_split(paths["test"])[TEXT_COLUMN].head(100).tolist()

    classificadores: list[Classifier] = []
    for backend in SUPPORTED_BACKENDS:
        try:
            classificadores.append(
                load_classifier(settings.model_copy(update={"model_backend": backend}))
            )
        except (FileNotFoundError, OSError) as erro:
            logger.warning("backend %s indisponível, pulando: %s", backend, erro)

    if not classificadores:
        raise SystemExit("nenhum backend disponível — rode `make train` antes de medir")

    relatorios = compare_backends(classificadores, textos)

    for relatorio in relatorios:
        save_report(relatorio, settings.metrics_dir / f"latency_{relatorio.backend}.json")
        logger.info(
            "backend=%-8s p50=%.3fms p95=%.3fms p99=%.3fms throughput=%.1f req/s",
            relatorio.backend,
            relatorio.p50_ms,
            relatorio.p95_ms,
            relatorio.p99_ms,
            relatorio.throughput_rps,
        )

    if len(relatorios) > 1:
        destino = save_comparison(relatorios, settings.metrics_dir / "latency_comparison.json")
        referencia = relatorios[0]
        for relatorio in relatorios[1:]:
            logger.info(
                "%s é %.2fx mais rápido que %s no p50",
                relatorio.backend,
                referencia.p50_ms / relatorio.p50_ms,
                referencia.backend,
            )
        logger.info("comparativo gravado em %s", destino)


if __name__ == "__main__":
    main()
