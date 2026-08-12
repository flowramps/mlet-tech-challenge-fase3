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


def save_report(report: LatencyReport, destination: Path) -> Path:
    """Grava o relatório como JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2))
    return destination


def main() -> None:
    from triagem.config import get_settings
    from triagem.data.download import download_dataset
    from triagem.data.prepare import TEXT_COLUMN, load_split
    from triagem.inference.factory import load_classifier

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    classifier = load_classifier(settings)
    paths = download_dataset(settings.data_dir / "raw")
    textos = load_split(paths["test"])[TEXT_COLUMN].head(100).tolist()

    relatorio = measure_latency(classifier, textos)
    save_report(relatorio, settings.metrics_dir / f"latency_{relatorio.backend}.json")

    logger.info(
        "backend=%s p50=%.3fms p95=%.3fms p99=%.3fms throughput=%.1f req/s",
        relatorio.backend,
        relatorio.p50_ms,
        relatorio.p95_ms,
        relatorio.p99_ms,
        relatorio.throughput_rps,
    )


if __name__ == "__main__":
    main()
