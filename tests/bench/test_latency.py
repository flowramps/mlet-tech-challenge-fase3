import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from triagem.bench.latency import LatencyReport, measure_latency, save_report
from triagem.inference.base import Prediction


class ClassificadorFalso:
    name = "fake"
    version = "1.0.0"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return [Prediction(condition="neoplasms", confidence=0.5) for _ in texts]


def test_relatorio_traz_o_numero_de_amostras_pedido():
    relatorio = measure_latency(ClassificadorFalso(), ["laudo"], iterations=50, warmup=5)
    assert isinstance(relatorio, LatencyReport)
    assert relatorio.samples == 50
    assert relatorio.backend == "fake"


def test_percentis_sao_monotonicos():
    relatorio = measure_latency(ClassificadorFalso(), ["laudo"], iterations=50, warmup=5)
    assert relatorio.p50_ms <= relatorio.p95_ms <= relatorio.p99_ms


def test_metricas_sao_positivas():
    relatorio = measure_latency(ClassificadorFalso(), ["laudo"], iterations=30, warmup=2)
    assert relatorio.mean_ms > 0
    assert relatorio.throughput_rps > 0


def test_exige_ao_menos_um_texto():
    with pytest.raises(ValueError, match="ao menos um texto"):
        measure_latency(ClassificadorFalso(), [], iterations=10, warmup=1)


def test_save_report_grava_json(tmp_path: Path):
    relatorio = measure_latency(ClassificadorFalso(), ["laudo"], iterations=10, warmup=1)
    destino = save_report(relatorio, tmp_path / "sub" / "latency.json")
    corpo = json.loads(destino.read_text())
    assert corpo["backend"] == "fake"
    assert corpo["samples"] == 10
