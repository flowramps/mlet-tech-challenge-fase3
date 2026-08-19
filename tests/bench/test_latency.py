import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from triagem.bench.latency import (
    LatencyReport,
    compare_backends,
    measure_latency,
    save_comparison,
    save_report,
)
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


class ClassificadorRapido(ClassificadorFalso):
    name = "rapido"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        time.sleep(0.001)
        return super().predict(texts)


class ClassificadorLento(ClassificadorFalso):
    name = "lento"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        time.sleep(0.004)
        return super().predict(texts)


def test_compare_backends_mede_todos_na_mesma_rodada(tmp_path: Path):
    """Os backends precisam ser medidos no mesmo processo e com a mesma amostra.

    Rodar cada um em processo separado deixaria a comparação à mercê do estado da máquina
    entre as execuções — e o número que a entrega publica é justamente a razão entre eles.
    """
    relatorios = compare_backends(
        [ClassificadorRapido(), ClassificadorLento()], ["laudo"], iterations=20, warmup=2
    )

    assert [relatorio.backend for relatorio in relatorios] == ["rapido", "lento"]
    assert relatorios[1].p50_ms > relatorios[0].p50_ms


def test_save_comparison_registra_o_ganho_relativo(tmp_path: Path):
    """O JSON do comparativo carrega a razão entre os backends, não só os números soltos."""
    relatorios = compare_backends(
        [ClassificadorLento(), ClassificadorRapido()], ["laudo"], iterations=20, warmup=2
    )

    destino = save_comparison(relatorios, tmp_path / "latency_comparison.json")
    corpo = json.loads(destino.read_text())

    assert corpo["baseline"] == "lento"  # o primeiro da lista é a referência
    assert corpo["backends"]["rapido"]["speedup_p50"] > 1.0
    assert corpo["backends"]["lento"]["speedup_p50"] == 1.0
