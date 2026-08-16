from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from triagem.api.main import create_app
from triagem.inference.base import Prediction

LAUDO = "paciente com cefaleia súbita intensa, rigidez de nuca e fotofobia há duas horas"


class ClassificadorFalso:
    name = "fake"
    version = "1.0.0"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return [Prediction(condition="nervous system diseases", confidence=0.91) for _ in texts]


@pytest.fixture
def client():
    with TestClient(create_app(ClassificadorFalso())) as cliente:
        yield cliente


def test_metrics_responde_no_formato_prometheus(client):
    resposta = client.get("/metrics")
    assert resposta.status_code == 200
    assert resposta.headers["content-type"].startswith("text/plain")


def test_predict_incrementa_o_contador_por_rota_e_status(client):
    client.post("/predict", json={"text": LAUDO})
    client.post("/predict", json={"text": LAUDO})
    corpo = client.get("/metrics").text
    assert 'triagem_http_requests_total{method="POST",route="/predict",status="200"} 2.0' in corpo


def test_erro_de_validacao_conta_com_o_status_422(client):
    """Sem contar os 4xx não existe painel de taxa de erro que mereça o nome."""
    client.post("/predict", json={"text": "curto"})
    corpo = client.get("/metrics").text
    assert 'triagem_http_requests_total{method="POST",route="/predict",status="422"} 1.0' in corpo


def test_duracao_da_requisicao_e_observada_por_rota(client):
    client.post("/predict", json={"text": LAUDO})
    corpo = client.get("/metrics").text
    assert (
        'triagem_http_request_duration_seconds_count{method="POST",route="/predict"} 1.0' in corpo
    )


def test_inferencia_e_observada_por_backend(client):
    client.post("/predict", json={"text": LAUDO})
    corpo = client.get("/metrics").text
    assert 'triagem_inference_duration_seconds_count{backend="fake"} 1.0' in corpo


def test_scrape_do_metrics_nao_conta_como_trafego(client):
    """O coletor raspa a cada 5 s: se o scrape contasse, o painel de total de
    requisições subiria sozinho com a API parada."""
    client.get("/metrics")
    corpo = client.get("/metrics").text
    assert 'route="/metrics"' not in corpo


def test_cada_aplicacao_tem_registro_proprio():
    """A suíte cria dezenas de apps no mesmo processo; um registry global colidiria na
    segunda criação e vazaria contadores de um teste para o outro."""
    with TestClient(create_app(ClassificadorFalso())) as primeira:
        primeira.post("/predict", json={"text": LAUDO})

    with TestClient(create_app(ClassificadorFalso())) as segunda:
        corpo = segunda.get("/metrics").text

    assert 'route="/predict"' not in corpo
