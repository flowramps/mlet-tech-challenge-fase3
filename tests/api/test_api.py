from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from triagem.api.main import create_app
from triagem.inference.base import Prediction

LAUDO = "paciente com dor precordial intensa e supradesnivelamento do segmento ST"


class ClassificadorFalso:
    name = "fake"
    version = "1.0.0"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return [Prediction(condition="cardiovascular diseases", confidence=0.87) for _ in texts]


@pytest.fixture
def client():
    # O `with` é obrigatório: sem ele o TestClient não dispara o lifespan e o modelo
    # nunca chega em app.state. Usá-lo faz o teste exercitar o startup real.
    with TestClient(create_app(ClassificadorFalso())) as cliente:
        yield cliente


def test_health_reporta_versao_e_backend(client):
    resposta = client.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert corpo["model_version"] == "1.0.0"
    assert corpo["backend"] == "fake"


def test_predict_classifica_o_laudo(client):
    resposta = client.post("/predict", json={"text": LAUDO})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["condition"] == "cardiovascular diseases"
    assert corpo["confidence"] == pytest.approx(0.87)


def test_predict_anexa_prioridade_marcada_como_regra(client):
    corpo = client.post("/predict", json={"text": LAUDO}).json()
    assert corpo["priority"] == "alta"
    assert corpo["priority_source"] == "regra_de_negocio"


def test_predict_reporta_tempo_de_inferencia(client):
    corpo = client.post("/predict", json={"text": LAUDO}).json()
    assert corpo["inference_ms"] >= 0.0
    assert corpo["backend"] == "fake"


def test_predict_rejeita_texto_curto_demais(client):
    resposta = client.post("/predict", json={"text": "curto"})
    assert resposta.status_code == 422


def test_predict_rejeita_corpo_sem_texto(client):
    assert client.post("/predict", json={}).status_code == 422


def test_openapi_documenta_as_duas_rotas(client):
    caminhos = client.get("/openapi.json").json()["paths"]
    assert "/health" in caminhos
    assert "/predict" in caminhos


def test_startup_registra_o_modelo_carregado(caplog):
    """Sem isso não há como saber, pelo log, qual modelo subiu em produção."""
    with (
        caplog.at_level("INFO", logger="triagem.api.main"),
        TestClient(create_app(ClassificadorFalso())),
    ):
        pass

    assert any("fake" in registro.getMessage() for registro in caplog.records)
