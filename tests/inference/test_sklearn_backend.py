from pathlib import Path

import pytest

from triagem.inference.base import Classifier, Prediction
from triagem.inference.sklearn_backend import SklearnClassifier
from triagem.model.train import save_model, train_model

TEXTOS = [
    "tumor maligno com metástase óssea difusa",
    "carcinoma invasivo em biópsia mamária",
    "infarto agudo do miocárdio com dor precordial",
    "insuficiência cardíaca com fração de ejeção reduzida",
] * 8
ROTULOS = [1, 1, 4, 4] * 8
NOMES = {1: "neoplasms", 4: "cardiovascular diseases"}


@pytest.fixture
def classificador(tmp_path: Path) -> SklearnClassifier:
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    caminho = save_model(
        pipeline,
        NOMES,
        version="1.0.0",
        model_type="logistic_regression",
        destination=tmp_path / "model.joblib",
    )
    return SklearnClassifier.load(caminho)


def test_backend_satisfaz_o_protocolo(classificador):
    assert isinstance(classificador, Classifier)
    assert classificador.name == "sklearn"
    assert classificador.version == "1.0.0"


def test_predict_devolve_uma_predicao_por_texto(classificador):
    resultado = classificador.predict(["infarto agudo", "tumor maligno"])
    assert len(resultado) == 2
    assert all(isinstance(item, Prediction) for item in resultado)


def test_predict_traduz_rotulo_para_nome_legivel(classificador):
    predicao = classificador.predict(["infarto agudo do miocárdio com dor precordial"])[0]
    assert predicao.condition == "cardiovascular diseases"


def test_predict_preserva_a_ordem_de_entrada(classificador):
    resultado = classificador.predict(
        [
            "infarto agudo do miocárdio com dor precordial",
            "tumor maligno com metástase óssea difusa",
        ]
    )
    assert resultado[0].condition == "cardiovascular diseases"
    assert resultado[1].condition == "neoplasms"


def test_confianca_fica_entre_zero_e_um(classificador):
    predicao = classificador.predict(["tumor maligno com metástase"])[0]
    assert 0.0 <= predicao.confidence <= 1.0


def test_predict_com_lista_vazia_devolve_lista_vazia(classificador):
    assert classificador.predict([]) == []


def test_prediction_e_imutavel():
    predicao = Prediction(condition="neoplasms", confidence=0.9)
    with pytest.raises(AttributeError):
        predicao.condition = "outra"
