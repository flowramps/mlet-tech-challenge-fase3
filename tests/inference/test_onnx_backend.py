"""Backend ONNX.

Espelha `test_sklearn_backend.py` de propósito: os dois backends implementam o mesmo
Protocol, então devem passar pelas mesmas perguntas. O teste que não tem equivalente é o de
paridade — ele existe porque um backend novo só vale se servir o *mesmo* modelo.
"""

from pathlib import Path

import pytest

from triagem.inference.base import Classifier, Prediction
from triagem.inference.onnx_backend import OnnxClassifier
from triagem.inference.sklearn_backend import SklearnClassifier
from triagem.model.export_onnx import export_pipeline
from triagem.model.train import save_model, train_model

TEXTOS = [
    "malignant tumor with diffuse bone metastasis",
    "invasive carcinoma found in breast biopsy",
    "acute myocardial infarction with chest pain",
    "heart failure with reduced ejection fraction",
] * 8
ROTULOS = [1, 1, 4, 4] * 8
NOMES = {1: "neoplasms", 4: "cardiovascular diseases"}


@pytest.fixture
def pipeline():
    return train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)


@pytest.fixture
def classificador(pipeline, tmp_path: Path) -> OnnxClassifier:
    caminho = export_pipeline(
        pipeline,
        tmp_path / "model.onnx",
        labels_by_id=NOMES,
        version="1.0.0",
        model_type="logistic_regression",
    )
    return OnnxClassifier.load(caminho)


def test_backend_satisfaz_o_protocolo(classificador):
    assert isinstance(classificador, Classifier)
    assert classificador.name == "onnx"
    assert classificador.version == "1.0.0"


def test_metadados_viajam_dentro_do_proprio_grafo(classificador):
    """O .onnx é autossuficiente: rótulos e versão não dependem de um arquivo ao lado."""
    assert classificador.model_type == "logistic_regression"


def test_predict_devolve_uma_predicao_por_texto(classificador):
    resultado = classificador.predict(["acute myocardial infarction", "malignant tumor"])
    assert len(resultado) == 2
    assert all(isinstance(item, Prediction) for item in resultado)


def test_predict_traduz_rotulo_para_nome_legivel(classificador):
    predicao = classificador.predict(["acute myocardial infarction with chest pain"])[0]
    assert predicao.condition == "cardiovascular diseases"


def test_predict_preserva_a_ordem_de_entrada(classificador):
    resultado = classificador.predict(
        [
            "acute myocardial infarction with chest pain",
            "malignant tumor with diffuse bone metastasis",
        ]
    )
    assert resultado[0].condition == "cardiovascular diseases"
    assert resultado[1].condition == "neoplasms"


def test_confianca_fica_entre_zero_e_um(classificador):
    predicao = classificador.predict(["malignant tumor with metastasis"])[0]
    assert 0.0 <= predicao.confidence <= 1.0


def test_predict_com_lista_vazia_devolve_lista_vazia(classificador):
    assert classificador.predict([]) == []


def test_paridade_com_o_backend_sklearn(classificador, pipeline, tmp_path: Path):
    """Trocar o motor não pode trocar a resposta: mesmos rótulos, confiança equivalente.

    Este é o teste que sustenta a comparação de latência. Sem ele, um backend "mais rápido"
    poderia estar apenas respondendo outra coisa.
    """
    referencia = SklearnClassifier.load(
        save_model(
            pipeline,
            NOMES,
            version="1.0.0",
            model_type="logistic_regression",
            destination=tmp_path / "model.joblib",
        )
    )

    esperado = referencia.predict(TEXTOS)
    obtido = classificador.predict(TEXTOS)

    pares = list(zip(esperado, obtido, strict=True))
    concordancia = sum(a.condition == b.condition for a, b in pares) / len(TEXTOS)
    assert concordancia >= 0.99, f"concordância de apenas {concordancia:.2%}"

    for a, b in pares:
        assert a.confidence == pytest.approx(b.confidence, abs=1e-5)
