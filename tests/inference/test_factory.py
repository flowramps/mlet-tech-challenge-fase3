from pathlib import Path

import pytest

from triagem.config import Settings
from triagem.inference.factory import load_classifier
from triagem.inference.onnx_backend import OnnxClassifier
from triagem.inference.sklearn_backend import SklearnClassifier
from triagem.model.export_onnx import export_pipeline
from triagem.model.train import save_model, train_model

TEXTOS = [
    "tumor maligno com metástase óssea difusa",
    "infarto agudo do miocárdio com dor precordial",
] * 16
ROTULOS = [1, 4] * 16
NOMES = {1: "neoplasms", 4: "cardiovascular diseases"}


def test_carrega_backend_sklearn(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    save_model(
        pipeline,
        NOMES,
        version="1.0.0",
        model_type="logistic_regression",
        destination=tmp_path / "model.joblib",
    )

    classificador = load_classifier(Settings(models_dir=tmp_path, model_backend="sklearn"))

    assert isinstance(classificador, SklearnClassifier)
    assert classificador.name == "sklearn"


def test_carrega_backend_onnx(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    export_pipeline(
        pipeline,
        tmp_path / "model.onnx",
        labels_by_id=NOMES,
        version="1.0.0",
        model_type="logistic_regression",
    )

    classificador = load_classifier(Settings(models_dir=tmp_path, model_backend="onnx"))

    assert isinstance(classificador, OnnxClassifier)
    assert classificador.name == "onnx"


def test_backend_desconhecido_falha_com_mensagem_util(tmp_path: Path):
    """A mensagem precisa listar as opções disponíveis, não só recusar."""
    settings = Settings(models_dir=tmp_path, model_backend="tensorrt")

    with pytest.raises(ValueError, match="backend não suportado") as erro:
        load_classifier(settings)

    assert "sklearn" in str(erro.value)
    assert "onnx" in str(erro.value)
