from pathlib import Path

import pytest

from triagem.config import Settings
from triagem.inference.factory import load_classifier
from triagem.inference.sklearn_backend import SklearnClassifier
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


def test_backend_desconhecido_falha_com_mensagem_util(tmp_path: Path):
    """A mensagem precisa listar as opções — é o que a Etapa 4 vai estender."""
    settings = Settings(models_dir=tmp_path, model_backend="tensorrt")

    with pytest.raises(ValueError, match="backend não suportado") as erro:
        load_classifier(settings)

    assert "sklearn" in str(erro.value)
