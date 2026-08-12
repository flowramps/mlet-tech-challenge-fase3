from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

from triagem.model.train import (
    MODEL_TYPES,
    build_pipeline,
    load_bundle,
    save_model,
    select_champion,
    train_model,
)

TEXTOS = [
    "tumor maligno com metástase óssea difusa e perda de peso",
    "carcinoma invasivo detectado em biópsia de mama esquerda",
    "infarto agudo do miocárdio com supradesnivelamento do segmento ST",
    "insuficiência cardíaca congestiva com fração de ejeção reduzida",
] * 8
ROTULOS = [1, 1, 4, 4] * 8


def test_model_types_expoe_os_dois_candidatos():
    assert MODEL_TYPES == ("random_forest", "logistic_regression")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_build_pipeline_tem_vetorizador_e_classificador(model_type):
    pipeline = build_pipeline(model_type, seed=42)
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["tfidf", "clf"]


def test_build_pipeline_rejeita_tipo_desconhecido():
    with pytest.raises(ValueError, match="modelo desconhecido"):
        build_pipeline("rede_neural_gigante", seed=42)


def test_candidatos_compartilham_a_mesma_vetorizacao():
    """A comparação só é justa se o vetorizador for idêntico nos dois."""
    floresta = build_pipeline("random_forest", seed=42).named_steps["tfidf"]
    logistica = build_pipeline("logistic_regression", seed=42).named_steps["tfidf"]
    assert floresta.get_params() == logistica.get_params()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_train_model_aprende_a_separar_as_classes(model_type):
    pipeline = train_model(TEXTOS, ROTULOS, model_type=model_type, seed=42)
    previsto = pipeline.predict(["infarto agudo do miocárdio com dor precordial"])
    assert previsto[0] == 4


def test_train_model_e_deterministico_com_a_mesma_seed():
    primeiro = train_model(TEXTOS, ROTULOS, model_type="random_forest", seed=42).predict(TEXTOS)
    segundo = train_model(TEXTOS, ROTULOS, model_type="random_forest", seed=42).predict(TEXTOS)
    assert primeiro.tolist() == segundo.tolist()


def test_select_champion_escolhe_o_maior_f1():
    assert select_champion({"random_forest": 0.53, "logistic_regression": 0.58}) == (
        "logistic_regression"
    )


def test_select_champion_desempata_pela_ordem_dos_candidatos():
    assert select_champion({"logistic_regression": 0.5, "random_forest": 0.5}) == "random_forest"


def test_select_champion_rejeita_dicionario_vazio():
    with pytest.raises(ValueError, match="nenhum candidato"):
        select_champion({})


def test_save_model_grava_bundle_com_rotulos_versao_e_tipo(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    destino = save_model(
        pipeline,
        {1: "neoplasms", 4: "cardiovascular diseases"},
        version="1.0.0",
        model_type="logistic_regression",
        destination=tmp_path / "sub" / "model.joblib",
    )

    assert destino.exists()
    bundle = joblib.load(destino)
    assert set(bundle) == {"pipeline", "labels", "version", "model_type"}
    assert bundle["version"] == "1.0.0"
    assert bundle["model_type"] == "logistic_regression"
    assert bundle["labels"][4] == "cardiovascular diseases"


def test_load_bundle_faz_roundtrip(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    destino = save_model(
        pipeline,
        {1: "neoplasms"},
        version="9.9.9",
        model_type="logistic_regression",
        destination=tmp_path / "model.joblib",
    )

    bundle = load_bundle(destino)

    assert bundle["version"] == "9.9.9"
    assert bundle["pipeline"].predict(["tumor maligno"]).shape == (1,)
