from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from triagem.model.train import build_pipeline, load_bundle, save_model, train_model

TEXTOS = [
    "tumor maligno com metástase óssea difusa e perda de peso",
    "carcinoma invasivo detectado em biópsia de mama esquerda",
    "infarto agudo do miocárdio com supradesnivelamento do segmento ST",
    "insuficiência cardíaca congestiva com fração de ejeção reduzida",
] * 8
ROTULOS = [1, 1, 4, 4] * 8


def test_build_pipeline_tem_vetorizador_e_classificador():
    pipeline = build_pipeline(seed=42)
    assert isinstance(pipeline, Pipeline)
    assert list(dict(pipeline.steps)) == ["tfidf", "clf"]


def test_train_model_aprende_a_separar_as_classes():
    pipeline = train_model(TEXTOS, ROTULOS, seed=42)
    previsto = pipeline.predict(["infarto agudo do miocárdio com dor precordial"])
    assert previsto[0] == 4


def test_train_model_e_deterministico_com_a_mesma_seed():
    primeiro = train_model(TEXTOS, ROTULOS, seed=42).predict(TEXTOS)
    segundo = train_model(TEXTOS, ROTULOS, seed=42).predict(TEXTOS)
    assert primeiro.tolist() == segundo.tolist()


def test_save_model_grava_bundle_com_rotulos_e_versao(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, seed=42)
    destino = save_model(
        pipeline,
        {1: "neoplasms", 4: "cardiovascular diseases"},
        "1.0.0",
        tmp_path / "sub" / "model.joblib",
    )

    assert destino.exists()
    bundle = joblib.load(destino)
    assert set(bundle) == {"pipeline", "labels", "version"}
    assert bundle["version"] == "1.0.0"
    assert bundle["labels"][4] == "cardiovascular diseases"


def test_load_bundle_faz_roundtrip(tmp_path: Path):
    pipeline = train_model(TEXTOS, ROTULOS, seed=42)
    destino = save_model(pipeline, {1: "neoplasms"}, "9.9.9", tmp_path / "model.joblib")

    bundle = load_bundle(destino)

    assert bundle["version"] == "9.9.9"
    assert bundle["pipeline"].predict(["tumor maligno"]).shape == (1,)
