import json
from pathlib import Path

from triagem.model.evaluate import evaluate_model, save_metrics
from triagem.model.train import train_model

TEXTOS = [
    "tumor maligno com metástase óssea difusa",
    "carcinoma invasivo em biópsia mamária",
    "infarto agudo do miocárdio com dor precordial",
    "insuficiência cardíaca com fração de ejeção reduzida",
] * 8
ROTULOS = [1, 1, 4, 4] * 8
NOMES = {1: "neoplasms", 4: "cardiovascular diseases"}


def test_evaluate_model_retorna_metricas_esperadas():
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    metricas = evaluate_model(pipeline, TEXTOS, ROTULOS, NOMES)

    assert set(metricas) == {
        "accuracy",
        "f1_macro",
        "per_class",
        "confusion_matrix",
        "support",
    }
    assert 0.0 <= metricas["f1_macro"] <= 1.0
    assert metricas["support"] == len(ROTULOS)


def test_per_class_usa_nome_legivel_da_condicao():
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    metricas = evaluate_model(pipeline, TEXTOS, ROTULOS, NOMES)
    assert "cardiovascular diseases" in metricas["per_class"]
    assert "precision" in metricas["per_class"]["cardiovascular diseases"]


def test_matriz_de_confusao_e_quadrada_e_serializavel():
    pipeline = train_model(TEXTOS, ROTULOS, model_type="logistic_regression", seed=42)
    matriz = evaluate_model(pipeline, TEXTOS, ROTULOS, NOMES)["confusion_matrix"]
    assert len(matriz) == len(matriz[0]) == 2
    json.dumps(matriz)


def test_save_metrics_grava_json_indentado(tmp_path: Path):
    destino = save_metrics({"f1_macro": 0.5}, tmp_path / "sub" / "metrics.json")
    assert json.loads(destino.read_text())["f1_macro"] == 0.5
