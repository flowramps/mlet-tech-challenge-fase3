import json
from pathlib import Path

import pytest

from triagem.model.evaluate import evaluate_model, priority_recall, save_metrics
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


def test_priority_recall_mede_a_prioridade_alta():
    # cardiovascular diseases -> "alta"; neoplasms -> "media" (src/triagem/inference/priority.py)
    reais = [4, 4, 4, 4, 1, 1]
    previstas = [4, 4, 1, 4, 1, 1]

    recall = priority_recall(reais, previstas, NOMES, priority="alta")

    # 4 casos reais de prioridade alta (os quatro "4"); 3 previstos corretamente como alta
    assert recall == 3 / 4


def test_priority_recall_acerta_mesmo_com_condicao_diferente_na_mesma_prioridade():
    """cardiovascular <-> nervous system: condições diferentes, mesma prioridade "alta"."""
    nomes = {4: "cardiovascular diseases", 3: "nervous system diseases"}
    recall = priority_recall([4, 4], [3, 4], nomes, priority="alta")
    assert recall == 1.0


def test_priority_recall_sem_casos_reais_da_prioridade():
    with pytest.raises(ValueError, match="alta"):
        priority_recall([1, 1], [1, 4], NOMES, priority="alta")
