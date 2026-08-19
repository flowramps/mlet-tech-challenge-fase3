"""Exportação do pipeline para ONNX.

O risco que estes testes cobrem é específico: o `TfidfVectorizer` do scikit-learn e a
implementação ONNX do mesmo operador tokenizam por caminhos diferentes. Uma divergência
ali não quebra nada de forma visível — o modelo continua respondendo, só que com outra
distribuição de features e, portanto, outras predições. Por isso a paridade é medida sobre
os rótulos previstos, não sobre a existência do arquivo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from triagem.data.prepare import LABEL_COLUMN, TEXT_COLUMN
from triagem.model.export_onnx import export_pipeline, quantize_dynamic_int8
from triagem.model.train import train_model

CONDICOES = {
    1: "malignant neoplasm with diffuse bone metastasis and marked weight loss",
    4: "acute myocardial infarction with ST segment elevation and chest pain",
}

# Vocabulário grande de propósito. A quantização INT8 encolhe a matriz de coeficientes do
# classificador, mas acrescenta tensores de escala e zero-point ao grafo. Num vocabulário de
# brinquedo esse overhead fixo supera a economia e o artefato quantizado sai *maior* — medir
# o ganho ali diria mais sobre o tamanho do fixture do que sobre a quantização.
VOCABULARIO_POR_CLASSE = 600
LAUDOS_POR_CLASSE = 60


@pytest.fixture
def corpus() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    registros = []
    for posicao, (rotulo, texto) in enumerate(CONDICOES.items()):
        marcadores = [f"term{posicao}x{i:04d}" for i in range(VOCABULARIO_POR_CLASSE)]
        for _ in range(LAUDOS_POR_CLASSE):
            # Cada laudo sorteia 40 marcadores da própria classe: o vocabulário fica amplo e
            # as classes seguem separáveis, então o modelo continua aprendendo algo.
            escolhidos = gerador.choice(marcadores, size=40, replace=False)
            registros.append({LABEL_COLUMN: rotulo, TEXT_COLUMN: f"{texto} {' '.join(escolhidos)}"})
    return pd.DataFrame(registros)


@pytest.fixture
def pipeline(corpus: pd.DataFrame):
    return train_model(
        corpus[TEXT_COLUMN], corpus[LABEL_COLUMN], model_type="logistic_regression", seed=42
    )


def _rotulos_onnx(caminho: Path, textos: list[str]) -> np.ndarray:
    import onnxruntime

    sessao = onnxruntime.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
    entrada = np.array(textos, dtype=object).reshape(-1, 1)
    nome = sessao.get_inputs()[0].name
    return np.asarray(sessao.run(None, {nome: entrada})[0]).ravel()


def test_export_pipeline_grava_um_grafo_carregavel(pipeline, tmp_path: Path):
    destino = export_pipeline(pipeline, tmp_path / "model.onnx")

    assert destino.exists()
    assert destino.stat().st_size > 0


def test_grafo_nao_depende_de_locale_do_sistema(pipeline, tmp_path: Path):
    """O grafo precisa declarar `locale=C` no `StringNormalizer`.

    Sem o atributo, o ONNX Runtime assume `en_US.UTF-8` e falha na inicialização em qualquer
    imagem que não tenha esse locale gerado — `python:3.12-slim`, entre elas. O sintoma é
    péssimo: a suíte passa na máquina de quem desenvolve e o container morre na subida.

    `C` é seguro aqui pelo mesmo motivo que `strip_accents=None`: o corpus é 100% ASCII, e a
    conversão para minúsculas em ASCII independe de locale. Verificado sobre os 2.888 laudos
    de teste, com 100% de concordância entre os dois locales.
    """
    import onnx

    destino = export_pipeline(pipeline, tmp_path / "model.onnx")
    grafo = onnx.load(str(destino)).graph

    normalizadores = [node for node in grafo.node if node.op_type == "StringNormalizer"]
    assert normalizadores, "o grafo deveria conter um StringNormalizer"

    for node in normalizadores:
        locales = [attr.s.decode() for attr in node.attribute if attr.name == "locale"]
        assert locales == ["C"], f"locale esperado 'C', encontrado {locales}"


def test_onnx_reproduz_os_rotulos_do_sklearn(pipeline, corpus: pd.DataFrame, tmp_path: Path):
    """Paridade exigida pelo plano: ao menos 99% de concordância com o backend de origem."""
    destino = export_pipeline(pipeline, tmp_path / "model.onnx")

    textos = corpus[TEXT_COLUMN].tolist()
    esperado = np.asarray(pipeline.predict(textos))
    obtido = _rotulos_onnx(destino, textos)

    concordancia = float((esperado == obtido).mean())
    assert concordancia >= 0.99, f"concordância de apenas {concordancia:.2%} com o sklearn"


def test_quantizacao_int8_preserva_os_rotulos(pipeline, corpus: pd.DataFrame, tmp_path: Path):
    """Quantizar sem medir a paridade seria trocar qualidade por bytes às cegas."""
    original = export_pipeline(pipeline, tmp_path / "model.onnx")
    quantizado = quantize_dynamic_int8(original, tmp_path / "model.int8.onnx")

    assert quantizado.exists()

    textos = corpus[TEXT_COLUMN].tolist()
    concordancia = float(
        (_rotulos_onnx(original, textos) == _rotulos_onnx(quantizado, textos)).mean()
    )
    assert concordancia >= 0.99, f"a quantização mudou {1 - concordancia:.2%} das predições"


def test_quantizacao_nao_altera_o_grafo_de_um_classificador_linear(pipeline, tmp_path: Path):
    """Registra por teste o motivo de a quantização INT8 não render nada neste modelo.

    `quantize_dynamic` age sobre `MatMul`/`Gemm`/`Conv`. O conversor traduz a regressão
    logística para `LinearClassifier`, um operador do domínio `ai.onnx.ml` que a quantização
    não reconhece — então o grafo sai idêntico e o artefato não encolhe um byte. Está aqui
    como asserção, e não só como comentário no README, para que uma versão futura do
    onnxruntime que passe a cobrir esse operador **quebre o teste** e obrigue a remedir os
    números publicados, em vez de deixar a documentação envelhecer em silêncio.
    """
    import onnx

    original = export_pipeline(pipeline, tmp_path / "model.onnx")
    quantizado = quantize_dynamic_int8(original, tmp_path / "model.int8.onnx")

    def operadores(caminho: Path) -> list[str]:
        return sorted(node.op_type for node in onnx.load(str(caminho)).graph.node)

    assert "MatMul" not in operadores(original)
    assert "LinearClassifier" in operadores(original)
    assert operadores(quantizado) == operadores(original)
