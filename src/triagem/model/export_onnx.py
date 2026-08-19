"""Exportação do pipeline treinado para ONNX, com variante quantizada em INT8.

**O pipeline inteiro é exportado, não só o classificador.** A medição da Etapa 1 mostrou que
a vetorização TF-IDF consome 0,664 ms dos 0,678 ms de uma inferência — o classificador linear
é ruído estatístico dentro desse total. Exportar apenas o classificador, como um plano B
conservador sugeriria, otimizaria os 2% e deixaria intactos os 98%. O alvo é o tokenizador.

Isso traz o risco conhecido de o operador `TfidfVectorizer` do ONNX tokenizar diferente do
scikit-learn. O risco não é evitado, é **medido**: a suíte exige ao menos 99% de concordância
de rótulos com o backend de origem, e a mesma verificação roda sobre a variante quantizada.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# O ONNX recebe uma coluna de strings, não um vetor 1-D: o conversor do TfidfVectorizer
# espera a forma (N, 1). `None` deixa o lote livre para a sessão aceitar qualquer tamanho.
INPUT_NAME = "laudo"
OPSET = 15

# O `StringNormalizer` do ONNX Runtime assume `en_US.UTF-8` quando o grafo não declara um
# locale, e falha na *inicialização da sessão* se o sistema não tiver esse locale gerado —
# o caso de `python:3.12-slim`, a base da imagem da API. Declarar `C` torna o artefato
# independente de pacotes de idioma do sistema operacional. É seguro pelo mesmo motivo que
# `strip_accents=None`: o corpus é 100% ASCII, e minusculizar ASCII não depende de locale.
NORMALIZER_LOCALE = "C"


def _fixar_locale(modelo) -> None:
    """Declara ``locale`` nos nós ``StringNormalizer`` do grafo já convertido.

    O conversor não emite o atributo, então o valor fica a cargo do default do runtime. Isso
    é ajustado aqui, depois da conversão, em vez de num fork do conversor: são poucos nós e
    a alteração é local ao grafo que estamos gravando.
    """
    from onnx import helper

    for node in modelo.graph.node:
        if node.op_type == "StringNormalizer":
            if any(attr.name == "locale" for attr in node.attribute):
                continue
            node.attribute.append(helper.make_attribute("locale", NORMALIZER_LOCALE))


def export_pipeline(
    pipeline: Pipeline,
    destination: Path | str,
    *,
    labels_by_id: dict[int, str] | None = None,
    version: str = "1.0.0",
    model_type: str = "desconhecido",
) -> Path:
    """Converte o pipeline scikit-learn em um grafo ONNX equivalente.

    O ``ZipMap`` é desligado de propósito. Ligado — que é o padrão do conversor — a saída de
    probabilidades vira uma lista de dicionários por amostra, o que obriga o runtime a alocar
    um dicionário Python por inferência e joga fora boa parte do ganho que motiva a exportação.
    Desligado, sai um tensor denso, que é o que o backend de inferência consome.

    Os rótulos legíveis, a versão e o tipo do modelo são gravados em ``metadata_props``, dentro
    do próprio grafo. O ``.onnx`` fica autossuficiente: um arquivo ao lado poderia ser copiado
    sem o outro e o serviço subiria traduzindo rótulos errados, sem nada quebrar de forma
    visível. É o mesmo motivo de ``save_model`` serializar rótulos junto do pipeline.
    """
    import json

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    caminho = Path(destination)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    modelo = to_onnx(
        pipeline,
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        options={id(pipeline): {"zipmap": False}},
        target_opset=OPSET,
    )

    _fixar_locale(modelo)

    metadados = {
        "triagem_labels": json.dumps(labels_by_id or {}),
        "triagem_version": version,
        "triagem_model_type": model_type,
    }
    for chave, valor in metadados.items():
        entrada = modelo.metadata_props.add()
        entrada.key = chave
        entrada.value = valor

    caminho.write_bytes(modelo.SerializeToString())

    logger.info("grafo ONNX exportado: %s (%.2f MB)", caminho, caminho.stat().st_size / 1e6)
    return caminho


def quantize_dynamic_int8(source: Path | str, destination: Path | str) -> Path:
    """Gera a variante INT8 por quantização dinâmica.

    Dinâmica, e não estática, porque a estática exigiria um conjunto de calibração
    representativo e um passo a mais no pipeline de treino para produzi-lo. A dinâmica calcula
    a escala dos ativadores em tempo de execução: menos ganho potencial, nenhuma dependência
    de dado extra, e o custo de qualidade é verificável pela mesma checagem de paridade.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    origem = Path(source)
    caminho = Path(destination)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(origem),
        model_output=str(caminho),
        weight_type=QuantType.QUInt8,
    )

    logger.info(
        "variante INT8 gerada: %s (%.2f MB, %.0f%% do original)",
        caminho,
        caminho.stat().st_size / 1e6,
        100 * caminho.stat().st_size / origem.stat().st_size,
    )
    return caminho


def main() -> None:
    """Reexporta o modelo publicado, fora do pipeline de treino.

    O caminho normal é a exportação sair da DAG, a jusante da promoção. Este comando existe
    para regerar o ``.onnx`` sem retreinar — depois de mudar o conversor, por exemplo, ou de
    clonar o repositório com um ``model.joblib`` já treinado.
    """
    from triagem.config import get_settings
    from triagem.pipeline.steps import export_onnx

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    if not settings.model_path.exists():
        raise SystemExit(f"nenhum modelo publicado em {settings.model_path} — rode `make train`")

    artefatos = export_onnx(settings.model_path, settings.onnx_path, settings.onnx_int8_path)
    logger.info("artefatos gerados: %s", artefatos)


if __name__ == "__main__":
    main()
