"""Backend de inferência sobre o ONNX Runtime.

Serve o mesmo modelo que o backend scikit-learn, por um motor diferente: o pipeline inteiro
— tokenização, TF-IDF e classificador — roda como um grafo compilado, sem passar pelo
interpretador Python a cada laudo. A troca é uma variável de ambiente, e a suíte exige
paridade de rótulos entre os dois, então "mais rápido" nunca significa "outra resposta".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from triagem.inference.base import Prediction


class OnnxClassifier:
    """Executa o grafo exportado por ``triagem.model.export_onnx``."""

    name = "onnx"

    def __init__(
        self,
        session,
        labels_by_id: dict[int, str],
        version: str,
        model_type: str = "desconhecido",
    ) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._labels_by_id = labels_by_id
        self.version = version
        self.model_type = model_type

    @classmethod
    def load(cls, path: Path) -> OnnxClassifier:
        """Abre uma sessão de inferência e recupera os metadados de dentro do grafo."""
        import onnxruntime

        opcoes = onnxruntime.SessionOptions()
        # A carga é de um laudo por vez, não de lotes grandes: paralelizar dentro do
        # operador só adiciona sincronização entre threads a um trabalho que dura
        # microssegundos. Uma thread por inferência deixa o servidor livre para atender
        # requisições concorrentes, que é onde o paralelismo de fato rende aqui.
        opcoes.intra_op_num_threads = 1
        opcoes.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        sessao = onnxruntime.InferenceSession(
            str(path), sess_options=opcoes, providers=["CPUExecutionProvider"]
        )

        metadados = sessao.get_modelmeta().custom_metadata_map
        rotulos = {
            int(chave): valor
            for chave, valor in json.loads(metadados.get("triagem_labels", "{}")).items()
        }
        return cls(
            sessao,
            rotulos,
            metadados.get("triagem_version", "desconhecida"),
            metadados.get("triagem_model_type", "desconhecido"),
        )

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        entradas = list(texts)
        if not entradas:
            return []

        # O grafo declara a entrada como (N, 1) — uma coluna de strings, que é a forma que o
        # operador de vetorização espera.
        lote = np.array(entradas, dtype=object).reshape(-1, 1)
        _, probabilidades = self._session.run(None, {self._input_name: lote})

        probabilidades = np.asarray(probabilidades)
        indices = probabilidades.argmax(axis=1)

        return [
            Prediction(
                condition=self._labels_by_id[int(self._session_classes()[indice])],
                confidence=float(linha[indice]),
            )
            for linha, indice in zip(probabilidades, indices, strict=True)
        ]

    def _session_classes(self) -> np.ndarray:
        """Rótulos numéricos na ordem das colunas de probabilidade.

        O conversor preserva a ordem de ``classes_`` do scikit-learn, que é a ordem crescente
        dos rótulos — a mesma que ``sorted`` reproduz a partir do mapa de nomes.
        """
        if not hasattr(self, "_classes_cache"):
            self._classes_cache = np.array(sorted(self._labels_by_id))
        return self._classes_cache
