"""Configuração central da aplicação, resolvida por variáveis de ambiente."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros de execução. Prefixo de ambiente: ``TRIAGEM_``."""

    model_config = SettingsConfigDict(
        env_prefix="TRIAGEM_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    metrics_dir: Path = PROJECT_ROOT / "metrics"

    model_filename: str = "model.joblib"
    onnx_filename: str = "model.onnx"
    onnx_int8_filename: str = "model.int8.onnx"

    # Padrão `onnx` desde a Etapa 4: é o backend mais rápido (6,7x no p50) e a suíte exige
    # paridade de rótulos com o scikit-learn, então a troca não muda a resposta. `sklearn`
    # segue disponível e a imagem carrega os dois artefatos — alternar em tempo de execução
    # é o que permite comparar os backends no Grafana, medindo o serviço real.
    model_backend: str = "onnx"
    model_version: str = "1.0.0"

    # Piso de qualidade para promover um modelo recém-treinado. Calibrado em 0,53 a partir
    # do f1-macro de validação do campeão (0,5869), com margem de 0,05 para absorver a
    # variação natural entre retreinos sem transformar o gate em enfeite que sempre passa.
    min_f1_macro: float = 0.53

    # Piso de recall na prioridade "alta" (cardiovascular + nervous system) — a métrica de
    # negócio da triagem, não só a técnica: rebaixar um caso realmente urgente pesa mais do
    # que confundir duas condições que já dariam na mesma prioridade. Calibrado em 0,72 a
    # partir do recall do campeão no teste (0,7739), mesma margem de 0,05 do min_f1_macro.
    min_priority_recall_alta: float = 0.72

    random_seed: int = 42
    validation_size: float = 0.2

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def onnx_path(self) -> Path:
        return self.models_dir / self.onnx_filename

    @property
    def onnx_int8_path(self) -> Path:
        return self.models_dir / self.onnx_int8_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
