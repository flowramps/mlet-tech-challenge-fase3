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
    model_backend: str = "sklearn"
    model_version: str = "1.0.0"

    min_f1_macro: float = 0.0
    random_seed: int = 42
    validation_size: float = 0.2

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
