from pathlib import Path

from triagem.config import Settings, get_settings


def test_settings_expoe_caminhos_absolutos():
    settings = Settings()
    assert settings.data_dir.is_absolute()
    assert settings.models_dir.is_absolute()
    assert settings.model_filename == "model.joblib"


def test_settings_default_usa_backend_onnx():
    """Desde a Etapa 4 o padrão é o backend otimizado — 6,7x mais rápido, mesma resposta."""
    assert Settings().model_backend == "onnx"


def test_settings_le_variaveis_de_ambiente(monkeypatch):
    monkeypatch.setenv("TRIAGEM_MODEL_BACKEND", "sklearn")
    monkeypatch.setenv("TRIAGEM_RANDOM_SEED", "7")
    settings = Settings()
    assert settings.model_backend == "sklearn"
    assert settings.random_seed == 7


def test_caminhos_dos_artefatos_onnx(tmp_path: Path):
    settings = Settings(models_dir=tmp_path)
    assert settings.onnx_path == tmp_path / "model.onnx"
    assert settings.onnx_int8_path == tmp_path / "model.int8.onnx"


def test_model_path_combina_diretorio_e_arquivo(tmp_path: Path):
    settings = Settings(models_dir=tmp_path)
    assert settings.model_path == tmp_path / "model.joblib"


def test_get_settings_e_cacheado():
    assert get_settings() is get_settings()
