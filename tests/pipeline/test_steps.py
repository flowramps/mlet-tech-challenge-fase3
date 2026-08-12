from pathlib import Path

import pandas as pd
import pytest

from triagem.data.prepare import LABEL_COLUMN, TEXT_COLUMN
from triagem.pipeline.steps import (
    QualityGateError,
    ingest,
    prepare,
    publish,
    select_and_evaluate,
    should_publish,
    train_candidates,
)

CONDICOES = {
    1: "tumor maligno com metástase óssea difusa e perda de peso acentuada",
    4: "infarto agudo do miocárdio com supradesnivelamento do segmento ST",
}


def _corpus(linhas_por_classe: int = 40) -> pd.DataFrame:
    registros = []
    for rotulo, texto in CONDICOES.items():
        for i in range(linhas_por_classe):
            registros.append({LABEL_COLUMN: rotulo, TEXT_COLUMN: f"{texto} caso {i}"})
    return pd.DataFrame(registros)


@pytest.fixture
def corpus_csv(tmp_path: Path) -> Path:
    caminho = tmp_path / "corpus.csv"
    _corpus().to_csv(caminho, index=False)
    return caminho


def test_ingest_devolve_caminhos_como_texto(tmp_path: Path, monkeypatch):
    """XCom do Airflow serializa o retorno: Path não sobrevive, str sim."""
    import triagem.pipeline.steps as steps

    def _fake_download(dest_dir, *, force=False):
        return {"train": dest_dir / "a.csv", "test": dest_dir / "b.csv"}

    monkeypatch.setattr(steps, "download_dataset", _fake_download)

    resultado = ingest(tmp_path)

    assert all(isinstance(valor, str) for valor in resultado.values())
    assert resultado["train"].endswith("a.csv")


def test_prepare_grava_as_duas_particoes(corpus_csv: Path, tmp_path: Path):
    resultado = prepare(str(corpus_csv), tmp_path / "interim", validation_size=0.25, seed=42)

    treino = pd.read_csv(resultado["train"])
    validacao = pd.read_csv(resultado["validation"])

    assert len(treino) == 60
    assert len(validacao) == 20
    assert set(treino.columns) == {LABEL_COLUMN, TEXT_COLUMN}


def test_prepare_devolve_caminhos_como_texto(corpus_csv: Path, tmp_path: Path):
    resultado = prepare(str(corpus_csv), tmp_path / "interim")
    assert all(isinstance(valor, str) for valor in resultado.values())


def test_train_candidates_pontua_todos_os_candidatos(corpus_csv: Path, tmp_path: Path):
    particoes = prepare(str(corpus_csv), tmp_path / "interim", seed=42)

    scores = train_candidates(
        particoes["train"], particoes["validation"], tmp_path / "candidatos", seed=42
    )

    assert set(scores) == {"random_forest", "logistic_regression"}
    assert all(0.0 <= valor <= 1.0 for valor in scores.values())
    assert (tmp_path / "candidatos" / "random_forest.joblib").exists()
    assert (tmp_path / "candidatos" / "logistic_regression.joblib").exists()


def test_select_and_evaluate_resume_o_campeao(corpus_csv: Path, tmp_path: Path):
    particoes = prepare(str(corpus_csv), tmp_path / "interim", seed=42)
    scores = train_candidates(
        particoes["train"], particoes["validation"], tmp_path / "candidatos", seed=42
    )

    resumo = select_and_evaluate(
        scores, tmp_path / "candidatos", str(corpus_csv), tmp_path / "metricas"
    )

    assert resumo["champion"] in scores
    assert 0.0 <= resumo["f1_macro"] <= 1.0
    assert Path(resumo["candidate_path"]).exists()
    assert (tmp_path / "metricas" / "metrics.json").exists()
    assert (tmp_path / "metricas" / "model_selection.json").exists()


def test_should_publish_compara_com_o_piso():
    assert should_publish(0.60, 0.53) is True
    assert should_publish(0.53, 0.53) is True
    assert should_publish(0.52, 0.53) is False


def test_publish_copia_o_campeao_quando_aprovado(tmp_path: Path):
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo")
    destino = tmp_path / "publicado" / "model.joblib"

    resultado = publish(str(candidato), destino, f1_macro=0.60, minimum=0.53)

    assert Path(resultado) == destino
    assert destino.read_bytes() == b"modelo"


def test_publish_falha_quando_o_modelo_regride(tmp_path: Path):
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo ruim")
    destino = tmp_path / "publicado" / "model.joblib"

    with pytest.raises(QualityGateError, match="0.52"):
        publish(str(candidato), destino, f1_macro=0.52, minimum=0.53)

    assert not destino.exists()


def test_publish_preserva_o_modelo_anterior_ao_reprovar(tmp_path: Path):
    """Um retreino ruim não pode derrubar o modelo que já está servindo."""
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo bom em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo ruim")

    with pytest.raises(QualityGateError):
        publish(str(candidato), destino, f1_macro=0.10, minimum=0.53)

    assert destino.read_bytes() == b"modelo bom em producao"
