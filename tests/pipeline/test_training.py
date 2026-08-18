from pathlib import Path

import pandas as pd
import pytest

from triagem.config import Settings
from triagem.data.prepare import LABEL_COLUMN, TEXT_COLUMN
from triagem.pipeline.steps import QualityGateError
from triagem.pipeline.training import run_pipeline

CONDICOES = {
    1: "tumor maligno com metástase óssea difusa e perda de peso acentuada",
    4: "infarto agudo do miocárdio com supradesnivelamento do segmento ST",
}


def _corpus_csv(tmp_path: Path, linhas_por_classe: int = 40) -> Path:
    registros = []
    for rotulo, texto in CONDICOES.items():
        for i in range(linhas_por_classe):
            registros.append({LABEL_COLUMN: rotulo, TEXT_COLUMN: f"{texto} caso {i}"})
    caminho = tmp_path / "corpus.csv"
    pd.DataFrame(registros).to_csv(caminho, index=False)
    return caminho


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        min_f1_macro=0.0,
        min_priority_recall_alta=0.0,
        random_seed=42,
        validation_size=0.2,
        model_version="1.0.0",
    )


@pytest.fixture
def fake_corpus(tmp_path: Path, monkeypatch):
    """Troca o download real por um corpus sintético local, reaproveitado como treino e teste.

    O mesmo truque de `test_ingest_devolve_caminhos_como_texto` em test_steps.py: substitui
    `download_dataset` no módulo `steps`, não a função original, porque é o nome que `ingest`
    de fato chama.
    """
    import triagem.pipeline.steps as steps

    caminho = _corpus_csv(tmp_path)

    def _fake_download(dest_dir, *, force=False):
        return {"train": caminho, "test": caminho, "labels": caminho}

    monkeypatch.setattr(steps, "download_dataset", _fake_download)
    return caminho


def test_run_pipeline_encadeia_as_etapas_e_promove_o_bootstrap(tmp_path: Path, fake_corpus):
    """Sem modelo publicado ainda, o pipeline local roda de ponta a ponta e promove o campeão.

    Este é o teste que faltava: `run_pipeline` só passa dicionários de uma etapa para a
    próxima (`resumo["f1_macro"]`, `incumbente["priority_recall_alta"]`, ...) — se alguma
    chave for trocada por engano, nenhuma métrica de modelo acusa isso, só rodar o código de
    verdade com uma entrada conhecida.
    """
    settings = _settings(tmp_path)

    resumo = run_pipeline(settings)

    assert resumo["baseline"] is None  # bootstrap: nenhum incumbente para comparar
    assert resumo["champion"] in {"random_forest", "logistic_regression"}
    assert 0.0 <= resumo["f1_macro"] <= 1.0
    assert 0.0 <= resumo["priority_recall_alta"] <= 1.0
    assert Path(resumo["published_path"]) == settings.model_path
    assert settings.model_path.exists()
    assert (settings.metrics_dir / "training_history.jsonl").exists()


def test_run_pipeline_nao_promove_retreino_identico_mas_conclui_sem_erro(
    tmp_path: Path, fake_corpus
):
    """Rodar o pipeline duas vezes com o mesmo dado e seed não promove a segunda vez — e isso
    é um desfecho normal, não um erro.

    Sobre um corpus estático toda execução periódica cai aqui: o retreino reproduz o
    incumbente. Se isso levantasse exceção, a DAG semanal ficaria vermelha para sempre e o
    sinal de falha deixaria de significar alguma coisa.
    """
    settings = _settings(tmp_path)

    primeiro = run_pipeline(settings)
    assert primeiro["promoted"] is True

    segundo = run_pipeline(settings)  # candidato idêntico ao incumbente

    assert segundo["promoted"] is False
    assert segundo["published_path"] is None
    assert segundo["rejection_reasons"]
    assert settings.model_path.exists()  # o modelo anterior segue publicado

    linhas = (
        (settings.metrics_dir / "training_history.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(linhas) == 2  # as duas execuções ficam registradas, promovida ou não


def test_run_pipeline_falha_quando_o_candidato_fica_abaixo_do_piso(tmp_path: Path, fake_corpus):
    """Violar um piso absoluto continua derrubando o run: aí sim algo está errado no treino."""
    settings = _settings(tmp_path).model_copy(update={"min_f1_macro": 1.1})

    with pytest.raises(QualityGateError):
        run_pipeline(settings)

    assert not settings.model_path.exists()
