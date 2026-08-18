import json
from pathlib import Path

import pandas as pd
import pytest

from triagem.data.prepare import LABEL_COLUMN, TEXT_COLUMN
from triagem.pipeline.steps import (
    ModelNotPromoted,
    QualityGateError,
    evaluate_incumbent,
    ingest,
    prepare,
    promote,
    publish,
    select_and_evaluate,
    should_promote,
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
    assert 0.0 <= resumo["priority_recall_alta"] <= 1.0
    assert Path(resumo["candidate_path"]).exists()
    assert (tmp_path / "metricas" / "candidate_metrics.json").exists()
    assert (tmp_path / "metricas" / "model_selection.json").exists()
    # `metrics.json` descreve o modelo *publicado*: avaliar um candidato não pode reescrevê-lo
    # antes de a promoção ser decidida, senão o arquivo passa a mentir sobre o que está no ar.
    assert not (tmp_path / "metricas" / "metrics.json").exists()


GATE_PADRAO = {"min_f1_macro": 0.53, "min_priority_recall_alta": 0.5}


def test_should_promote_compara_com_o_piso_sem_incumbente():
    assert should_promote(0.60, 0.80, None, None, **GATE_PADRAO) is True
    assert should_promote(0.53, 0.80, None, None, **GATE_PADRAO) is True
    assert should_promote(0.52, 0.80, None, None, **GATE_PADRAO) is False


def test_should_promote_exige_piso_de_recall_de_prioridade_alta():
    # acima do piso de f1-macro, mas abaixo do piso da métrica de negócio: não promove
    assert should_promote(0.60, 0.49, None, None, **GATE_PADRAO) is False
    assert should_promote(0.60, 0.50, None, None, **GATE_PADRAO) is True


def test_should_promote_exige_superar_o_incumbente():
    # acima dos dois pisos, mas não supera o f1-macro do modelo em produção: não promove
    assert should_promote(0.55, 0.80, 0.60, 0.80, **GATE_PADRAO) is False
    assert should_promote(0.60, 0.80, 0.60, 0.80, **GATE_PADRAO) is False
    assert should_promote(0.61, 0.80, 0.60, 0.80, **GATE_PADRAO) is True


def test_should_promote_nao_permite_regredir_o_recall_de_prioridade_alta():
    """f1-macro melhor não basta: a métrica de negócio não pode piorar em relação à produção."""
    assert should_promote(0.65, 0.70, 0.60, 0.80, **GATE_PADRAO) is False
    # empatar no recall de prioridade alta é aceitável — só não pode piorar
    assert should_promote(0.65, 0.80, 0.60, 0.80, **GATE_PADRAO) is True


def test_evaluate_incumbent_sem_modelo_publicado(tmp_path: Path):
    assert evaluate_incumbent(tmp_path / "inexistente.joblib", "irrelevante.csv") is None


def test_evaluate_incumbent_mede_o_modelo_publicado(corpus_csv: Path, tmp_path: Path):
    particoes = prepare(str(corpus_csv), tmp_path / "interim", seed=42)
    train_candidates(particoes["train"], particoes["validation"], tmp_path / "candidatos", seed=42)

    modelo_publicado = tmp_path / "candidatos" / "logistic_regression.joblib"
    incumbente = evaluate_incumbent(modelo_publicado, str(corpus_csv))

    assert incumbente is not None
    assert 0.0 <= incumbente["f1_macro"] <= 1.0
    assert 0.0 <= incumbente["priority_recall_alta"] <= 1.0


def test_publish_copia_o_campeao_quando_aprovado(tmp_path: Path):
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo")
    destino = tmp_path / "publicado" / "model.joblib"

    resultado = publish(
        str(candidato),
        destino,
        f1_macro=0.60,
        priority_recall_alta=0.80,
        baseline_f1_macro=None,
        baseline_priority_recall_alta=None,
        **GATE_PADRAO,
    )

    assert Path(resultado) == destino
    assert destino.read_bytes() == b"modelo"


def test_publish_falha_quando_o_modelo_regride_abaixo_do_piso(tmp_path: Path):
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo ruim")
    destino = tmp_path / "publicado" / "model.joblib"

    with pytest.raises(QualityGateError, match="0.52"):
        publish(
            str(candidato),
            destino,
            f1_macro=0.52,
            priority_recall_alta=0.80,
            baseline_f1_macro=None,
            baseline_priority_recall_alta=None,
            **GATE_PADRAO,
        )

    assert not destino.exists()


def test_publish_falha_quando_recall_de_prioridade_alta_fica_abaixo_do_piso(tmp_path: Path):
    """f1-macro acima do piso não basta: a métrica de negócio também precisa passar."""
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo com recall ruim em casos urgentes")
    destino = tmp_path / "publicado" / "model.joblib"

    with pytest.raises(QualityGateError, match="prioridade alta"):
        publish(
            str(candidato),
            destino,
            f1_macro=0.60,
            priority_recall_alta=0.40,
            baseline_f1_macro=None,
            baseline_priority_recall_alta=None,
            **GATE_PADRAO,
        )

    assert not destino.exists()


def test_publish_sinaliza_nao_promocao_quando_nao_supera_o_incumbente(tmp_path: Path):
    """Não superar a produção é 'nada a promover', não 'pipeline quebrado'.

    O candidato passou nos dois pisos absolutos — ele é um modelo utilizável, só não é
    melhor que o que já está no ar. Sinalizar isso como falha faria a DAG semanal sobre um
    corpus estático ficar vermelha para sempre, já que o retreino reproduz o incumbente.
    """
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo bom em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo um pouco pior")

    with pytest.raises(ModelNotPromoted, match="produção"):
        publish(
            str(candidato),
            destino,
            f1_macro=0.55,
            priority_recall_alta=0.80,
            baseline_f1_macro=0.60,
            baseline_priority_recall_alta=0.80,
            **GATE_PADRAO,
        )

    assert destino.read_bytes() == b"modelo bom em producao"


def test_publish_sinaliza_nao_promocao_quando_recall_de_prioridade_alta_regride(tmp_path: Path):
    """f1-macro melhor não salva um candidato que piora a segurança da fila de triagem."""
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo bom em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo com f1 melhor mas recall de alta pior")

    with pytest.raises(ModelNotPromoted, match="prioridade alta"):
        publish(
            str(candidato),
            destino,
            f1_macro=0.65,
            priority_recall_alta=0.70,
            baseline_f1_macro=0.60,
            baseline_priority_recall_alta=0.80,
            **GATE_PADRAO,
        )

    assert destino.read_bytes() == b"modelo bom em producao"


def test_nao_promocao_nao_e_confundida_com_reprovacao_de_qualidade(tmp_path: Path):
    """As duas condições precisam ser distinguíveis por tipo, não só pela mensagem.

    A DAG converte uma em `skip` e deixa a outra falhar o run — se `ModelNotPromoted` fosse
    subclasse de `QualityGateError`, um `except QualityGateError` engoliria as duas.
    """
    assert not issubclass(ModelNotPromoted, QualityGateError)
    assert not issubclass(QualityGateError, ModelNotPromoted)


def test_publish_prioriza_o_piso_absoluto_sobre_a_nao_promocao(tmp_path: Path):
    """Candidato ruim E pior que a produção é falha de qualidade, não mero 'não promover'.

    A condição mais severa manda: um modelo abaixo do piso indica que algo quebrou no treino
    e alguém precisa olhar, então o run tem que falhar de verdade.
    """
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo bom em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo quebrado")

    with pytest.raises(QualityGateError):
        publish(
            str(candidato),
            destino,
            f1_macro=0.10,
            priority_recall_alta=0.10,
            baseline_f1_macro=0.60,
            baseline_priority_recall_alta=0.80,
            **GATE_PADRAO,
        )

    assert destino.read_bytes() == b"modelo bom em producao"


def test_publish_preserva_o_modelo_anterior_ao_reprovar(tmp_path: Path):
    """Um retreino ruim não pode derrubar o modelo que já está servindo."""
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo bom em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo ruim")

    with pytest.raises(QualityGateError):
        publish(
            str(candidato),
            destino,
            f1_macro=0.10,
            priority_recall_alta=0.10,
            baseline_f1_macro=None,
            baseline_priority_recall_alta=None,
            **GATE_PADRAO,
        )

    assert destino.read_bytes() == b"modelo bom em producao"


def _resumo(
    f1_macro: float, priority_recall_alta: float, candidate_path: Path
) -> dict[str, object]:
    return {
        "champion": "logistic_regression",
        "f1_macro": f1_macro,
        "priority_recall_alta": priority_recall_alta,
        "candidate_path": str(candidate_path),
    }


def test_promote_registra_sucesso_no_historico(tmp_path: Path):
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo")
    destino = tmp_path / "publicado" / "model.joblib"

    caminho = promote(
        _resumo(0.60, 0.80, candidato), None, destino, tmp_path / "metricas", **GATE_PADRAO
    )

    assert Path(caminho) == destino
    linhas = (
        (tmp_path / "metricas" / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(linhas) == 1

    entrada = json.loads(linhas[0])
    assert entrada["champion"] == "logistic_regression"
    assert entrada["f1_macro"] == 0.60
    assert entrada["promoted"] is True
    assert entrada["rejection_reasons"] == []


def test_promote_registra_reprovacao_e_ainda_levanta(tmp_path: Path):
    """O histórico precisa capturar runs reprovados também, não só os promovidos."""
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo ruim")
    destino = tmp_path / "model.joblib"

    with pytest.raises(QualityGateError):
        promote(_resumo(0.10, 0.10, candidato), None, destino, tmp_path / "metricas", **GATE_PADRAO)

    entrada = json.loads(
        (tmp_path / "metricas" / "training_history.jsonl").read_text(encoding="utf-8").strip()
    )
    assert entrada["promoted"] is False
    assert entrada["rejection_reasons"]
    assert not destino.exists()


def test_promote_publica_as_metricas_junto_com_o_modelo(tmp_path: Path):
    """Promover o artefato e promover as métricas dele é a mesma operação.

    `metrics.json` é a resposta a "quanto vale o modelo que está atendendo?" — se ele fosse
    escrito na avaliação do candidato, um run que não promove deixaria o arquivo descrevendo
    um modelo que nunca entrou no ar.
    """
    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"modelo")
    metrics_dir = tmp_path / "metricas"
    metrics_dir.mkdir()
    (metrics_dir / "candidate_metrics.json").write_text(
        json.dumps({"f1_macro": 0.60, "priority_recall_alta": 0.80}), encoding="utf-8"
    )

    promote(
        _resumo(0.60, 0.80, candidato),
        None,
        tmp_path / "publicado" / "model.joblib",
        metrics_dir,
        **GATE_PADRAO,
    )

    publicadas = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    assert publicadas["f1_macro"] == 0.60


def test_promote_nao_toca_nas_metricas_publicadas_quando_nao_promove(tmp_path: Path):
    """Run que não promove deixa `metrics.json` como estava — ele descreve o que segue no ar."""
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"candidato identico")

    metrics_dir = tmp_path / "metricas"
    metrics_dir.mkdir()
    (metrics_dir / "metrics.json").write_text(
        json.dumps({"f1_macro": 0.60, "origem": "modelo em producao"}), encoding="utf-8"
    )
    (metrics_dir / "candidate_metrics.json").write_text(
        json.dumps({"f1_macro": 0.60, "origem": "candidato"}), encoding="utf-8"
    )

    with pytest.raises(ModelNotPromoted):
        promote(
            _resumo(0.60, 0.80, candidato),
            {"f1_macro": 0.60, "priority_recall_alta": 0.80},
            destino,
            metrics_dir,
            **GATE_PADRAO,
        )

    publicadas = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    assert publicadas["origem"] == "modelo em producao"


def test_promote_registra_no_historico_quando_apenas_nao_promove(tmp_path: Path):
    """Um run que não promove por empate ainda precisa deixar rastro, com o motivo."""
    destino = tmp_path / "model.joblib"
    destino.write_bytes(b"modelo em producao")

    candidato = tmp_path / "candidato.joblib"
    candidato.write_bytes(b"candidato identico ao incumbente")

    incumbente = {"f1_macro": 0.60, "priority_recall_alta": 0.80}

    with pytest.raises(ModelNotPromoted):
        promote(
            _resumo(0.60, 0.80, candidato),
            incumbente,
            destino,
            tmp_path / "metricas",
            **GATE_PADRAO,
        )

    entrada = json.loads(
        (tmp_path / "metricas" / "training_history.jsonl").read_text(encoding="utf-8").strip()
    )
    assert entrada["promoted"] is False
    assert entrada["rejection_reasons"]
    assert destino.read_bytes() == b"modelo em producao"


def test_promote_acumula_execucoes_sem_sobrescrever(tmp_path: Path):
    destino = tmp_path / "model.joblib"
    metrics_dir = tmp_path / "metricas"

    candidato1 = tmp_path / "c1.joblib"
    candidato1.write_bytes(b"modelo 1")
    promote(_resumo(0.60, 0.80, candidato1), None, destino, metrics_dir, **GATE_PADRAO)

    candidato2 = tmp_path / "c2.joblib"
    candidato2.write_bytes(b"modelo 2")
    promote(
        _resumo(0.65, 0.85, candidato2),
        {"f1_macro": 0.60, "priority_recall_alta": 0.80},
        destino,
        metrics_dir,
        **GATE_PADRAO,
    )

    linhas = (
        (metrics_dir / "training_history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(linhas) == 2
    assert all(json.loads(linha)["promoted"] for linha in linhas)
