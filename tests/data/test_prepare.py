from pathlib import Path

import pandas as pd
import pytest

from triagem.data.prepare import (
    CONDITION_NAMES,
    LABEL_COLUMN,
    TEXT_COLUMN,
    load_split,
    split_train_validation,
)


def _csv(tmp_path: Path, frame: pd.DataFrame) -> Path:
    destino = tmp_path / "amostra.csv"
    frame.to_csv(destino, index=False)
    return destino


def test_condition_names_cobre_os_cinco_rotulos():
    assert set(CONDITION_NAMES) == {1, 2, 3, 4, 5}
    assert CONDITION_NAMES[4] == "cardiovascular diseases"


def test_load_split_le_colunas_esperadas(tmp_path: Path):
    caminho = _csv(tmp_path, pd.DataFrame({LABEL_COLUMN: [1, 2], TEXT_COLUMN: ["ab", "cd"]}))
    frame = load_split(caminho)
    assert list(frame.columns) == [LABEL_COLUMN, TEXT_COLUMN]
    assert len(frame) == 2


def test_load_split_descarta_texto_vazio_ou_nulo(tmp_path: Path):
    caminho = _csv(
        tmp_path,
        pd.DataFrame({LABEL_COLUMN: [1, 2, 3], TEXT_COLUMN: ["válido", "   ", None]}),
    )
    frame = load_split(caminho)
    assert len(frame) == 1
    assert frame.loc[0, TEXT_COLUMN] == "válido"


def test_load_split_falha_com_coluna_ausente(tmp_path: Path):
    caminho = _csv(tmp_path, pd.DataFrame({"outra": [1]}))
    with pytest.raises(ValueError, match="colunas ausentes"):
        load_split(caminho)


def test_split_preserva_proporcao_das_classes():
    frame = pd.DataFrame(
        {LABEL_COLUMN: [1] * 80 + [2] * 20, TEXT_COLUMN: [f"laudo {i}" for i in range(100)]}
    )
    treino, validacao = split_train_validation(frame, validation_size=0.2, seed=42)

    assert len(treino) == 80
    assert len(validacao) == 20
    assert (validacao[LABEL_COLUMN] == 1).sum() == 16


def test_split_e_deterministico_com_a_mesma_seed():
    frame = pd.DataFrame(
        {LABEL_COLUMN: [1, 2] * 50, TEXT_COLUMN: [f"laudo {i}" for i in range(100)]}
    )
    primeiro, _ = split_train_validation(frame, seed=42)
    segundo, _ = split_train_validation(frame, seed=42)
    assert primeiro[TEXT_COLUMN].tolist() == segundo[TEXT_COLUMN].tolist()
