"""Carga e particionamento do corpus.

As classes são desbalanceadas (de 1.195 a 3.844 amostras), então o split de validação é
estratificado — sem isso a classe minoritária ficaria sub-representada e o f1-macro
mediria ruído em vez de generalização.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLUMN = "medical_abstract"
LABEL_COLUMN = "condition_label"

CONDITION_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}


def load_split(csv_path: Path) -> pd.DataFrame:
    """Lê um CSV do corpus, validando o schema e descartando linhas inúteis."""
    frame = pd.read_csv(csv_path)

    ausentes = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if ausentes:
        raise ValueError(f"colunas ausentes em {csv_path.name}: {sorted(ausentes)}")

    frame = frame.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    frame = frame[frame[TEXT_COLUMN].astype(str).str.strip().astype(bool)]
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(int)

    return frame[[LABEL_COLUMN, TEXT_COLUMN]].reset_index(drop=True)


def split_train_validation(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa uma partição de validação estratificada pelo rótulo."""
    treino, validacao = train_test_split(
        frame,
        test_size=validation_size,
        random_state=seed,
        stratify=frame[LABEL_COLUMN],
    )
    return treino.reset_index(drop=True), validacao.reset_index(drop=True)
