"""Obtenção do corpus público de laudos médicos.

O corpus é servido como CSV estático, então o download é idempotente: arquivos já
presentes em disco são reaproveitados, o que mantém treino e CI reprodutíveis sem
depender da rede a cada execução.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"

FILES: dict[str, str] = {
    "train": "medical_tc_train.csv",
    "test": "medical_tc_test.csv",
    "labels": "medical_tc_labels.csv",
}

TIMEOUT_SECONDS = 120.0


def download_dataset(dest_dir: Path, *, force: bool = False) -> dict[str, Path]:
    """Baixa o corpus para ``dest_dir`` e devolve o caminho de cada arquivo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for key, filename in FILES.items():
        target = dest_dir / filename

        if target.exists() and not force:
            logger.info("reaproveitando %s", target)
            paths[key] = target
            continue

        url = f"{BASE_URL}/{filename}"
        logger.info("baixando %s", url)
        response = httpx.get(url, timeout=TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
        target.write_bytes(response.content)
        paths[key] = target

    return paths


def main() -> None:
    from triagem.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = download_dataset(get_settings().data_dir / "raw")
    for key, path in paths.items():
        logger.info("%s -> %s (%.1f KB)", key, path, path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
