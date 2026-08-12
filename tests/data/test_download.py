from pathlib import Path

import httpx
import pytest

from triagem.data.download import FILES, download_dataset


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_baixa_os_tres_arquivos_quando_ausentes(tmp_path: Path, monkeypatch):
    chamadas: list[str] = []

    def _fake_get(url: str, **_kwargs):
        chamadas.append(url)
        return _FakeResponse(b"condition_label,medical_abstract\n1,texto\n")

    monkeypatch.setattr(httpx, "get", _fake_get)

    paths = download_dataset(tmp_path)

    assert len(chamadas) == 3
    assert set(paths) == {"train", "test", "labels"}
    assert paths["train"] == tmp_path / "medical_tc_train.csv"
    assert paths["train"].read_bytes().startswith(b"condition_label")


def test_nao_acessa_a_rede_quando_o_arquivo_ja_existe(tmp_path: Path, monkeypatch):
    for filename in FILES.values():
        (tmp_path / filename).write_text("condition_label,medical_abstract\n")

    def _explode(*_args, **_kwargs):
        raise AssertionError("cache falhou: não deveria acessar a rede")

    monkeypatch.setattr(httpx, "get", _explode)

    paths = download_dataset(tmp_path)

    assert set(paths) == {"train", "test", "labels"}


def test_force_rebaixa_mesmo_com_cache(tmp_path: Path, monkeypatch):
    for filename in FILES.values():
        (tmp_path / filename).write_text("antigo")

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _FakeResponse(b"novo"))

    paths = download_dataset(tmp_path, force=True)

    assert paths["train"].read_text() == "novo"


def test_propaga_erro_http(tmp_path: Path, monkeypatch):
    class _Erro(_FakeResponse):
        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("404", request=None, response=None)

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _Erro(b""))

    with pytest.raises(httpx.HTTPStatusError):
        download_dataset(tmp_path)
