"""Contrato de autenticação da UI do Airflow.

O `standalone` do Airflow 3 autentica pelo `SimpleAuthManager`, que sorteia uma senha
aleatória por container e a grava em `simple_auth_manager_passwords.json.generated`. As
variáveis `_AIRFLOW_WWW_USER_USERNAME` / `_AIRFLOW_WWW_USER_PASSWORD` são um mecanismo do
Airflow 2 e ele ignora em silêncio — quem seguisse o README esbarrava num 401.

A correção é pré-semear o arquivo de senhas: o `init()` do manager só sorteia senha para
usuário ausente do arquivo. Estes testes prendem as três pontas dessa correção (arquivo
semeado, compose apontando para ele, imagem copiando com permissão de escrita), porque
qualquer uma delas sozinha volta a produzir a senha aleatória.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.airflow.yml"
DOCKERFILE = ROOT / "docker" / "airflow" / "Dockerfile"
SEED = ROOT / "docker" / "airflow" / "simple_auth_manager_passwords.json"

# Caminho dentro da imagem. Fora de qualquer volume montado, então sobrevive ao container.
SEED_IN_IMAGE = "/opt/airflow/simple_auth_manager_passwords.json"

# Credencial de desenvolvimento, igual à do Grafana e à documentada no README. O
# SimpleAuthManager guarda senha em texto puro e é dev-only por design — não há segredo
# a proteger aqui, e fixá-la é o que torna o passo a passo reproduzível.
DEV_USER = "admin"
DEV_PASSWORD = "admin"


@pytest.fixture(scope="module")
def compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_arquivo_de_senhas_semeia_o_admin():
    """Sem o usuário já presente no arquivo, o `init()` sorteia uma senha nova."""
    assert SEED.is_file(), f"{SEED} não existe"
    assert json.loads(SEED.read_text(encoding="utf-8")) == {DEV_USER: DEV_PASSWORD}


def test_compose_aponta_para_o_arquivo_de_senhas(compose_text: str):
    """Sem a configuração, o manager usa o caminho `.generated` e sorteia a senha."""
    assert "AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE" in compose_text
    assert SEED_IN_IMAGE in compose_text


def test_compose_nao_promete_as_variaveis_do_airflow_2(compose_text: str):
    """`_AIRFLOW_WWW_USER_*` não tem efeito no Airflow 3 e sugere uma senha que não vale."""
    assert "_AIRFLOW_WWW_USER_USERNAME" not in compose_text
    assert "_AIRFLOW_WWW_USER_PASSWORD" not in compose_text


def test_imagem_copia_o_arquivo_com_escrita_para_o_grupo():
    """O manager abre o arquivo em modo `r+`: só de leitura, a UI quebra ao autenticar.

    O container roda como `${AIRFLOW_UID}:0`, um UID que varia por máquina. O que dá
    escrita de forma portável é o bit de grupo, já que o GID é sempre 0.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert SEED_IN_IMAGE in dockerfile, "a imagem não copia o arquivo de senhas"
    assert "--chmod=664" in dockerfile
    assert "--chown=airflow:0" in dockerfile
