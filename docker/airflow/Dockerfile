# Airflow com as dependências de ML do projeto. A imagem oficial slim não traz
# scikit-learn nem pandas; instalá-los aqui resolve para as mesmas versões do venv do
# projeto, então o modelo treinado pela DAG é binariamente compatível com o que a API serve.
#
# As faixas de versão precisam acompanhar as do `pyproject.toml`. A DAG não roda com o
# venv do projeto — ela importa `src/` montado como volume sobre as dependências desta
# imagem —, então uma dependência nova no projeto que a tarefa use tem que ser adicionada
# aqui também, ou a tarefa falha só em tempo de execução, dentro do container.
FROM apache/airflow:slim-3.3.1-python3.12

USER airflow

RUN pip install --no-cache-dir \
    "scikit-learn>=1.5,<2.0" \
    "pandas>=2.2,<3.0" \
    "joblib>=1.4,<2.0" \
    "httpx>=0.27,<1.0" \
    "onnxruntime>=1.20,<2.0" \
    "skl2onnx>=1.17,<2.0"

# Senha fixa do admin na UI. O `standalone` do Airflow 3 autentica pelo SimpleAuthManager,
# que sorteia uma senha por container e a grava em `...json.generated` — as variáveis
# `_AIRFLOW_WWW_USER_*` são do Airflow 2 e ele ignora em silêncio. O `init()` do manager só
# sorteia senha para usuário ausente do arquivo, então semeá-lo com o admin fixa a
# credencial e torna o passo a passo do README reproduzível.
#
# O manager abre o arquivo em modo `r+`, ou seja, precisa de escrita mesmo só para ler. O
# container roda como `${AIRFLOW_UID}:0` com UID variável por máquina, então quem garante a
# escrita de forma portável é o bit de grupo — o GID é sempre 0.
COPY --chown=airflow:0 --chmod=664 \
    docker/airflow/simple_auth_manager_passwords.json \
    /opt/airflow/simple_auth_manager_passwords.json
