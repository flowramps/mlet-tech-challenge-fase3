# Airflow com as dependências de ML do projeto. A imagem oficial slim não traz
# scikit-learn nem pandas; instalá-los aqui resolve para as mesmas versões do venv do
# projeto, então o modelo treinado pela DAG é binariamente compatível com o que a API serve.
FROM apache/airflow:slim-3.3.1-python3.12

USER airflow

RUN pip install --no-cache-dir \
    "scikit-learn>=1.5,<2.0" \
    "pandas>=2.2,<3.0" \
    "joblib>=1.4,<2.0" \
    "httpx>=0.27,<1.0"
