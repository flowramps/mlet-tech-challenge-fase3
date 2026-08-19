# Build em dois estágios: o estágio de dependências carrega Poetry e ferramentas de
# compilação; o estágio final leva só o virtualenv pronto e o código, o que reduz a
# superfície da imagem e o tempo de subida do container.

FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.3.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

RUN poetry install --only main --no-interaction


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    TRIAGEM_MODELS_DIR=/app/models

WORKDIR /app

# Usuário sem privilégios: um comprometimento do processo de inferência não vira root
# dentro do container.
RUN useradd --create-home --uid 1000 triagem

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

# Os dois artefatos entram na imagem, somando menos de 0,6 MB. Não é redundância: com ambos
# presentes, `TRIAGEM_MODEL_BACKEND` alterna o motor de inferência sem reconstruir a imagem,
# que é o que permite comparar os backends no Grafana medindo o serviço de verdade — e o que
# dá um caminho de volta imediato se o backend otimizado apresentar problema.
COPY models/model.joblib ./models/model.joblib
COPY models/model.onnx ./models/model.onnx

RUN chown -R triagem:triagem /app
USER triagem

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "triagem.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
