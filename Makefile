.PHONY: help install lint format test data train evaluate bench api docker-build docker-run clean

help:             ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:          ## Instala dependências e hooks
	poetry install
	poetry run pre-commit install

lint:             ## Verifica estilo e formatação
	poetry run ruff check .
	poetry run ruff format --check .

format:           ## Aplica formatação
	poetry run ruff check --fix .
	poetry run ruff format .

test:             ## Roda a suíte com cobertura
	poetry run pytest -ra --cov=triagem --cov-report=term-missing

data:             ## Baixa o corpus público
	poetry run python -m triagem.data.download

train:            ## Treina os candidatos, avalia e promove o campeão
	poetry run python -m triagem.pipeline.training

evaluate:         ## Avalia o modelo e grava metrics/
	poetry run python -m triagem.model.evaluate

bench:            ## Mede a latência de inferência
	poetry run python -m triagem.bench.latency

api:              ## Sobe a API local com reload
	poetry run uvicorn triagem.api.main:app --reload --port 8000

docker-build:     ## Constrói a imagem (exige `make train` antes)
	docker build -t triagem-api:local .

docker-run:       ## Sobe a API em container
	docker run --rm -p 8000:8000 triagem-api:local

# O Airflow roda com o UID do host para que os artefatos gravados nos volumes montados
# (data/, models/, metrics/) pertençam a quem executou, e não a uid 50000.
export AIRFLOW_UID := $(shell id -u)

airflow-up:       ## Sobe o Airflow (http://localhost:8080, admin/admin)
	mkdir -p data/raw data/interim models/candidates metrics
	docker compose -f docker-compose.airflow.yml up -d --build

airflow-down:     ## Derruba o Airflow
	docker compose -f docker-compose.airflow.yml down

# Usa `exec`, não `run`: o banco de metadados do Airflow vive dentro do container que o
# `standalone` migrou no startup. Um container novo subiria sem esquema e falharia.
airflow-test:     ## Executa a DAG de treino de ponta a ponta (exige airflow-up antes)
	docker compose -f docker-compose.airflow.yml exec -T airflow \
		airflow dags test triagem_training

clean:            ## Remove caches locais
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
