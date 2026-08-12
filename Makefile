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

train:            ## Treina e salva o modelo
	poetry run python -m triagem.model.train

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

clean:            ## Remove caches locais
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
