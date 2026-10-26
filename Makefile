# AgentForge
#
# `docker compose` is spelled `docker-compose` here because the Homebrew docker
# CLI does not register the compose plugin unless cliPluginsExtraDirs is set in
# ~/.docker/config.json.
PY := .venv/bin/python
COMPOSE := docker-compose

.PHONY: help install db db-down test lint init ingest-scifact index eval-retrieval eval-fast

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## install the package and dev dependencies
	$(PY) -m pip install -q -e ".[dev]"

db: ## start ParadeDB (pg_search + pgvector) and wait for health
	$(COMPOSE) up -d db
	@until [ "$$(docker inspect --format '{{.State.Health.Status}}' agentforge-db 2>/dev/null)" = "healthy" ]; do sleep 2; done
	@echo "db healthy"

db-down: ## stop the database (keeps the volume)
	$(COMPOSE) down

test: ## run the test suite (requires a running db; tests fail rather than skip)
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests

init: ## apply schema, print engine versions
	$(PY) -m agentforge.cli init

ingest-scifact: ## load the BEIR SciFact tier (~5k abstracts, no API key)
	$(PY) -m agentforge.cli ingest-scifact

index: ## build BM25 and dense indexes over the scifact corpus
	$(PY) -m agentforge.cli index

eval-retrieval: ## run the 4-arm ladder, write results/retrieval-scifact.md
	$(PY) -m agentforge.cli eval-retrieval

eval-fast: ## CI gate: the ladder on a 300-doc slice, no API key, no cost
	$(PY) -m agentforge.cli ingest-scifact --limit 300
	$(PY) -m agentforge.cli index
	$(PY) -m agentforge.cli eval-retrieval
