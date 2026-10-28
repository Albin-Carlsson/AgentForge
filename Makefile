# AgentForge
#
# `docker compose` is spelled `docker-compose` here because the Homebrew docker
# CLI does not register the compose plugin unless cliPluginsExtraDirs is set in
# ~/.docker/config.json.
PY := .venv/bin/python
COMPOSE := docker-compose

.PHONY: help install db db-down test lint init \
        ingest-scifact ingest-arxiv load-questions index eval-retrieval report \
        measure-parser load-judge-reference validate-judge review-labels eval-e2e report-e2e eval-fast serve

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

# ---------------------------------------------------------------- no API key
ingest-scifact: ## load the BEIR SciFact tier (~5k abstracts)
	$(PY) -m agentforge.cli ingest-scifact

ingest-arxiv: ## fetch, parse and chunk 300 arXiv papers
	$(PY) -m agentforge.cli ingest-arxiv -n 300

load-questions: ## load the frozen 40-question arXiv set (data/arxiv40.jsonl)
	$(PY) -m agentforge.cli load-questions

index: ## build BM25 and dense indexes (CORPUS=scifact|arxiv-rag)
	$(PY) -m agentforge.cli index --corpus $(or $(CORPUS),scifact)

eval-retrieval: ## run the retrieval ladder (TIER=scifact|arxiv40)
	$(PY) -m agentforge.cli eval-retrieval --corpus $(or $(CORPUS),scifact) --tier $(or $(TIER),scifact)

report: ## regenerate the ablation table from stored runs
	$(PY) -m agentforge.cli report --tier $(or $(TIER),scifact)

measure-parser: ## score the PDF parser against arXiv's LaTeXML rendering
	$(PY) -m agentforge.cli measure-parser

eval-fast: ## CI gate: the ladder on a 300-doc slice, no API key, no cost
	$(PY) -m agentforge.cli ingest-scifact --limit 300
	$(PY) -m agentforge.cli index
	$(PY) -m agentforge.cli eval-retrieval

# ------------------------------------------------------- requires an API key
eval-e2e: ## one end-to-end arm over the arXiv questions (ARM=control|single|graph); resumable
	$(PY) -m agentforge.cli eval-e2e --arm $(ARM) $(if $(GENERATOR),--generator $(GENERATOR))

report-e2e: ## the end-to-end table from the latest finished run of each arm
	$(PY) -m agentforge.cli report-e2e $(if $(GENERATOR),--generator $(GENERATOR))

load-judge-reference: ## load the 100 reference pairs (data/judge_reference.jsonl), quotes verified
	$(PY) -m agentforge.cli load-judge-reference

validate-judge: ## score candidate judges against the reference labels
	$(PY) -m agentforge.cli validate-judge

review-labels: ## review reference labels by hand (the part that cannot be automated)
	$(PY) -m agentforge.cli review-labels

serve: ## run the API locally
	$(PY) -m uvicorn agentforge.api.main:app --reload --port 8000
