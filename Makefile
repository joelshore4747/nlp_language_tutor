PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest
MAKEFLAGS += --no-print-directory

.DEFAULT_GOAL := help

.PHONY: help all setup api web web-install dev test train-lang-baseline train-lang-bilstm train-sentiment-baseline train-sentiment-bilstm

help:
	@printf "Available targets:\n"
	@printf "  make all                      Run the API and web app together\n"
	@printf "  make setup                    Create .venv and install Python dependencies\n"
	@printf "  make dev                      Run the API and web app together\n"
	@printf "  make api                      Run the FastAPI backend\n"
	@printf "  make web-install              Install web dependencies with npm\n"
	@printf "  make web                      Run the web app in dev mode\n"
	@printf "  make test                     Run the test suite\n"
	@printf "  make train-lang-baseline      Train the baseline language detection model\n"
	@printf "  make train-lang-bilstm        Train the BiLSTM language detection model\n"
	@printf "  make train-sentiment-baseline Train the baseline sentiment model\n"
	@printf "  make train-sentiment-bilstm   Train the BiLSTM sentiment model\n"

all: dev

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

dev:
	$(MAKE) -j2 api web

api:
	$(UVICORN) api.main:app --reload

web-install:
	cd web && npm install

web:
	cd web && npm run dev

test:
	$(PYTEST) -q

train-lang-baseline:
	$(PYTHON) scripts/train_language_detection_baseline_wili.py

train-lang-bilstm:
	$(PYTHON) scripts/train_language_detection_bilstm_wili.py

train-sentiment-baseline:
	$(PYTHON) scripts/train_sentiment_baseline.py

train-sentiment-bilstm:
	$(PYTHON) scripts/train_sentiment_bilstm.py
