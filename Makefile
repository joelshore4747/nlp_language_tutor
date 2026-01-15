PYTHON := .venv/bin/python
PIP := .venv/bin/pip
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest

.PHONY: setup api web web-install test docs submission train-lang-baseline train-lang-bilstm train-sentiment-baseline train-sentiment-bilstm

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

api:
	$(UVICORN) api.main:app --reload

web-install:
	cd web && npm install

web:
	cd web && npm run dev

test:
	$(PYTEST) -q

docs:
	$(PYTHON) scripts/build_docs.py

submission:
	$(PYTHON) scripts/build_submission_bundle.py

train-lang-baseline:
	$(PYTHON) scripts/train_language_detection_baseline_wili.py

train-lang-bilstm:
	$(PYTHON) scripts/train_language_detection_bilstm_wili.py

train-sentiment-baseline:
	$(PYTHON) scripts/train_sentiment_baseline.py

train-sentiment-bilstm:
	$(PYTHON) scripts/train_sentiment_bilstm.py
