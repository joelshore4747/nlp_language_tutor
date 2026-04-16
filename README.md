# NLP Adaptive Tutor

A compact NLP tutoring system that evaluates short learner responses with language ID, syntax checks, semantic similarity, NER, and fluency scoring. It exposes a FastAPI backend and a small Solid.js web UI.

## Features
- Unified pipeline: language detection, syntax checks, semantics, NER, fluency, and policy gating.
- Two language ID models: char TF-IDF baseline and char BiLSTM.
- Semantic scoring: TF-IDF cosine plus transformer embeddings.
- API endpoints for tutoring and semantic scoring.
- Web UI for interactive feedback panels.

## Repo Layout
- `api/`: FastAPI app and routes.
- `nlp_tutor/`: core NLP pipeline and models.
- `nlp_tutor/resources/`: packaged lesson bank and baseline models.
- `data/`: datasets, models, and lesson bank.
- `scripts/`: training and helper scripts.
- `tests/`: unit tests.
- `web/`: Solid.js frontend.
- `reports/`: project report plus a single analysis notebook at `reports/notebooks/01_nlp_models_analysis.ipynb`.

## Setup

### Python
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### spaCy Models
The required spaCy models are pinned in `requirements.txt`. If they do not install, run:
```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download pl_core_news_sm
```

### Web
```bash
cd web
bun install
or 
npm install
```

## Run

### API
```bash
. .venv/bin/activate
uvicorn api.main:app --reload
```

### Web UI
```bash
cd web
bun dev
or
npm run dev
```

The web app reads `VITE_API_BASE` from `web/.env`.

## API Endpoints

### Health
`GET /health`

### Tutor Evaluate
`POST /tutor/evaluate`
```json
{
  "lesson_id": 1,
  "item_id": 1,
  "learner_text": "Me llamo Joel y estudio informatica.",
  "expected_lang": "ES",
  "allow_mixed": false
}
```

### Semantic Score
`POST /semantic/score`
```json
{
  "lesson_id": 1,
  "item_id": 1,
  "learner_es": "Me llamo Joel y estudio informatica."
}
```

## Results Summary (Sample Evaluations)

### Language Detection (WiLI 2018 test sample)
- Setup: 500 samples per class (EN, ES, PL), total N=1500.
- Baseline (char TF-IDF + logreg): Accuracy 0.9927, Macro F1 0.9927.
- Char BiLSTM: Accuracy 0.9440, Macro F1 0.9437.

### Semantic Similarity (lesson bank sanity check)
- Setup: exact target vs random negative target (12 items).
- TF-IDF: mean positive 1.0000, mean negative 0.0133, pairwise accuracy 1.00.
- Transformer: mean positive 1.0000, mean negative 0.2634, pairwise accuracy 1.00.

Full write-up is in `reports/Project_Report.pdf`, with supporting analysis in `reports/notebooks/01_nlp_models_analysis.ipynb`.

## Training

### Language Detection
```bash
python scripts/train_language_detection_baseline_wili.py
python scripts/train_language_detection_bilstm_wili.py
```

### Sentiment (optional)
```bash
python scripts/train_sentiment_baseline.py
python scripts/train_sentiment_bilstm.py
```

## Tests
```bash
. .venv/bin/activate
pytest -q
```

## Submission Bundle (No Raw Datasets)
This project runs without the raw training datasets. Only the lesson bank and pretrained models are required.
To build a submission zip that excludes `data/raw`:
```bash
python scripts/build_submission_bundle.py
```
The bundle is written to `dist/nlp-adaptive-tutor-submission.zip`.

## Docs and Reports
- Project report: `reports/Project_Report.md` and `reports/Project_Report.pdf`
- Analysis notebook (single): `reports/notebooks/01_nlp_models_analysis.ipynb`
- Data guide: `data/README.md`

## Makefile Shortcuts
```bash
make setup
make all
make dev
make api
make web
make test
```

## Notes
- Transformer models download on first use.
- The lesson bank is small and intended for demonstration and class exercises.
- Raw datasets under `data/raw` are only used for training scripts and are not required to run the API or UI.
- Override paths if needed:
  - `NLP_TUTOR_LESSON_BANK=/path/to/lesson_bank.csv`
  - `NLP_TUTOR_MODELS_DIR=/path/to/models`
  - `NLP_TUTOR_LM_PL=/path/to/polish_corpus.txt`
- For Polish fluency, add `nlp_tutor/resources/lm/pl.txt` (one sentence per line) or set `NLP_TUTOR_LM_PL`.
