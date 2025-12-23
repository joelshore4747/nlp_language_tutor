# api/main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.settings import SETTINGS
from api.schemas import (
    SemanticScoreRequest,
    SemanticScoreResponse,
    SentimentRequest,
    SentimentResponse,
    LanguageDetectRequest,
    LanguageDetectResponse, NerResponse, NerRequest
)
from api.services.analyser_service import semantic_score, sentiment_classify, language_detect, ner_extract

app = FastAPI(title="NLP Adaptive Tutor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/semantic/score", response_model=SemanticScoreResponse)
def post_semantic_score(req: SemanticScoreRequest):
    try:
        return semantic_score(req)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/sentiment", response_model=SentimentResponse)
def post_sentiment(req: SentimentRequest):
    try:
        return sentiment_classify(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/language", response_model=LanguageDetectResponse)
def post_language(req: LanguageDetectRequest):
    try:
        return language_detect(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/ner", response_model=NerResponse)
def post_ner(req: NerRequest):
    try:
        return ner_extract(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

