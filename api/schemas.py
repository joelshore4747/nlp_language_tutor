# api/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple


# -------------------------
# Lesson 5: Semantics
# -------------------------

class SemanticScoreRequest(BaseModel):
    lesson_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    learner_es: str = Field(..., min_length=1)


class SimilarityResultOut(BaseModel):
    backend: str
    score: float
    interpretation: str


class SemanticScoreResponse(BaseModel):
    prompt_en: str
    target_es: str
    gloss_en: str
    scores: Dict[str, SimilarityResultOut]
    nearest: Dict[str, List[Tuple[str, float]]]


# -------------------------
# Lesson 6: Sentiment
# -------------------------

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SentimentResponse(BaseModel):
    label: str
    # Keep as string for now because your predict_sentiment() may be returning str(probs).
    # Later improvement: change to Optional[Dict[str, float]] and return a dict.
    probs: Optional[str] = None



class SentimentLSTMResponse(BaseModel):
    label: str
    prob_positive: float


class LabelScore(BaseModel):
    label: str
    score: float

class LanguageDetectRequest(BaseModel):
    text: str = Field(..., min_length=1)

class LanguageDetectResponse(BaseModel):
    label: str
    top_k: List[LabelScore] = []


class NerRequest(BaseModel):
    lang: str = Field(..., min_length=2)  # "EN" or "ES"
    text: str = Field(..., min_length=1)

class EntityOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    explanation: Optional[str] = None

class NerResponse(BaseModel):
    entities: List[EntityOut]
    noun_phrases: List[str]
