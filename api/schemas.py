# api/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple


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


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SentimentResponse(BaseModel):
    label: str
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


class TutorEvaluateRequest(BaseModel):
    lesson_id: int = Field(..., ge=1)
    item_id: int = Field(..., ge=1)
    learner_text: str = Field(..., min_length=1)
    expected_lang: str = Field("ES", min_length=2)
    allow_mixed: bool = False
    prompt_en: Optional[str] = None
    target_text: Optional[str] = None
    gloss_en: Optional[str] = None


class TutorAction(BaseModel):
    code: str
    message: str
    severity: str

class FluencyOut(BaseModel):
    perplexity: float
    band: str


class TutorEvaluateResponse(BaseModel):
    # lesson context
    prompt_en: str
    target_es: str
    gloss_en: str
    expected_lang: str

    # NLP signals
    detected_lang: str
    detected_top_k: List["LabelScore"]

    syntax_issues: List[Dict]
    fluency: Optional["FluencyOut"] = None
    ner: Dict
    semantics: Dict[str, "SimilarityResultOut"]
    nearest: Dict[str, List[Tuple[str, float]]]

    # dialogue policy output
    action: TutorAction


