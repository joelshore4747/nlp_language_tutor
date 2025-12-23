# api/services/analyzer_service.py
from __future__ import annotations

from typing import Dict

from api.schemas import (
    SemanticScoreRequest,
    SemanticScoreResponse,
    SimilarityResultOut,
    SentimentRequest,
    SentimentResponse,
    LanguageDetectRequest,
    LanguageDetectResponse,
    LabelScore, NerResponse,
    EntityOut, NerRequest
)
from api.state import get_state

from nlp_tutor.classification.imdb_baseline import predict_sentiment
from nlp_tutor.classification.imdb_bilstm_infer import load_bilstm, predict_bilstm
from nlp_tutor.classification.lang_detect_baseline import predict_language
from nlp_tutor.languages import Lang
from nlp_tutor.ner import extract_ner


def interpret_similarity(score: float) -> str:
    if score >= 0.80:
        return "very close meaning"
    if score >= 0.65:
        return "similar meaning"
    if score >= 0.50:
        return "partially related"
    return "likely different meaning"


def semantic_score(req: SemanticScoreRequest) -> SemanticScoreResponse:
    state = get_state()

    key = (req.lesson_id, req.item_id)
    if key not in state.index:
        raise KeyError(f"Lesson item not found: {key}")

    item = state.index[key]

    scores = state.semantic_engine.score_answer(req.learner_es, item.target_es)
    nearest = state.semantic_engine.nearest_targets(req.learner_es, k=5)

    scores_out: Dict[str, SimilarityResultOut] = {}
    for name, r in scores.items():
        scores_out[name] = SimilarityResultOut(
            backend=r.backend,
            score=r.score,
            interpretation=r.interpretation,
        )

    if hasattr(state.semantic_engine, "score_transformer"):
        t_score = state.semantic_engine.score_transformer(req.learner_es, item.target_es)
        scores_out["transformer_cosine"] = SimilarityResultOut(
            backend="transformer_cosine",
            score=float(t_score),
            interpretation=interpret_similarity(float(t_score)),
        )

    if hasattr(state.semantic_engine, "nearest_targets_transformer"):
        t_nearest = state.semantic_engine.nearest_targets_transformer(req.learner_es, k=5)
        # Add a new key to the nearest dict (keep existing nearest results intact)
        nearest["transformer_cosine"] = t_nearest

    return SemanticScoreResponse(
        prompt_en=item.prompt_en,
        target_es=item.target_es,
        gloss_en=item.gloss_en,
        scores=scores_out,
        nearest=nearest,
    )


def sentiment_classify(req: SentimentRequest) -> SentimentResponse:
    state = get_state()

    if state.sentiment_model is None:
        raise RuntimeError(
            "Sentiment model not loaded. Train it first:\n"
            "  python scripts/train_sentiment_baseline.py"
        )

    out = predict_sentiment(req.text, model=state.sentiment_model)
    return SentimentResponse(label=out["label"], probs=out.get("probs"))




_LSTM = None
_VOCAB = None
try:
    _LSTM, _VOCAB = load_bilstm()
except Exception:
    _LSTM, _VOCAB = None, None

def sentiment_classify_lstm(req: SentimentRequest):
    if _LSTM is None or _VOCAB is None:
        raise RuntimeError("BiLSTM not available. Train: python scripts/train_sentiment_bilstm.py")
    out = predict_bilstm(req.text, _LSTM, _VOCAB)
    return out





def language_detect(req: LanguageDetectRequest) -> LanguageDetectResponse:
    state = get_state()
    if state.lang_model is None:
        raise RuntimeError(
            "Language model not loaded. Train it first:\n"
            "  python scripts/train_language_detection_baseline.py"
        )

    out = predict_language(req.text, model=state.lang_model, top_k=5)
    top = [LabelScore(label=l, score=s) for (l, s) in out.get("top_k", [])]
    return LanguageDetectResponse(label=out["label"], top_k=top)


def ner_extract(req: NerRequest) -> NerResponse:
    lang = Lang[req.lang.upper()]  # expects "EN"/"ES"
    r = extract_ner(lang, req.text)
    return NerResponse(
        entities=[EntityOut(**e.__dict__) for e in r.entities],
        noun_phrases=r.noun_phrases
    )