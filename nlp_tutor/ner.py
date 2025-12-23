from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import spacy
from spacy.language import Language

from nlp_tutor.languages import Lang


@dataclass(frozen=True)
class EntityOut:
    text: str
    label: str
    start_char: int
    end_char: int
    explanation: Optional[str] = None


@dataclass(frozen=True)
class NerResult:
    entities: List[EntityOut]
    noun_phrases: List[str]


_SPACY_CACHE: dict[Lang, Language] = {}


def _model_name(lang: Lang) -> str:
    if lang == Lang.EN:
        return "en_core_web_sm"
    if lang == Lang.ES:
        return "es_core_news_sm"
    raise ValueError(f"No spaCy model configured for {lang}")


def get_nlp(lang: Lang) -> Language:
    if lang not in _SPACY_CACHE:
        _SPACY_CACHE[lang] = spacy.load(_model_name(lang))
    return _SPACY_CACHE[lang]


def extract_ner(lang: Lang, text: str) -> NerResult:
    nlp = get_nlp(lang)
    doc = nlp(text)

    ents: List[EntityOut] = []
    for e in doc.ents:
        ents.append(EntityOut(
            text=e.text,
            label=e.label_,
            start_char=e.start_char,
            end_char=e.end_char,
            explanation=spacy.explain(e.label_),
        ))

    # noun_chunks is available for English; Spanish support depends on pipeline.
    noun_phrases: List[str] = []
    try:
        noun_phrases = list(dict.fromkeys([nc.text for nc in doc.noun_chunks]))
    except Exception:
        noun_phrases = []

    return NerResult(entities=ents, noun_phrases=noun_phrases)
