from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import spacy
from spacy.language import Language

from nlp_tutor.languages import Lang
from nlp_tutor.spacy_loader import load_nlp as get_nlp


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


def extract_ner(lang: Lang, text: str) -> NerResult:
    if not isinstance(lang, Lang):
        raise TypeError(
            f"extract_ner(lang, text) expected lang=Lang, got {type(lang)} value={lang!r}"
        )


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

    noun_phrases: List[str] = []
    try:
        noun_phrases = list(dict.fromkeys([nc.text for nc in doc.noun_chunks]))
    except Exception:
        noun_phrases = []

    return NerResult(entities=ents, noun_phrases=noun_phrases)
