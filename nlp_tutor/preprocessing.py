# nlp_tutor/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import re

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from .languages import Lang


def ensure_nltk_resources() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)  # required by some NLTK versions
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


_lemmatizer = WordNetLemmatizer()
_word_pattern = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def sentence_segment(text: str, lang: Lang = Lang.EN) -> List[str]:
    """
    Sentence segmentation.

    NLTK's Punkt models are strongest in English, but are usable for Spanish.
    If you later want better Spanish segmentation, we can swap to spaCy for ES.
    """
    return sent_tokenize(text)


def tokenize(text: str, lang: Lang = Lang.EN, lemmatise: bool = True) -> List[str]:
    """
    Tokenise text; lemmatise for English (WordNet), skip for Spanish for now.

    Why skip ES lemmatisation?
    - NLTK WordNet lemmatiser is English-focused.
    - We keep tokenisation consistent and document this limitation.
    """
    tokens = word_tokenize(text)

    if not lemmatise:
        return tokens

    lemmas: List[str] = []
    for tok in tokens:
        if not _word_pattern.fullmatch(tok):
            # keep punctuation etc. as-is
            lemmas.append(tok)
            continue

        if lang == Lang.EN:
            lemmas.append(_lemmatizer.lemmatize(tok))
        else:
            # For ES (and other languages), keep token unchanged for now
            lemmas.append(tok)

    return lemmas


def tokenize_sentences(text: str, lang: Lang = Lang.EN, lemmatise: bool = True) -> List[List[str]]:
    sents = sentence_segment(text, lang=lang)
    return [tokenize(s, lang=lang, lemmatise=lemmatise) for s in sents]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class TextStats:
    n_tokens: int
    n_types: int
    type_token_ratio: float
    avg_sentence_length: float


def type_token_ratio(tokens: Iterable[str]) -> float:
    tokens = list(tokens)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_stats(text: str, lang: Lang = Lang.EN) -> TextStats:
    sents = sentence_segment(text, lang=lang)
    tokenised = [tokenize(s, lang=lang, lemmatise=True) for s in sents]
    all_tokens = [t for sent in tokenised for t in sent if _word_pattern.fullmatch(t)]

    if not tokenised:
        return TextStats(n_tokens=0, n_types=0, type_token_ratio=0.0, avg_sentence_length=0.0)

    n_tokens = len(all_tokens)
    n_types = len(set(all_tokens))
    ttr = type_token_ratio(all_tokens)
    avg_len = n_tokens / len(tokenised)

    return TextStats(n_tokens=n_tokens, n_types=n_types, type_token_ratio=ttr, avg_sentence_length=avg_len)
