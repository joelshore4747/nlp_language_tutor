# nlp_tutor/corpora.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import brown, cess_esp, movie_reviews
from sklearn.model_selection import train_test_split

from .config import PATHS
from .languages import Lang
from . import preprocessing as prep



@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: List[str]


# ---------------------------------------------------------------------------
# NLTK corpora: for language models & a small sentiment baseline
# ---------------------------------------------------------------------------

def ensure_nltk_corpora() -> None:
    nltk.download("brown", quiet=True)
    nltk.download("cess_esp", quiet=True)
    nltk.download("movie_reviews", quiet=True)


def load_lm_corpus(lang: Lang) -> List[List[str]]:
    ensure_nltk_corpora()
    prep.ensure_nltk_resources()

    if lang == Lang.EN:
        sents = brown.sents()
    elif lang == Lang.ES:
        sents = cess_esp.sents()
    else:
        raise ValueError(f"Unsupported language for LM: {lang}")

    tokenised: List[List[str]] = []
    for sent in sents:
        raw = " ".join(sent)
        norm = prep.normalise(raw)
        toks = prep.tokenize(norm, lang=lang, lemmatise=True)
        if toks:
            tokenised.append(toks)

    return tokenised


def load_movie_reviews_nltk() -> TextLabelDataset:
    ensure_nltk_corpora()
    fileids = movie_reviews.fileids()

    texts: List[str] = []
    labels: List[str] = []

    for fid in fileids:
        label = movie_reviews.categories(fid)[0]  # 'pos' or 'neg'
        words = movie_reviews.words(fid)
        raw = " ".join(words)
        texts.append(raw)
        labels.append(label)

    return TextLabelDataset(texts=texts, labels=labels)


# ---------------------------------------------------------------------------
# Kaggle: IMDB sentiment dataset
# ---------------------------------------------------------------------------

def load_imdb_kaggle() -> TextLabelDataset:
    path = PATHS.raw_dir / "imdb" / "IMDB Dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"IMDB dataset not found at {path}")

    df = pd.read_csv(path)
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()
    return TextLabelDataset(texts=texts, labels=labels)


# ---------------------------------------------------------------------------
# Kaggle: language detection dataset (for later lessons)
# ---------------------------------------------------------------------------

def load_language_detection_kaggle() -> TextLabelDataset:
    path = PATHS.raw_dir / "lang_detect" / "Language Detection.csv"
    if not path.exists():
        raise FileNotFoundError(f"Language detection dataset not found at {path}")

    df = pd.read_csv(path)
    texts = df["Text"].astype(str).tolist()
    labels = df["Language"].astype(str).tolist()
    return TextLabelDataset(texts=texts, labels=labels)



