# nlp_tutor/corpora.py
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import brown, cess_esp, movie_reviews
from sklearn.model_selection import train_test_split

from .config import PATHS, RESOURCE_LM_DIR
from .languages import Lang
from . import preprocessing as prep



@dataclass
class TextLabelDataset:
    texts: List[str]
    labels: List[str]


def ensure_nltk_corpora() -> None:
    nltk.download("brown", quiet=True)
    nltk.download("cess_esp", quiet=True)
    nltk.download("movie_reviews", quiet=True)

def _lm_env_var(lang: Lang) -> str:
    return f"NLP_TUTOR_LM_{lang.name}"


def _load_local_lm_corpus(lang: Lang) -> List[List[str]]:
    env_path = os.environ.get(_lm_env_var(lang))
    if env_path:
        path = Path(env_path)
    else:
        path = RESOURCE_LM_DIR / f"{lang.value}.txt"

    if not path.exists():
        return []

    prep.ensure_nltk_resources()

    tokenised: List[List[str]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        norm = prep.normalise(text)
        toks = prep.tokenize(norm, lang=lang, lemmatize=True)
        if toks:
            tokenised.append(toks)
    return tokenised


def load_lm_corpus(lang: Lang) -> List[List[str]]:
    local = _load_local_lm_corpus(lang)
    if local:
        return local

    if lang == Lang.EN:
        ensure_nltk_corpora()
        prep.ensure_nltk_resources()
        sents = brown.sents()
    elif lang == Lang.ES:
        ensure_nltk_corpora()
        prep.ensure_nltk_resources()
        sents = cess_esp.sents()
    elif lang == Lang.PL:
        return []
    else:
        return []

    tokenised: List[List[str]] = []
    for sent in sents:
        raw = " ".join(sent)
        norm = prep.normalise(raw)
        toks = prep.tokenize(norm, lang=lang, lemmatize=True)
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


def load_imdb_kaggle() -> TextLabelDataset:
    path = PATHS.raw_dir / "imdb" / "IMDB Dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"IMDB dataset not found at {path}")

    df = pd.read_csv(path)
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(str).tolist()
    return TextLabelDataset(texts=texts, labels=labels)



def load_language_detection_kaggle() -> TextLabelDataset:
    import glob


    folder = (PATHS.raw_dir / "lang_detect")
    candidates = sorted(glob.glob(str(folder / "*.csv")))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in {folder}. Put the dataset there.")

    path = Path(candidates[0])
    df = pd.read_csv(path)


    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("sentence") or cols.get("content")
    lang_col = cols.get("language") or cols.get("label") or cols.get("lang")

    if text_col is None or lang_col is None:
        raise ValueError(
            f"Language detection CSV must contain Text + Language columns. "
            f"Found columns: {list(df.columns)}"
        )

    texts = df[text_col].astype(str).tolist()
    labels = df[lang_col].astype(str).tolist()
    return TextLabelDataset(texts=texts, labels=labels)

