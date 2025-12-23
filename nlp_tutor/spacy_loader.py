# nlp_tutor/spacy_loader.py
from __future__ import annotations

from functools import lru_cache
import spacy

from .languages import Lang


SPACY_MODEL_NAME = {
    Lang.EN: "en_core_web_sm",
    Lang.ES: "es_core_news_sm",
    Lang.PL: "pl_core_news_sm",
}


@lru_cache(maxsize=8)
def load_nlp(lang: Lang):
    if lang not in SPACY_MODEL_NAME:
        raise ValueError(f"No spaCy model configured for {lang}")

    model_name = SPACY_MODEL_NAME[lang]
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise OSError(
            f"spaCy model '{model_name}' not installed. Run:\n"
            f"  python -m spacy download {model_name}\n"
        ) from e
