from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .languages import Lang
from . import preprocessing as prep
from .corpora import load_lm_corpus
from .ngram_model import NgramLanguageModel, NgramLMConfig

@dataclass
class FluencyScore:
    perplexity: float
    band: str

class FluencyScorer:
    def __init__(self, order: int = 2, add_k: float = 1.0) -> None:
        self.config = NgramLMConfig(order=order, add_k=add_k)
        self.models: dict[Lang, NgramLanguageModel] = {}

    def train_if_needed(self, lang: Lang) -> None:
        if lang in self.models:
            return

        corpus = load_lm_corpus(lang)
        lm = NgramLanguageModel(self.config)
        lm.fit(corpus)
        self.models[lang] = lm

    def score_text(self, text: str, lang: Lang) -> FluencyScore:
        self.train_if_needed(lang)

        norm = prep.normalise(text)
        tokens = prep.tokenize(norm, lang-lang, lemmatize=True)

        ppl = self.models[lang].perplexity(tokens)

        if ppl < 60:
            band = "high"
        elif ppl < 150:
            band = "medium"
        else:
            band = "low"

        return FluencyScore(perplexity=ppl, band=band)

