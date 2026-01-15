from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import random
import re
import math

import numpy as np

from .languages import Lang
from . import preprocessing as prep
from .corpora import load_lm_corpus
from .ngram_model import NgramLanguageModel, NgramLMConfig


@dataclass
class FluencyScore:
    perplexity: float
    band: str


_ARTEFACT_RE = re.compile(r"^-[a-z]{2,10}-$", re.IGNORECASE)


def _clean_tokens(
    tokens: Iterable[str],
    *,
    drop_artefacts: bool = True,
    drop_underscore_tokens: bool = True,
    drop_numeric_tokens: bool = True,
) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if not t:
            continue
        tt = t.strip()
        if not tt:
            continue

        if drop_artefacts and _ARTEFACT_RE.match(tt):
            continue
        if drop_underscore_tokens and "_" in tt:
            continue
        if drop_numeric_tokens and tt.isdigit():
            continue

        out.append(tt)
    return out


class FluencyScorer:
    def __init__(
        self,
        order: int = 2,
        add_k: float = 2.0,
        calibrate_samples: int = 400,
        q_high: float = 0.50,
        q_medium: float = 0.80,
        lemmatize: bool = False,
        debug: bool = False,
        # token cleanup knobs
        drop_artefacts: bool = True,
        drop_underscore_tokens: bool = True,
        drop_numeric_tokens: bool = True,
        # calibration knobs
        min_lesson_calib_sents: int = 10,
    ) -> None:
        if not (0.0 < q_high < 1.0) or not (0.0 < q_medium < 1.0):
            raise ValueError("q_high and q_medium must be quantiles in (0, 1)")
        if q_medium <= q_high:
            raise ValueError("q_medium must be > q_high")

        self.config = NgramLMConfig(order=order, add_k=add_k)

        self.models: Dict[Lang, NgramLanguageModel] = {}
        self.thresholds: Dict[Lang, Tuple[float, float]] = {}

        self.calibrate_samples = calibrate_samples
        self.q_high = q_high
        self.q_medium = q_medium
        self.lemmatize = lemmatize
        self.debug = debug

        self.drop_artefacts = drop_artefacts
        self.drop_underscore_tokens = drop_underscore_tokens
        self.drop_numeric_tokens = drop_numeric_tokens

        self.min_lesson_calib_sents = min_lesson_calib_sents

    # -------------------------
    # Public helpers (testing)
    # -------------------------

    def train_from_tokenized(
        self,
        lang: Lang,
        tokenized_sents: List[List[str]],
        *,
        calibrate: bool = True,
    ) -> None:

        sents = [
            _clean_tokens(
                s,
                drop_artefacts=self.drop_artefacts,
                drop_underscore_tokens=self.drop_underscore_tokens,
                drop_numeric_tokens=self.drop_numeric_tokens,
            )
            for s in tokenized_sents
        ]
        sents = [s for s in sents if len(s) >= 1]

        lm = NgramLanguageModel(self.config)
        lm.fit(sents)
        self.models[lang] = lm

        if calibrate:
            self.thresholds[lang] = self._calibrate_thresholds(lm, sents)

    def train_if_needed(self, lang: Lang) -> None:
        if lang in self.models:
            return

        raw_corpus = load_lm_corpus(lang)  # List[List[str]] from NLTK sources
        if not raw_corpus:

            self.thresholds[lang] = (150.0, 300.0)
            return

        train_sents: List[List[str]] = []
        for toks in raw_corpus:
            if not toks:
                continue
            text = " ".join(toks)
            norm = prep.normalise(text)
            scored_toks = prep.tokenize(norm, lang, lemmatize=self.lemmatize)
            scored_toks = _clean_tokens(
                scored_toks,
                drop_artefacts=self.drop_artefacts,
                drop_underscore_tokens=self.drop_underscore_tokens,
                drop_numeric_tokens=self.drop_numeric_tokens,
            )
            if len(scored_toks) >= 1:
                train_sents.append(scored_toks)

        if not train_sents:
            self.thresholds[lang] = (150.0, 300.0)
            return

        lm = NgramLanguageModel(self.config)
        lm.fit(train_sents)
        self.models[lang] = lm

        lesson_calib = self._lesson_calibration_sents(lang)
        if len(lesson_calib) >= self.min_lesson_calib_sents:
            calib_sents = lesson_calib
            calib_source = "lesson_bank"
        else:
            calib_sents = train_sents
            calib_source = "corpus"

        self.thresholds[lang] = self._calibrate_thresholds(lm, calib_sents)

        if self.debug:
            hi, med = self.thresholds[lang]
            print(
                f"[fluency] {lang.name} trained on {len(train_sents)} sents; "
                f"calib_source={calib_source} (n={len(calib_sents)}); "
                f"thresholds high={hi:.1f} medium={med:.1f}"
            )
            print("[fluency] sample_train:", train_sents[0][:12])
            if lesson_calib:
                print("[fluency] sample_lesson_calib:", lesson_calib[0][:12])

    def score_text(self, text: str, lang: Lang) -> FluencyScore:
        self.train_if_needed(lang)

        if lang not in self.models:
            return FluencyScore(perplexity=float("nan"), band="unknown")

        norm = prep.normalise(text)
        tokens = prep.tokenize(norm, lang, lemmatize=self.lemmatize)
        tokens = _clean_tokens(
            tokens,
            drop_artefacts=self.drop_artefacts,
            drop_underscore_tokens=self.drop_underscore_tokens,
            drop_numeric_tokens=self.drop_numeric_tokens,
        )

        if not tokens:
            return FluencyScore(perplexity=float("inf"), band="low")

        ppl = float(self.models[lang].perplexity(tokens))
        if math.isnan(ppl) or math.isinf(ppl):
            return FluencyScore(perplexity=ppl, band="low")

        high_thr, med_thr = self.thresholds.get(lang, (150.0, 300.0))

        if self.debug:
            print("[fluency] score_tokens:", tokens[:12], "ppl=", ppl)

        if ppl <= high_thr:
            band = "high"
        elif ppl <= med_thr:
            band = "medium"
        else:
            band = "low"

        return FluencyScore(perplexity=ppl, band=band)

    def _calibrate_thresholds(
        self,
        lm: NgramLanguageModel,
        sents: List[List[str]],
    ) -> Tuple[float, float]:

        if not sents:
            return (150.0, 300.0)

        k = min(self.calibrate_samples, len(sents))
        sample = random.sample(sents, k=k) if len(sents) > k else sents

        ppls: List[float] = []
        for toks in sample:
            if not toks:
                continue

            if len(toks) < self.config.order:
                continue
            try:
                p = float(lm.perplexity(toks))
                if not (math.isnan(p) or math.isinf(p)):
                    ppls.append(p)
            except Exception:
                continue

        if not ppls:
            return (150.0, 300.0)

        high = float(np.quantile(ppls, self.q_high))
        medium = float(np.quantile(ppls, self.q_medium))

        # Safety: ensure ordering + non-trivial separation
        if medium <= high:
            medium = high * 1.25

        high = max(high, 10.0)
        medium = max(medium, high + 1.0)

        return (high, medium)

    def _lesson_calibration_sents(self, lang: Lang) -> List[List[str]]:
        try:
            from .lessons import load_lesson_bank
        except Exception:
            return []

        try:
            bank = load_lesson_bank()
        except Exception:
            return []

        # IMPORTANT: bank is already an iterable (likely a list)
        items = bank

        texts: List[str] = []
        if lang == Lang.ES:
            texts = [it.target_es for it in items if getattr(it, "target_es", None)]
        elif lang == Lang.EN:
            texts = [
                (getattr(it, "target_en", None) or getattr(it, "prompt_en", None))
                for it in items
                if (getattr(it, "target_en", None) or getattr(it, "prompt_en", None))
            ]
        elif lang == Lang.PL:
            texts = [it.target_pl for it in items if getattr(it, "target_pl", None)]
        else:
            return []

        sents: List[List[str]] = []
        for t in texts:
            norm = prep.normalise(t)
            toks = prep.tokenize(norm, lang, lemmatize=self.lemmatize)
            toks = _clean_tokens(
                toks,
                drop_artefacts=self.drop_artefacts,
                drop_underscore_tokens=self.drop_underscore_tokens,
                drop_numeric_tokens=self.drop_numeric_tokens,
            )
            if len(toks) >= self.config.order:
                sents.append(toks)

        return sents
