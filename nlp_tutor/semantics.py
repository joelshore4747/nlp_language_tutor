from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .languages import Lang
from . import preprocessing as prep

Backend = Literal["tfidf", "sbert"]


@dataclass(frozen=True)
class SimilarityResult:
    backend: Backend
    score: float
    interpretation: str


class SemanticScorer:
    """
    Semantic similarity scorer supporting:
      - TF-IDF (classic IR baseline)
      - SBERT multilingual sentence embeddings (modern neural semantics)

    Design:
      - fit(reference_texts) once, then:
          * score_pair(learner, target)
          * nearest(learner, k)
    """

    def __init__(self, backend: Backend = "tfidf") -> None:
        self.backend: Backend = backend

        # TF-IDF
        self._tfidf: Optional[TfidfVectorizer] = None
        self._ref_tfidf = None

        # SBERT
        self._sbert_model = None
        self._ref_emb = None

        # shared
        self._ref_texts: Optional[List[str]] = None

    def fit(self, reference_texts: List[str], lang: Lang) -> None:
        if not reference_texts:
            raise ValueError("reference_texts is empty")

        # light normalisation only; do not over-normalise semantics
        refs = [prep.normalise(t) for t in reference_texts]
        self._ref_texts = refs

        if self.backend == "tfidf":
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )
            self._ref_tfidf = self._tfidf.fit_transform(refs)

        elif self.backend == "sbert":
            self._ensure_sbert()
            self._ref_emb = self._encode_sbert(refs)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def score_pair(self, learner_text: str, target_text: str, lang: Lang) -> SimilarityResult:
        a = prep.normalise(learner_text)
        b = prep.normalise(target_text)

        if self.backend == "tfidf":
            vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            m = vec.fit_transform([a, b])
            score = float(cosine_similarity(m[0], m[1])[0, 0])
            return SimilarityResult("tfidf", score, self._interpret(score))

        self._ensure_sbert()
        emb = self._encode_sbert([a, b])
        score = float(cosine_similarity(emb[0:1], emb[1:2])[0, 0])
        return SimilarityResult("sbert", score, self._interpret(score))

    def nearest(self, learner_text: str, k: int = 5) -> List[Tuple[str, float]]:
        if self._ref_texts is None:
            raise RuntimeError("Call fit() before nearest().")

        q = prep.normalise(learner_text)

        if self.backend == "tfidf":
            if self._tfidf is None or self._ref_tfidf is None:
                raise RuntimeError("TF-IDF not fitted. Call fit() first.")
            q_vec = self._tfidf.transform([q])
            sims = cosine_similarity(q_vec, self._ref_tfidf).ravel()

        else:
            self._ensure_sbert()
            if self._ref_emb is None:
                raise RuntimeError("SBERT not fitted. Call fit() first.")
            q_emb = self._encode_sbert([q])
            sims = cosine_similarity(q_emb, self._ref_emb).ravel()

        idx = np.argsort(-sims)[:k]
        return [(self._ref_texts[i], float(sims[i])) for i in idx]

    def _ensure_sbert(self) -> None:
        if self._sbert_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Install with: python -m pip install sentence-transformers"
            ) from e

        # multilingual, good for EN/ES (and many more)
        self._sbert_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def _encode_sbert(self, texts: List[str]) -> np.ndarray:
        assert self._sbert_model is not None
        emb = self._sbert_model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb)

    def _interpret(self, score: float) -> str:
        if score >= 0.85:
            return "Meaning matches very well."
        if score >= 0.70:
            return "Meaning is mostly correct; minor differences."
        if score >= 0.55:
            return "Meaning is partially correct; key information may be missing."
        return "Meaning does not match the target well."
