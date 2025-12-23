from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd

from .config import PATHS
from .languages import Lang
from .semantics import SemanticScorer, SimilarityResult
from semantics_transformer import TransformerEmbedder, EmbedderConfig


@dataclass(frozen=True)
class LessonItem:
    lesson_id: int
    item_id: int
    topic: str
    prompt_en: str
    target_es: str
    gloss_en: str


def load_lesson_bank() -> List[LessonItem]:
    path = PATHS.processed_dir / "lesson_bank.csv"
    if not path.exists():
        raise FileNotFoundError(f"lesson_bank.csv not found at {path}")

    df = pd.read_csv(path)
    items: List[LessonItem] = []
    for _, r in df.iterrows():
        items.append(LessonItem(
            lesson_id=int(r["lesson_id"]),
            item_id=int(r["item_id"]),
            topic=str(r.get("topic", "")),
            prompt_en=str(r["prompt_en"]),
            target_es=str(r["target_es"]),
            gloss_en=str(r["gloss_en"]),
        ))
    return items


def index_lesson_bank(items: List[LessonItem]) -> Dict[Tuple[int, int], LessonItem]:
    return {(i.lesson_id, i.item_id): i for i in items}


class LessonSemanticEngine:
    """
    Holds fitted semantic scorers over *all targets* so we can:
      - score learner answer vs target
      - retrieve nearest targets (helpful for feedback / grading)
    """

    def __init__(self, items: List[LessonItem]) -> None:
        self.items = items
        self.targets_es = [i.target_es for i in items]

        self.tfidf = SemanticScorer("tfidf")
        self.tfidf.fit(self.targets_es, lang=Lang.ES)

        self.sbert: Optional[SemanticScorer] = None
        try:
            s = SemanticScorer("sbert")
            s.fit(self.targets_es, lang=Lang.ES)
            self.sbert = s
        except Exception:
            self.sbert = None

        cfg = EmbedderConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_length=128,
            batch_size=16,
        )
        self.embedder = TransformerEmbedder(cfg)

        self._targets_es = [it.target_es for it in items]
        self._target_embs = self.embedder.encode(self._targets_es)  # shape (N, H)

    def score_answer(self, learner_es: str, target_es: str) -> Dict[str, SimilarityResult]:
        out: Dict[str, SimilarityResult] = {}
        out["tfidf"] = self.tfidf.score_pair(learner_es, target_es, lang=Lang.ES)
        if self.sbert is not None:
            out["sbert"] = self.sbert.score_pair(learner_es, target_es, lang=Lang.ES)
        return out

    def nearest_targets(self, learner_es: str, k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        out["tfidf"] = self.tfidf.nearest(learner_es, k=k)
        if self.sbert is not None:
            out["sbert"] = self.sbert.nearest(learner_es, k=k)
        return out

    def interpret_similarity(score: float) -> str:
        if score >= 0.80:
            return "very close meaning"
        if score >= 0.65:
            return "similar meaning"
        if score >= 0.50:
            return "partially related"
        return "likely different meaning"

