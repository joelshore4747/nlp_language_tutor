from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd

from .config import resolve_lesson_bank_path
from .languages import Lang
from .semantics import SemanticScorer, SimilarityResult
from .semantics_transformer import EmbedderConfig, TransformerEmbedder


@dataclass(frozen=True)
class LessonItem:
    lesson_id: int
    item_id: int
    topic: str
    prompt_en: str
    target_es: str
    target_en: str
    target_pl: str
    gloss_en: str


def load_lesson_bank() -> List[LessonItem]:
    path = resolve_lesson_bank_path()
    if not path.exists():
        raise FileNotFoundError(f"lesson_bank.csv not found at {path}")

    df = pd.read_csv(path)
    items: List[LessonItem] = []
    def _cell_str(val: object) -> str:
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
        return str(val)
    for _, r in df.iterrows():
        prompt_en = _cell_str(r.get("prompt_en", ""))
        target_es = _cell_str(r.get("target_es", ""))
        gloss_en = _cell_str(r.get("gloss_en", ""))
        target_en = _cell_str(r.get("target_en", "")) or gloss_en or prompt_en
        target_pl = _cell_str(r.get("target_pl", ""))
        items.append(LessonItem(
            lesson_id=int(r["lesson_id"]),
            item_id=int(r["item_id"]),
            topic=str(r.get("topic", "")),
            prompt_en=prompt_en,
            target_es=target_es,
            target_en=target_en,
            target_pl=target_pl,
            gloss_en=gloss_en,
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
        self.targets_by_lang: Dict[Lang, List[str]] = {
            Lang.ES: [i.target_es for i in items if i.target_es],
            Lang.EN: [i.target_en or i.prompt_en for i in items if (i.target_en or i.prompt_en)],
            Lang.PL: [i.target_pl for i in items if i.target_pl],
        }
        self.targets_es = self.targets_by_lang.get(Lang.ES, [])

        self.tfidf_by_lang: Dict[Lang, SemanticScorer] = {}
        self.sbert_by_lang: Dict[Lang, SemanticScorer] = {}
        for lang, targets in self.targets_by_lang.items():
            if not targets:
                continue
            tfidf = SemanticScorer("tfidf")
            tfidf.fit(targets, lang=lang)
            self.tfidf_by_lang[lang] = tfidf

            try:
                s = SemanticScorer("sbert")
                s.fit(targets, lang=lang)
                self.sbert_by_lang[lang] = s
            except Exception:
                pass

        cfg = EmbedderConfig(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            max_length=128,
            batch_size=16,
        )
        self.embedder = TransformerEmbedder(cfg)

        self._targets_by_lang: Dict[Lang, List[str]] = {
            lang: targets for lang, targets in self.targets_by_lang.items() if targets
        }
        self._target_embs_by_lang: Dict[Lang, "np.ndarray"] = {}
        for lang, targets in self._targets_by_lang.items():
            self._target_embs_by_lang[lang] = self.embedder.encode(targets)

    def _pick_lang(self, lang: Lang) -> Lang:
        if self.targets_by_lang.get(lang):
            return lang
        if self.targets_by_lang.get(Lang.ES):
            return Lang.ES
        for cand, targets in self.targets_by_lang.items():
            if targets:
                return cand
        return lang

    def score_answer(self, learner_text: str, target_text: str, lang: Lang = Lang.ES) -> Dict[str, SimilarityResult]:
        out: Dict[str, SimilarityResult] = {}
        chosen = self._pick_lang(lang)
        tfidf = self.tfidf_by_lang.get(chosen)
        if tfidf is not None:
            out["tfidf"] = tfidf.score_pair(learner_text, target_text, lang=chosen)
        sbert = self.sbert_by_lang.get(chosen)
        if sbert is not None:
            out["sbert"] = sbert.score_pair(learner_text, target_text, lang=chosen)
        return out

    def score_transformer(self, learner_text: str, target_text: str, lang: Lang = Lang.ES) -> float:
        return self.embedder.similarity(learner_text, target_text)

    def nearest_targets(self, learner_text: str, lang: Lang = Lang.ES, k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        chosen = self._pick_lang(lang)
        tfidf = self.tfidf_by_lang.get(chosen)
        if tfidf is not None:
            out["tfidf"] = tfidf.nearest(learner_text, k=k)
        sbert = self.sbert_by_lang.get(chosen)
        if sbert is not None:
            out["sbert"] = sbert.nearest(learner_text, k=k)
        return out

    def nearest_targets_transformer(self, learner_text: str, lang: Lang = Lang.ES, k: int = 5) -> List[Tuple[str, float]]:
        from .semantics_transformer import cosine_sim
        chosen = self._pick_lang(lang)
        targets = self._targets_by_lang.get(chosen, [])
        target_embs = self._target_embs_by_lang.get(chosen)
        if target_embs is None or not targets:
            return []
        q_emb = self.embedder.encode([learner_text])[0]
        # target_embs: (N, H), q_emb: (H,)
        sims = []
        for i, t_emb in enumerate(target_embs):
            sims.append(cosine_sim(q_emb, t_emb))

        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [(targets[i], float(sims[i])) for i in idx]

    def nearest_targets_by_lang(self, learner_text: str, k_per_lang: int = 1) -> List[Tuple[str, float]]:
        from .semantics_transformer import cosine_sim
        q_emb = self.embedder.encode([learner_text])[0]
        out: List[Tuple[str, float]] = []
        for lang in (Lang.EN, Lang.PL, Lang.ES):
            targets = self._targets_by_lang.get(lang)
            target_embs = self._target_embs_by_lang.get(lang)
            if not targets or target_embs is None:
                continue
            sims = []
            for i, t_emb in enumerate(target_embs):
                sims.append(cosine_sim(q_emb, t_emb))
            idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k_per_lang]
            for i in idx:
                out.append((f"{lang.name}: {targets[i]}", float(sims[i])))
        return out

    @staticmethod
    def interpret_similarity(score: float) -> str:
        if score >= 0.80:
            return "very close meaning"
        if score >= 0.65:
            return "similar meaning"
        if score >= 0.50:
            return "partially related"
        return "likely different meaning"
