# api/state.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from nlp_tutor.lessons import load_lesson_bank, index_lesson_bank, LessonSemanticEngine
from nlp_tutor.classification.imdb_baseline import load_model
from nlp_tutor.classification.lang_detect_baseline import load_model as load_lang_model



try:
    lang_model = load_lang_model()
except Exception:
    lang_model = None

@dataclass
class AppState:
    items: list
    index: dict
    semantic_engine: LessonSemanticEngine
    sentiment_model: Optional[object]
    lang_model: Optional[object]

@lru_cache(maxsize=1)
def get_state() -> AppState:
    """
    Load once per process. Works well with uvicorn --reload because each reload
    starts a new process anyway.
    """
    items = load_lesson_bank()
    idx = index_lesson_bank(items)
    engine = LessonSemanticEngine(items)

    try:
        sentiment_model = load_model()
    except Exception:
        sentiment_model = None

    return AppState(
        items=items,
        index=idx,
        semantic_engine=engine,
        sentiment_model=sentiment_model,
        lang_model=lang_model,
    )


