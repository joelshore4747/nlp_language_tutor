# api/state.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from nlp_tutor.lessons import load_lesson_bank, index_lesson_bank, LessonSemanticEngine
from nlp_tutor.classification.imdb_baseline import load_model
from nlp_tutor.classification.lang_detect_baseline import load_model as load_lang_model
from nlp_tutor.classification.lang_detect_bilstm_chars import load_model as load_bilstm_pack, CharBiLSTM
from nlp_tutor.pipeline import TutorPipeline
from nlp_tutor.fluency import FluencyScorer
import torch


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
    bilstm_model: Optional[CharBiLSTM] = None
    bilstm_pack: Optional[dict] = None
    pipeline: Optional[TutorPipeline] = None

@lru_cache(maxsize=1)
def get_state() -> AppState:
    items = load_lesson_bank()
    idx = index_lesson_bank(items)
    engine = LessonSemanticEngine(items)

    try:
        sentiment_model = load_model()
    except Exception:
        sentiment_model = None
    
    bilstm_model = None
    bilstm_pack = None
    try:
        bilstm_pack = load_bilstm_pack()
        cfg = bilstm_pack["config"]
        char2idx = bilstm_pack["char2idx"]
        idx2label = bilstm_pack["idx2label"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bilstm_model = CharBiLSTM(
            vocab_size=len(char2idx),
            n_classes=len(idx2label),
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        bilstm_model.load_state_dict(bilstm_pack["state_dict"])
        bilstm_model.eval()
    except Exception:
        bilstm_model = None
        bilstm_pack = None

    fluency = FluencyScorer()
    from nlp_tutor.languages import Lang
    for l in [Lang.EN, Lang.ES]:
        try:
            fluency.train_if_needed(l)
        except Exception as e:
            print(f"Warning: could not pre-train fluency for {l}: {e}")

    pipeline = TutorPipeline(
        lang_model=lang_model,
        semantic_engine=engine,
        fluency_scorer=fluency,
        bilstm_model=bilstm_model,
        bilstm_pack=bilstm_pack
    )

    return AppState(
        items=items,
        index=idx,
        semantic_engine=engine,
        sentiment_model=sentiment_model,
        lang_model=lang_model,
        bilstm_model=bilstm_model,
        bilstm_pack=bilstm_pack,
        pipeline=pipeline,
    )


