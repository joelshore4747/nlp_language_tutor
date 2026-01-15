from fastapi import APIRouter, HTTPException
from api.schemas import SemanticScoreRequest
from api.state import get_state
from nlp_tutor.languages import Lang

router = APIRouter(prefix="/semantic", tags=["semantic"])

@router.post("/score")
def semantic_score(req: SemanticScoreRequest) -> dict:
    state = get_state()

    eng = state.semantic_engine

    key = (req.lesson_id, req.item_id)
    item = state.index.get(key)
    if item is None:
        raise HTTPException(status_code=404, detail="Lesson item not found")

    scores = eng.score_answer(req.learner_es, item.target_es, lang=Lang.ES)
    nearest = eng.nearest_targets(req.learner_es, lang=Lang.ES, k=5)
    if hasattr(eng, "nearest_targets_by_lang"):
        try:
            nearest["by_lang"] = eng.nearest_targets_by_lang(req.learner_es, k_per_lang=1)
        except Exception:
            pass

    return {
        "prompt_en": item.prompt_en,
        "target_es": item.target_es,
        "gloss_en": item.gloss_en,
        "scores": {k: v.__dict__ for k, v in scores.items()},
        "nearest": nearest,
    }
