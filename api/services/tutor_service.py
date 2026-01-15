from __future__ import annotations

from typing import Dict, List, Tuple

from api.schemas import TutorEvaluateRequest, TutorEvaluateResponse, TutorAction, LabelScore, SimilarityResultOut, FluencyOut
from api.state import get_state

from nlp_tutor.languages import Lang


def tutor_evaluate(req: TutorEvaluateRequest) -> TutorEvaluateResponse:
    state = get_state()

    item = None
    if req.target_text is None or req.prompt_en is None or req.gloss_en is None:
        key = (req.lesson_id, req.item_id)
        if key not in state.index:
            raise KeyError(f"Lesson item not found: {key}")
        item = state.index[key]

    prompt_en = req.prompt_en or (item.prompt_en if item else "")
    target_text = req.target_text or (item.target_es if item else "")
    gloss_en = req.gloss_en or (item.gloss_en if item else "")

    if state.pipeline is None:
        raise RuntimeError("Tutor pipeline not initialized.")

    res = state.pipeline.analyze(
        text=req.learner_text,
        expected_lang=req.expected_lang,
        target_text=target_text
    )

    scores_out: Dict[str, SimilarityResultOut] = {}
    for name, r in res.semantic_scores.items():
        scores_out[name] = SimilarityResultOut(
            backend=r.backend,
            score=r.score,
            interpretation=r.interpretation,
        )

    return TutorEvaluateResponse(
        prompt_en=prompt_en,
        target_es=target_text,
        gloss_en=gloss_en,
        expected_lang=req.expected_lang.upper(),
        detected_lang=res.detected_lang,
        detected_top_k=[LabelScore(label=ls.label, score=ls.score) for ls in res.detected_top_k],
        syntax_issues=res.syntax_issues,
        fluency=(
            FluencyOut(perplexity=res.fluency_perplexity, band=res.fluency_band)
            if (res.fluency_band is not None and res.fluency_perplexity is not None)
            else None
        ),
        ner={
            "entities": res.entities,
            "noun_phrases": res.noun_phrases,
        },
        semantics=scores_out,
        nearest=res.nearest_targets,
        action=TutorAction(code=res.action_code, message=res.action_message, severity=res.action_severity),
    )
