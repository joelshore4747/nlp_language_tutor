from api.schemas import TutorEvaluateResponse, TutorAction, LabelScore, SimilarityResultOut, FluencyOut


def test_tutor_evaluate_response_accepts_fluency_object():
    payload = {
        "prompt_en": "My name is Joel.",
        "target_es": "Me llamo Joel.",
        "gloss_en": "My name is Joel.",
        "expected_lang": "ES",
        "detected_lang": "ES",
        "detected_top_k": [LabelScore(label="ES", score=0.95)],
        "syntax_issues": [],
        "fluency": FluencyOut(perplexity=120.0, band="medium"),
        "ner": {"entities": [], "noun_phrases": []},
        "semantics": {
            "tfidf": SimilarityResultOut(backend="tfidf", score=0.9, interpretation="high")
        },
        "nearest": {"tfidf": [("Me llamo Joel.", 0.9)]},
        "action": TutorAction(code="ADVANCE", message="Looks good.", severity="info"),
    }

    model = TutorEvaluateResponse(**payload)
    dumped = model.model_dump()
    assert isinstance(dumped["fluency"], dict)
    assert "perplexity" in dumped["fluency"]
    assert "band" in dumped["fluency"]
