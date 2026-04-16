def test_tutor_response_schema_accepts_fluency_object():
    from api.schemas import TutorEvaluateResponse, TutorAction, LabelScore, SimilarityResultOut, FluencyOut

    resp = TutorEvaluateResponse(
        prompt_en="Hello",
        target_es="Hola",
        gloss_en="Hello",
        expected_lang="ES",
        detected_lang="ES",
        detected_top_k=[LabelScore(label="ES", score=0.95)],
        syntax_issues=[],
        fluency=FluencyOut(perplexity=123.4, band="medium"),
        ner={"entities": [], "noun_phrases": []},
        semantics={"sbert": SimilarityResultOut(backend="sbert", score=0.9, interpretation="high")},
        nearest={"sbert": [("Hola", 0.9)]},
        action=TutorAction(code="ADVANCE", message="OK", severity="info"),
    )

    d = resp.model_dump()
    assert isinstance(d["fluency"], dict)
    assert "perplexity" in d["fluency"]
    assert "band" in d["fluency"]
