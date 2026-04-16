from nlp_tutor.classification.lang_detect_baseline import build_pipeline, predict_language


def test_language_detect_baseline_output_shape():
    model = build_pipeline()

    X = [
        "Hola amigo", "Hola amigo",
        "Hello friend", "Hello friend",
        "Cześć kolego", "Cześć kolego",
        "Bonjour mon ami", "Bonjour mon ami",
    ]
    y = ["ES", "ES", "EN", "EN", "PL", "PL", "FR", "FR"]

    model.fit(X, y)

    out = predict_language("Hola, ¿cómo estás?", model=model, top_k=3)

    assert "label" in out
    assert "top_k" in out
    assert isinstance(out["top_k"], list)
    assert len(out["top_k"]) <= 3
