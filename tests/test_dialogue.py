from nlp_tutor.dialogue_policy import choose_action


def test_fluency_is_soft_nudge_not_block():
    action = choose_action(
        expected_lang="ES",
        detected_lang="ES",
        lang_conf_ok=True,
        syntax_issues=[],
        semantic_score=0.9,     # good semantics
        fluency_band="low",     # low fluency should nudge
    )
    assert action.code == "FLUENCY_NUDGE"
    assert action.severity == "info"
