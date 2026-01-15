from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

from nlp_tutor.languages import Lang


@dataclass(frozen=True)
class TutorAction:
    code: str
    message: str
    severity: str  # "info" | "warn" | "block"


def choose_action(
    *,
    expected_lang: str,
    detected_lang: str,
    lang_conf_ok: bool,
    syntax_issues: list,
    semantic_score: float | None,
    fluency_band: str | None,
):
    # 1) Language gate
    if detected_lang != expected_lang:
        return TutorAction(
            code="LANGUAGE_MISMATCH",
            message=f"Please answer in {expected_lang}.",
            severity="warn",
        )

    if not lang_conf_ok:
        return TutorAction(
            code="LANGUAGE_UNCERTAIN",
            message="I’m not fully confident about the language. Please write a longer sentence.",
            severity="warn",
        )

    # 2) Syntax gate (only if same language)
    if syntax_issues:
        return TutorAction(
            code="SYNTAX_FIX",
            message="There are syntax issues to fix before we check meaning.",
            severity="warn",
        )

    # 3) Meaning gate
    if semantic_score is None:
        return TutorAction(
            code="NO_SEMANTICS",
            message="Semantic scoring is unavailable right now.",
            severity="info",
        )

    # These thresholds are reasonable starting points
    if semantic_score < 0.80:
        return TutorAction(
            code="MEANING_MISMATCH",
            message="You’re close, but key meaning is missing or different. Try completing the idea.",
            severity="warn",
        )

    # 4) Fluency only after meaning is acceptable
    if fluency_band == "low":
        return TutorAction(
            code="FLUENCY_NUDGE",
            message="Good attempt. Try rephrasing for a more natural expression.",
            severity="info",
        )

    return TutorAction(
        code="GOOD",
        message="Good job. That matches the meaning.",
        severity="info",
    )

