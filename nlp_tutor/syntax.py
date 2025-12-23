# nlp_tutor/syntax.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from .languages import Lang
from .spacy_loader import load_nlp
from . import preprocessing as prep


@dataclass
class SyntaxFeatures:
    n_tokens: int
    n_sents: int
    n_verbs: int
    n_nouns: int
    n_adjs: int
    n_advs: int
    root_pos: str | None
    has_main_verb: bool


@dataclass
class SyntaxIssue:
    code: str
    message: str


@dataclass
class SyntaxReport:
    lang: Lang
    features: SyntaxFeatures
    issues: List[SyntaxIssue]
    debug: Dict[str, Any]  # optional extra info for notebook/report


def analyse_syntax(text: str, lang: Lang) -> SyntaxReport:
    nlp = load_nlp(lang)

    cleaned = prep.normalise(text)
    doc = nlp(cleaned)

    tokens = [t for t in doc if not t.is_space]
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

    n_verbs = sum(1 for t in tokens if t.pos_ in ("VERB", "AUX"))
    n_nouns = sum(1 for t in tokens if t.pos_ in ("NOUN", "PROPN", "PRON"))
    n_adjs = sum(1 for t in tokens if t.pos_ == "ADJ")
    n_advs = sum(1 for t in tokens if t.pos_ == "ADV")

    root_pos = None
    if sents:
        root = next((t for t in sents[0] if t.dep_ == "ROOT"), None)
        root_pos = root.pos_ if root else None

    has_main_verb = n_verbs > 0

    features = SyntaxFeatures(
        n_tokens=len(tokens),
        n_sents=len(sents),
        n_verbs=n_verbs,
        n_nouns=n_nouns,
        n_adjs=n_adjs,
        n_advs=n_advs,
        root_pos=root_pos,
        has_main_verb=has_main_verb,
    )

    issues: List[SyntaxIssue] = []

    if not has_main_verb and len(tokens) >= 3:
        issues.append(SyntaxIssue(
            code="MISSING_VERB",
            message="This sentence appears to be missing a main verb (an action word)."
        ))

    if features.n_tokens >= 6 and features.n_verbs == 0 and features.n_nouns >= 4:
        issues.append(SyntaxIssue(
            code="NOUN_HEAVY_FRAGMENT",
            message="This looks like a noun-heavy fragment. Try adding a verb to make a complete sentence."
        ))

    if features.n_tokens >= 25:
        issues.append(SyntaxIssue(
            code="LONG_SENTENCE",
            message="This sentence is quite long. Consider splitting it into two shorter sentences."
        ))

    if lang == Lang.ES and features.n_verbs == 0 and features.n_tokens >= 4:
        issues.append(SyntaxIssue(
            code="ES_VERB_EXPECTED",
            message="In Spanish, most complete sentences need a conjugated verb. Try adding the main verb."
        ))

    debug = {
        "tokens": [(t.text, t.pos_, t.dep_, t.head.text) for t in tokens[:40]],
        "sent_texts": [s.text for s in sents[:5]],
    }

    return SyntaxReport(lang=lang, features=features, issues=issues, debug=debug)
