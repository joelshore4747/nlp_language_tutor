from __future__ import annotations
import re
from nlp_tutor import preprocessing as prep
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from nlp_tutor.languages import Lang
from nlp_tutor.ner import extract_ner
from nlp_tutor.syntax import analyse_syntax
from nlp_tutor.classification.lang_detect_baseline import predict_language as predict_lang_baseline, is_confident
from nlp_tutor.classification.lang_detect_bilstm_chars import predict_language as predict_lang_bilstm
from nlp_tutor.dialogue_policy import choose_action
from nlp_tutor.semantics import SimilarityResult

MIN_TEXT_CHARS = 12
MIN_WORD_TOKENS = 3

@dataclass
class LabelScore:
    label: str
    score: float

@dataclass
class AnalysisResult:
    detected_lang: str
    lang_confidence: float
    lang_conf_ok: bool
    detected_top_k: List[LabelScore]
    syntax_issues: List[Dict[str, str]]
    semantic_scores: Dict[str, Any]
    nearest_targets: Dict[str, List[Any]]
    fluency_band: Optional[str]
    fluency_perplexity: Optional[float]
    entities: List[Dict[str, Any]]
    noun_phrases: List[str]
    action_code: str
    action_message: str
    action_severity: str

class TutorPipeline:
    
    def __init__(
        self,
        lang_model,
        semantic_engine,
        fluency_scorer=None,
        bilstm_model=None,
        bilstm_pack=None
    ):
        self.lang_model = lang_model
        self.semantic_engine = semantic_engine
        self.fluency_scorer = fluency_scorer
        self.bilstm_model = bilstm_model
        self.bilstm_pack = bilstm_pack

    def analyze(self, text: str, expected_lang: str, target_text: str) -> AnalysisResult:
        norm = prep.normalise(text).strip()
        word_tokens = re.findall(r"\w+", norm)

        if len(norm.replace(" ", "")) < MIN_TEXT_CHARS or len(word_tokens) < MIN_WORD_TOKENS:
            return AnalysisResult(
                detected_lang="UNCERTAIN",
                lang_confidence=0.0,
                lang_conf_ok=False,
                detected_top_k=[],
                syntax_issues=[],
                semantic_scores={},
                nearest_targets={},
                fluency_band=None,
                fluency_perplexity=None,
                entities=[],
                noun_phrases=[],
                action_code="INPUT_TOO_SHORT",
                action_message=(
                    f"Please enter a longer sentence (at least {MIN_WORD_TOKENS} words) "
                    "so I can evaluate it reliably."
                ),
                action_severity="warn",
            )


        if self.bilstm_model is not None:
            lang_out = predict_lang_bilstm(
                text,
                model=self.bilstm_model,
                char2idx=self.bilstm_pack["char2idx"],
                idx2label=self.bilstm_pack["idx2label"],
                config=self.bilstm_pack["config"],
                top_k=5
            )
        else:
            lang_out = predict_lang_baseline(text, model=self.lang_model, top_k=5)
        
        detected_raw = str(lang_out["label"])
        try:
            detected = Lang.parse(detected_raw).name
        except Exception:
            detected = detected_raw

        top_tuples = lang_out.get("top_k", [])
        top = [LabelScore(label=l, score=s) for (l, s) in top_tuples]
        lang_conf_ok = is_confident(top_tuples)

        expected_enum = Lang.parse(expected_lang)

        lang_matches = (detected == expected_enum.name)
        lang_ok = (lang_matches and lang_conf_ok)

        if detected == expected_enum.name:
            syn = analyse_syntax(lang=expected_enum, text=text)
            syntax_issues = [{"code": i.code, "message": i.message} for i in syn.issues]
        else:
            syntax_issues = []


        scores = self.semantic_engine.score_answer(text, target_text, lang=expected_enum)
        nearest = self.semantic_engine.nearest_targets(text, lang=expected_enum, k=5)
        

        transformer_score = None
        if hasattr(self.semantic_engine, "score_transformer"):
            transformer_score = float(self.semantic_engine.score_transformer(text, target_text, lang=expected_enum))
            scores["transformer_cosine"] = SimilarityResult(
                backend="sbert",
                score=transformer_score,
                interpretation=self.semantic_engine.interpret_similarity(transformer_score)
            )
            if hasattr(self.semantic_engine, "nearest_targets_transformer"):
                nearest["transformer_cosine"] = self.semantic_engine.nearest_targets_transformer(
                    text, lang=expected_enum, k=5
                )
        if hasattr(self.semantic_engine, "nearest_targets_by_lang"):
            try:
                nearest["by_lang"] = self.semantic_engine.nearest_targets_by_lang(text, k_per_lang=1)
            except Exception:
                pass


        if transformer_score is None and "sbert" in scores:
            transformer_score = scores["sbert"].score


        ner_lang = None
        if detected == expected_enum.name:
            ner_lang = expected_enum
        elif lang_conf_ok:
            try:
                ner_lang = Lang.parse(detected)
            except Exception:
                ner_lang = None

        if ner_lang is not None:
            try:
                ner_r = extract_ner(lang=ner_lang, text=text)
            except Exception:
                ner_r = type("Tmp", (), {"entities": [], "noun_phrases": []})()
        else:
            ner_r = type("Tmp", (), {"entities": [], "noun_phrases": []})()  # or a small NerOut dataclass

        # 5) Fluency
        fluency_band = None
        fluency_perplexity = None
        if self.fluency_scorer:
            try:
                f_score = self.fluency_scorer.score_text(text, lang=expected_enum)
                fluency_band = f_score.band
                fluency_perplexity = float(f_score.perplexity)
            except Exception:
                fluency_band = None
                fluency_perplexity = None

        tfidf_score = None
        if "tfidf" in scores:
            try:
                tfidf_score = float(scores["tfidf"].score)
            except Exception:
                tfidf_score = None

        policy_semantic = transformer_score
        if detected == expected_enum.name and tfidf_score is not None:
            policy_semantic = tfidf_score

        fluency_for_policy = fluency_band
        if policy_semantic is not None and policy_semantic >= 0.95:
            fluency_for_policy = None

        action = choose_action(
            expected_lang=expected_enum.name,
            detected_lang=detected,
            lang_conf_ok=lang_conf_ok,
            syntax_issues=syntax_issues,
            semantic_score=policy_semantic,
            fluency_band=fluency_for_policy,
        )

        return AnalysisResult(
            detected_lang=detected,
            lang_confidence=top[0].score if top else 0.0,
            lang_conf_ok=lang_conf_ok,
            detected_top_k=top,
            syntax_issues=syntax_issues,
            semantic_scores=scores,
            nearest_targets=nearest,
            fluency_band=fluency_band,
            fluency_perplexity=fluency_perplexity,
            entities=[e.__dict__ for e in ner_r.entities],
            noun_phrases=ner_r.noun_phrases,
            action_code=action.code,
            action_message=action.message,
            action_severity=action.severity,
        )
