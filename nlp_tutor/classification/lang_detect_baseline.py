from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from nlp_tutor.config import PATHS
from nlp_tutor.corpora import TextLabelDataset
from nlp_tutor import preprocessing as prep


MODEL_NAME = "lang_detect_char_tfidf_logreg.joblib"


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    f1_macro: float
    confusion: List[List[int]]
    labels: List[str]
    report: str


def build_pipeline() -> Pipeline:
    """
    Character n-gram TF-IDF + Logistic Regression.
    Strong baseline for language identification.
    """
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    return Pipeline([
        ("tfidf", vec),
        ("clf", clf),
    ])

def predict_language_with_confidence(
    text: str,
    model: Pipeline | None = None,
    threshold: float = 0.70,
) -> Dict:
    """
    If confidence is low, return 'UNCERTAIN' instead of guessing.
    """
    if model is None:
        model = load_model()

    x = prep.normalise(text)

    if not hasattr(model, "predict_proba"):
        pred = model.predict([x])[0]
        return {"label": str(pred), "confidence": None}

    probs = model.predict_proba([x])[0]
    classes = list(model.named_steps["clf"].classes_)
    best_i = int(probs.argmax())
    best_label = str(classes[best_i])
    best_prob = float(probs[best_i])

    if best_prob < threshold:
        return {"label": "UNCERTAIN", "confidence": best_prob, "best_guess": best_label}

    return {"label": best_label, "confidence": best_prob}



def _filter_langs(ds: TextLabelDataset, target_langs: Optional[Set[str]]) -> TextLabelDataset:
    if not target_langs:
        return ds

    texts: List[str] = []
    labels: List[str] = []
    for t, y in zip(ds.texts, ds.labels):
        if str(y) in target_langs:
            texts.append(t)
            labels.append(str(y))

    if len(set(labels)) < 2:
        raise ValueError(
            f"After filtering, not enough classes. "
            f"Found={sorted(set(labels))}, target_langs={sorted(target_langs)}"
        )

    return TextLabelDataset(texts=texts, labels=labels)


def train_eval_save(
    ds: TextLabelDataset,
    target_langs: Optional[Set[str]] = None,
    max_chars: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: Path | None = None,
) -> TrainResult:
    """
    Train + evaluate on any TextLabelDataset.
    The dataset source (WiLI/Kaggle/custom) is handled elsewhere.
    """

    ds = _filter_langs(ds, target_langs)

    # Preprocess + truncate here (THIS is where [:50] belongs)
    X = [prep.normalise(t)[:max_chars] for t in ds.texts]
    y = [str(lbl) for lbl in ds.labels]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    labels_sorted = sorted(set(y))
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()
    rep = classification_report(y_test, y_pred)

    if save_path is None:
        save_path = PATHS.models_dir / MODEL_NAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, save_path)

    return TrainResult(
        accuracy=acc,
        f1_macro=f1m,
        confusion=cm,
        labels=labels_sorted,
        report=rep,
    )


def load_model(path: Path | None = None) -> Pipeline:
    if path is None:
        path = PATHS.models_dir / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Language model not found at {path}. Train it first."
        )
    return joblib.load(path)


def predict_language(text: str, model: Pipeline | None = None, top_k: int = 5) -> Dict:
    if model is None:
        model = load_model()

    x = prep.normalise(text)
    pred = model.predict([x])[0]

    out = {"label": str(pred), "top_k": []}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([x])[0]
        classes = list(model.named_steps["clf"].classes_)
        scored = sorted(
            [(classes[i], float(probs[i])) for i in range(len(classes))],
            key=lambda t: t[1],
            reverse=True
        )[:top_k]
        out["top_k"] = scored

    return out


def short_text_sanity_check(model: Pipeline) -> List[Tuple[str, str]]:
    tests = [
        "yes", "no", "ok", "thanks",
        "sí", "gracias", "hola",
        "tak", "nie", "dzień dobry",
        "przepraszam", "co tam",
    ]
    return [(t, str(model.predict([prep.normalise(t)])[0])) for t in tests]


def detect_mixed_language(
    text: str,
    model: Pipeline | None = None,
    delta: float = 0.10,
) -> Dict:
    if model is None:
        model = load_model()

    x = prep.normalise(text)
    if not hasattr(model, "predict_proba"):
        return {"label": str(model.predict([x])[0]), "mixed": False}

    probs = model.predict_proba([x])[0]
    classes = list(model.named_steps["clf"].classes_)

    pairs = sorted([(str(classes[i]), float(probs[i])) for i in range(len(classes))],
                   key=lambda p: p[1], reverse=True)

    top1, top2 = pairs[0], pairs[1]
    mixed = (top1[1] - top2[1]) < delta

    return {"top1": top1, "top2": top2, "mixed": mixed}
