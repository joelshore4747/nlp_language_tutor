from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from ..corpora import load_imdb_kaggle
from ..config import PATHS
from .. import preprocessing as prep


MODEL_NAME = "sentiment_tfidf_logreg.joblib"


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    f1_macro: float
    confusion: List[List[int]]
    report: str


def build_pipeline() -> Pipeline:
    """
    Strong classical baseline:
    - TF-IDF with uni+bi-grams
    - Logistic Regression (linear classifier)
    """
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,   # keep portable; some environments ignore n_jobs anyway
        solver="lbfgs"
    )

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    return Pipeline([
        ("tfidf", vec),
        ("clf", clf),
    ])


def train_evaluate_save(
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: Path | None = None,
) -> TrainResult:

    prep.ensure_nltk_resources()
    ds = load_imdb_kaggle()

    X = [prep.normalise(t) for t in ds.texts]
    y = [str(lbl) for lbl in ds.labels]  # 'positive'/'negative'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"]).tolist()
    rep = classification_report(y_test, y_pred)

    if save_path is None:
        save_path = PATHS.models_dir / MODEL_NAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, save_path)

    return TrainResult(
        accuracy=acc,
        f1_macro=f1m,
        confusion=cm,
        report=rep,
    )


def load_model(path: Path | None = None) -> Pipeline:
    if path is None:
        path = PATHS.models_dir / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first:\n"
            f"  python scripts/train_sentiment_baseline.py\n"
        )
    return joblib.load(path)


def predict_sentiment(text: str, model: Pipeline | None = None) -> Dict[str, str]:
    if model is None:
        model = load_model()

    x = prep.normalise(text)
    pred = model.predict([x])[0]
    # optional: probability if supported
    out = {"label": str(pred)}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([x])[0]
        # pipeline clf classes
        classes = list(model.named_steps["clf"].classes_)
        probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
        out["probs"] = str(probs)
    return out
