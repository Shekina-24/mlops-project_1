"""Build the scikit-learn Pipeline (TF-IDF + classifier) from config.

`build_pipeline` has no side effects (no I/O, no fitting) so it can be reused
identically by training, evaluation and tests.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import Config

_LOGREG_KEYS = {"C", "max_iter", "class_weight", "solver"}
_SVC_KEYS = {"C", "max_iter", "class_weight"}


def _build_vectorizer(config: Config) -> TfidfVectorizer:
    params = dict(config.model.tfidf)
    if "ngram_range" in params:
        params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def _build_classifier(config: Config):
    model_type = config.model.type
    if model_type == "logistic_regression":
        params = {k: v for k, v in config.model.logistic_regression.items() if k in _LOGREG_KEYS}
        return LogisticRegression(random_state=config.seed, **params)
    if model_type == "linear_svc":
        params = {k: v for k, v in config.model.linear_svc.items() if k in _SVC_KEYS}
        base = LinearSVC(random_state=config.seed, **params)
        # LinearSVC has no predict_proba; calibrate so the app can still show
        # top-5 profession probabilities regardless of the chosen model type.
        return CalibratedClassifierCV(base, cv=3)
    raise ValueError(f"Unknown model.type '{model_type}'. Use 'logistic_regression' or 'linear_svc'.")


def build_pipeline(config: Config) -> Pipeline:
    """Return an unfitted sklearn Pipeline: TfidfVectorizer -> classifier."""
    return Pipeline(
        steps=[
            ("tfidf", _build_vectorizer(config)),
            ("clf", _build_classifier(config)),
        ]
    )
