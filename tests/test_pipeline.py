"""Sanity checks: config loads, the pipeline builds and fits, preprocessing
produces the expected columns. Run with `pytest` (see Makefile `test` target).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pipeline import build_pipeline
from src.preprocess import clean_dataframe, preprocess_split, stratified_subsample


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def synthetic_raw_df():
    texts = [
        "He is a surgeon with over ten years of experience in cardiac care.",
        "She works as a nurse in the pediatric ward of a large hospital.",
        "He teaches high school mathematics and coaches the debate team.",
        "She is a software engineer specializing in distributed systems.",
        "He is a professor of philosophy at a small liberal arts college.",
        "She is a dietitian helping clients build healthier eating habits.",
        "x",  # too short, should be dropped after cleaning
        "He develops backend services and mentors junior engineers daily.",
    ]
    professions = [26, 13, 27, 25, 22, 7, 13, 25]  # arbitrary ids, don't need to be real
    genders = [0, 1, 0, 1, 0, 1, 1, 0]
    return pd.DataFrame({"hard_text": texts, "profession": professions, "gender": genders})


def test_config_loads(config):
    assert config.data.target_column == "profession"
    assert config.data.sensitive_column == "gender"
    assert config.data.num_labels == 28
    assert config.model.type in {"logistic_regression", "linear_svc"}


def test_build_pipeline_returns_valid_pipeline(config):
    pipeline = build_pipeline(config)
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]


def test_pipeline_fits_and_predicts_without_crashing(config):
    pipeline = build_pipeline(config)
    X = pd.Series(
        [
            "a nurse caring for patients in the emergency room",
            "a software engineer writing python and building apps",
            "a teacher grading papers and planning lessons",
        ]
        * 4
    )
    y = pd.Series([0, 1, 2] * 4)

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(y.unique()))

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 3)


def test_clean_dataframe_drops_short_rows_and_expected_columns(config, synthetic_raw_df):
    cleaned = clean_dataframe(synthetic_raw_df, config)
    assert set(["hard_text", "profession", "gender"]).issubset(cleaned.columns)
    # the single-character row "x" must be dropped (min_text_length filter)
    assert "x" not in cleaned["hard_text"].tolist()
    assert len(cleaned) == len(synthetic_raw_df) - 1


def test_stratified_subsample_keeps_small_groups_whole(synthetic_raw_df, config):
    cleaned = clean_dataframe(synthetic_raw_df, config)
    sampled = stratified_subsample(
        cleaned,
        stratify_by=["profession", "gender"],
        frac=0.5,
        min_samples_per_group=1,
        seed=config.seed,
    )
    assert len(sampled) <= len(cleaned)
    assert set(sampled.columns) == set(cleaned.columns)


def test_preprocess_split_end_to_end(synthetic_raw_df, config):
    processed = preprocess_split(synthetic_raw_df, config, target_rows=None)
    assert "hard_text" in processed.columns
    assert "profession" in processed.columns
    assert "gender" in processed.columns
    assert len(processed) > 0
    assert processed["hard_text"].str.len().min() >= config.preprocessing.min_text_length
