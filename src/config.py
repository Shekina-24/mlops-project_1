"""Typed access to configs/config.yaml.

Every other script imports `load_config()` from this module instead of
reading the YAML file directly, so the schema lives in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class DataConfig:
    dataset_repo: str
    raw_dir: str
    processed_dir: str
    text_column: str
    target_column: str
    sensitive_column: str
    num_labels: int


@dataclass
class SamplingConfig:
    enabled: bool
    total_samples: int | None
    stratify_by: list[str]
    min_samples_per_group: int


@dataclass
class SplitConfig:
    use_original_splits: bool
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class PreprocessingConfig:
    lowercase: bool
    strip_whitespace: bool
    collapse_whitespace: bool
    min_text_length: int


@dataclass
class ModelConfig:
    type: str
    tfidf: dict[str, Any]
    logistic_regression: dict[str, Any]
    linear_svc: dict[str, Any]


@dataclass
class HyperparameterSearchConfig:
    enabled: bool
    method: str
    n_iter: int
    cv_folds: int
    scoring: str
    n_jobs: int
    param_distributions: dict[str, dict[str, Any]]


@dataclass
class FairnessConfig:
    sensitive_column: str
    gender_labels: dict[int, str]
    top_k_imbalanced_professions: int


@dataclass
class HuggingFaceConfig:
    dataset_repo: str
    model_repo: str
    space_repo: str
    private: bool


@dataclass
class ArtifactsConfig:
    local_model_dir: str
    fairness_report_path: str
    metrics_path: str


@dataclass
class Config:
    seed: int
    data: DataConfig
    sampling: SamplingConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    hyperparameter_search: HyperparameterSearchConfig
    fairness: FairnessConfig
    huggingface: HuggingFaceConfig
    artifacts: ArtifactsConfig
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def root(self) -> Path:
        return PROJECT_ROOT


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate configs/config.yaml into a Config object."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        sampling=SamplingConfig(**raw["sampling"]),
        split=SplitConfig(**raw["split"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        model=ModelConfig(**raw["model"]),
        hyperparameter_search=HyperparameterSearchConfig(**raw["hyperparameter_search"]),
        fairness=FairnessConfig(**raw["fairness"]),
        huggingface=HuggingFaceConfig(**raw["huggingface"]),
        artifacts=ArtifactsConfig(**raw["artifacts"]),
        raw=raw,
    )
