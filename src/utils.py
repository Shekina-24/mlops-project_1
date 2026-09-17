"""Small reusable helpers: seeding, logging, text cleaning, HF Hub auth."""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Fix seeds for reproducibility across random, numpy (and sklearn via numpy)."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    lowercase: bool = False,
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Light, reversible cleaning applied to raw bios before vectorization.

    Kept intentionally minimal: TF-IDF is robust to punctuation/casing, and we
    do not want to destroy signal that TfidfVectorizer's own tokenizer can use.
    """
    if text is None:
        return ""
    cleaned = str(text)
    if collapse_whitespace:
        cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    if strip_whitespace:
        cleaned = cleaned.strip()
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def get_hf_token(explicit_token: str | None = None) -> str | None:
    """Resolve a Hugging Face Hub token from an explicit arg, env var, or local login.

    Returns None if no token can be found; callers decide whether that's fatal.
    """
    import os

    if explicit_token:
        return explicit_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def push_folder_to_hub(
    local_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Update model artifacts",
) -> str:
    """Create (if needed) a model repo on the HF Hub and push a local folder to it.

    Returns the resulting repo URL.
    """
    from huggingface_hub import HfApi, create_repo

    resolved_token = get_hf_token(token)
    if resolved_token is None:
        raise RuntimeError(
            "No Hugging Face token found. Run `huggingface-cli login` or set the "
            "HF_TOKEN environment variable before pushing to the Hub."
        )

    create_repo(
        repo_id=repo_id,
        token=resolved_token,
        private=private,
        repo_type="model",
        exist_ok=True,
    )
    api = HfApi()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=resolved_token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def build_base_model_card(config, pipeline, best_params: dict, dev_metrics: dict):
    """Build the skops Card shared by train.py (first push) and evaluate.py
    (final push): base metadata + description sections. Rebuilt from scratch
    each time (rather than parsing the previous README back) since
    `skops.card.parse_modelcard` requires an external pandoc install.
    """
    from skops import card

    model_card = card.Card(pipeline, metadata=None, trusted=True)
    model_card.metadata.license = "mit"
    model_card.metadata.library_name = "skops"
    model_card.metadata.tags = ["sklearn", "text-classification", "fairness", "bias-in-bios"]

    model_card.add(
        **{
            "Model description": (
                "Multinomial classifier predicting a person's **profession** (28 classes) "
                "from the free-text `hard_text` biography in the "
                "[LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios) dataset "
                "(derived from De-Arteaga et al., *Bias in Bios*, FAccT 2019, arXiv:1901.09451).\n\n"
                "**This is an educational / research demo project studying gender bias in automated "
                "resume screening. It is NOT intended, tested, or suitable for real hiring or HR "
                "decisions.**"
            ),
            "Model description/Training Procedure": (
                f"TF-IDF vectorizer + `{config.model.type}` (scikit-learn), tuned with "
                f"{config.hyperparameter_search.method} search over `clf__C` "
                f"(cv={config.hyperparameter_search.cv_folds}, scoring={config.hyperparameter_search.scoring}).\n\n"
                f"Best hyperparameters: `{best_params}`."
            ),
            "Model description/Intended uses & limitations": (
                "Intended use: demonstrating and studying gender bias in text-based occupation "
                "classifiers. Out of scope: any real candidate screening, resume filtering, or "
                "employment decision. See the 'Bias, Risks and Limitations' section below for the "
                "full fairness audit."
            ),
        }
    )
    if dev_metrics:
        model_card.add_metrics(
            section="Model description/Evaluation Results",
            description="Metrics computed on the `dev` split right after training (sanity check).",
            **{k: v for k, v in dev_metrics.items() if k != "n_samples"},
        )
    return model_card


def load_skops_pipeline(path: str | Path):
    """Load a scikit-learn pipeline saved with skops.io.dump.

    We trust the artifact because it is produced by our own src/train.py, so
    we auto-approve whatever custom types skops flags (TfidfVectorizer,
    LogisticRegression/LinearSVC, CalibratedClassifierCV, numpy dtypes, etc.)
    rather than hardcoding a type list that would break on every model change.
    """
    import skops.io as sio

    untrusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted)


def download_file_from_hub(repo_id: str, filename: str, token: str | None = None) -> str:
    """Download a single file from a model repo on the HF Hub, returning the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, token=get_hf_token(token))
