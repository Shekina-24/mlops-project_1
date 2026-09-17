"""Train the profession classifier, tune it lightly, and push it to the HF Hub.

Usage:
    python src/train.py [--config configs/config.yaml] [--no-push] [--hf-token TOKEN]

Steps:
    1. Load the processed train split.
    2. Build the pipeline (TF-IDF + classifier) from config.
    3. Run a small RandomizedSearchCV/GridSearchCV over `clf__C` (fast on CPU).
    4. Quickly sanity-check on the dev split.
    5. Save the fitted pipeline locally with skops and (unless --no-push)
       initialize/push a Hugging Face Hub model repo with a first-draft model
       card. `src/evaluate.py` fills in real metrics and the fairness report
       afterwards, as a follow-up commit to the same repo.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.pipeline import build_pipeline
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def _load_split(config: Config, split: str) -> pd.DataFrame:
    path = config.root / config.data.processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/preprocess.py` first.")
    return pd.read_parquet(path)


def _load_label_names(config: Config) -> dict:
    path = config.root / config.data.processed_dir / "label_names.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_hyperparameter_search(config: Config, X_train, y_train):
    pipeline = build_pipeline(config)
    hp = config.hyperparameter_search
    param_grid = hp.param_distributions.get(config.model.type, {})

    if not hp.enabled or not param_grid:
        logger.info("Hyperparameter search disabled or empty grid; fitting default pipeline.")
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    search_cls = RandomizedSearchCV if hp.method == "randomized" else GridSearchCV
    kwargs = dict(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=hp.n_iter,
        cv=hp.cv_folds,
        scoring=hp.scoring,
        n_jobs=hp.n_jobs,
        random_state=config.seed,
        verbose=1,
    )
    if search_cls is GridSearchCV:
        kwargs.pop("param_distributions")
        kwargs.pop("n_iter")
        kwargs.pop("random_state")
        kwargs["param_grid"] = param_grid

    logger.info(
        "Running %s over %s (cv=%d, scoring=%s)...",
        hp.method,
        param_grid,
        hp.cv_folds,
        hp.scoring,
    )
    search = search_cls(**kwargs)
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Search finished in %.1fs. Best params: %s (score=%.4f)", elapsed, search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_


def run_training(config_path: str | None = None, push: bool = True, hf_token: str | None = None) -> None:
    config = load_config(config_path)
    set_seed(config.seed)

    train_df = _load_split(config, "train")
    dev_df = _load_split(config, "dev") if (config.root / config.data.processed_dir / "dev.parquet").exists() else None
    label_names = _load_label_names(config)

    text_col, target_col = config.data.text_column, config.data.target_column
    X_train, y_train = train_df[text_col], train_df[target_col]

    logger.info("Training on %d rows (model type: %s)", len(train_df), config.model.type)
    best_pipeline, best_params = run_hyperparameter_search(config, X_train, y_train)

    dev_metrics = {}
    if dev_df is not None and len(dev_df) > 0:
        X_dev, y_dev = dev_df[text_col], dev_df[target_col]
        y_pred = best_pipeline.predict(X_dev)
        dev_metrics = {
            "accuracy": round(accuracy_score(y_dev, y_pred), 4),
            "f1_macro": round(f1_score(y_dev, y_pred, average="macro"), 4),
            "n_samples": len(dev_df),
        }
        logger.info("Dev metrics: %s", dev_metrics)

    # --- Save locally with skops -------------------------------------------------
    models_dir = config.root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tmp_skops_path = models_dir / "model.skops"

    import skops.io as sio

    sio.dump(best_pipeline, tmp_skops_path)
    logger.info("Saved fitted pipeline -> %s", tmp_skops_path)

    local_repo_dir = config.root / config.artifacts.local_model_dir
    if local_repo_dir.exists():
        shutil.rmtree(local_repo_dir)

    from skops import hub_utils

    example_texts = X_train.head(3).tolist()
    hub_utils.init(
        model=tmp_skops_path,
        requirements=["scikit-learn", "pandas", "numpy", "skops"],
        dst=local_repo_dir,
        task="text-classification",
        data=example_texts,
    )
    tmp_skops_path.unlink(missing_ok=True)
    logger.info("Initialized Hugging Face model repo folder -> %s", local_repo_dir)

    # Persist label names + config snapshot alongside the model for evaluate.py / the app.
    with open(local_repo_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2, ensure_ascii=False)
    with open(local_repo_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": config.model.type,
                "best_params": best_params,
                "dev_metrics": dev_metrics,
                "n_train_samples": len(train_df),
                "seed": config.seed,
            },
            f,
            indent=2,
        )

    # --- Draft model card (evaluate.py rebuilds the final version with fairness results) --
    from src.utils import build_base_model_card

    model_card = build_base_model_card(config, best_pipeline, best_params, dev_metrics)
    model_card.save(local_repo_dir / "README.md")
    logger.info("Wrote draft model card -> %s", local_repo_dir / "README.md")

    if not push:
        logger.info("--no-push set; skipping Hugging Face Hub upload.")
        return

    repo_id = config.huggingface.model_repo
    hub_utils.push(
        repo_id=repo_id,
        source=local_repo_dir,
        token=hf_token,
        commit_message="Train and push profession classifier pipeline",
        create_remote=True,
        private=config.huggingface.private,
    )
    logger.info("Pushed model to https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to the HF Hub")
    parser.add_argument("--hf-token", default=None, help="HF Hub token (else HF_TOKEN env var or local login)")
    args = parser.parse_args()
    run_training(args.config, push=not args.no_push, hf_token=args.hf_token)
