"""Evaluate the trained pipeline on the test split and audit it for gender bias.

Usage:
    python src/evaluate.py [--config configs/config.yaml] [--source local|hub] [--no-push]

Computes:
    - Overall accuracy / F1 macro on the held-out test split.
    - Accuracy / macro recall by gender (fairlearn MetricFrame).
    - True Positive Rate (recall) per (profession, gender), and the resulting
      TPR gap, which is the headline "does the model favor one gender for
      this profession" number from the Bias in Bios paper.
    - The professions with the largest male/female imbalance in the dataset
      itself, used to focus the report on the cases most likely to be biased.

Writes `metrics.json` and `fairness_report.json` to data/processed/, and
(unless --no-push) uploads both plus an updated model card to the existing
Hugging Face Hub model repo created by `src/train.py`.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.utils import get_logger

logger = get_logger(__name__)


def _load_split(config: Config, split: str) -> pd.DataFrame:
    path = config.root / config.data.processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/preprocess.py` first.")
    return pd.read_parquet(path)


def _load_label_names(config: Config) -> dict:
    path = config.root / config.data.processed_dir / "label_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pipeline_local(config: Config):
    from src.utils import load_skops_pipeline

    model_path = config.root / config.artifacts.local_model_dir / "model.skops"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python src/train.py` first.")
    return load_skops_pipeline(model_path)


def _load_pipeline_hub(config: Config, token: str | None = None):
    from src.utils import download_file_from_hub, load_skops_pipeline

    local_path = download_file_from_hub(config.huggingface.model_repo, "model.skops", token=token)
    return load_skops_pipeline(local_path)


def compute_gender_distribution(df: pd.DataFrame, config: Config, profession_names: list[str]) -> pd.DataFrame:
    """Per-profession male/female counts and imbalance, used to pick which
    professions the fairness report should highlight."""
    target_col, sensitive_col = config.data.target_column, config.data.sensitive_column
    gender_labels = config.fairness.gender_labels

    counts = (
        df.groupby([target_col, sensitive_col]).size().unstack(fill_value=0).rename(columns=gender_labels)
    )
    for label in gender_labels.values():
        if label not in counts.columns:
            counts[label] = 0

    counts["total"] = counts.sum(axis=1)
    counts["pct_female"] = counts.get("female", 0) / counts["total"].replace(0, pd.NA)
    counts["imbalance"] = (counts["pct_female"] - 0.5).abs()
    counts = counts.reset_index().rename(columns={target_col: "profession_id"})
    counts["profession"] = counts["profession_id"].apply(
        lambda i: profession_names[i] if i < len(profession_names) else str(i)
    )
    counts = counts.sort_values("imbalance", ascending=False)
    return counts


def compute_fairness_report(
    df: pd.DataFrame,
    y_pred_idx,
    config: Config,
    profession_names: list[str],
) -> dict:
    target_col, sensitive_col = config.data.target_column, config.data.sensitive_column
    gender_labels = config.fairness.gender_labels

    y_true_name = df[target_col].apply(lambda i: profession_names[i]).reset_index(drop=True)
    y_pred_name = pd.Series(y_pred_idx).apply(lambda i: profession_names[i]).reset_index(drop=True)
    gender_name = df[sensitive_col].map(gender_labels).reset_index(drop=True)

    # --- Overall performance by gender ------------------------------------------------
    recall_macro = functools.partial(recall_score, average="macro", zero_division=0)
    gender_mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall_macro": recall_macro},
        y_true=y_true_name,
        y_pred=y_pred_name,
        sensitive_features=gender_name,
    )
    by_gender = {
        gender: {k: round(v, 4) for k, v in row.items()}
        for gender, row in gender_mf.by_group.to_dict(orient="index").items()
    }
    by_gender_counts = gender_name.value_counts().to_dict()
    for gender, n in by_gender_counts.items():
        by_gender.setdefault(gender, {})["n_samples"] = int(n)

    # --- TPR (recall) per (profession, gender) ----------------------------------------
    tpr_mf = MetricFrame(
        metrics={"tpr": accuracy_score},
        y_true=y_true_name,
        y_pred=y_pred_name,
        sensitive_features=gender_name,
        control_features=y_true_name,
    )
    tpr_table = tpr_mf.by_group["tpr"].unstack()  # index=profession, columns=gender

    gender_dist = compute_gender_distribution(df, config, profession_names)
    top_imbalanced = gender_dist.head(config.fairness.top_k_imbalanced_professions)["profession"].tolist()

    tpr_by_profession = []
    for profession in tpr_table.index:
        row = tpr_table.loc[profession]
        male_tpr = row.get("male")
        female_tpr = row.get("female")
        gap = None
        if pd.notna(male_tpr) and pd.notna(female_tpr):
            gap = round(abs(male_tpr - female_tpr), 4)
        tpr_by_profession.append(
            {
                "profession": profession,
                "male_tpr": None if pd.isna(male_tpr) else round(male_tpr, 4),
                "female_tpr": None if pd.isna(female_tpr) else round(female_tpr, 4),
                "tpr_gap": gap,
                "is_highlighted_imbalanced": profession in top_imbalanced,
            }
        )
    tpr_by_profession.sort(key=lambda r: (r["tpr_gap"] is None, -(r["tpr_gap"] or 0)))

    gaps = [r["tpr_gap"] for r in tpr_by_profession if r["tpr_gap"] is not None]
    tpr_gap_summary = {
        "mean_abs_tpr_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "max_tpr_gap": max(gaps) if gaps else None,
        "max_tpr_gap_profession": next(
            (r["profession"] for r in tpr_by_profession if r["tpr_gap"] == max(gaps)), None
        )
        if gaps
        else None,
    }

    return {
        "by_gender": by_gender,
        "gender_distribution_by_profession": gender_dist[
            ["profession_id", "profession", "male", "female", "total", "pct_female", "imbalance"]
        ].to_dict(orient="records"),
        "top_imbalanced_professions": top_imbalanced,
        "tpr_by_profession_gender": tpr_by_profession,
        "tpr_gap_summary": tpr_gap_summary,
    }, gender_mf


def run_evaluation(
    config_path: str | None = None,
    source: str = "local",
    push: bool = True,
    hf_token: str | None = None,
) -> dict:
    config = load_config(config_path)
    test_df = _load_split(config, "test")
    label_names = _load_label_names(config)
    profession_names = label_names.get(config.data.target_column, [])
    if not profession_names:
        raise RuntimeError("Profession label names not found; re-run get-data.py / preprocess.py.")

    logger.info("Loading trained pipeline (source=%s)...", source)
    pipeline = _load_pipeline_local(config) if source == "local" else _load_pipeline_hub(config, hf_token)

    X_test, y_test = test_df[config.data.text_column], test_df[config.data.target_column]
    logger.info("Predicting on %d test rows...", len(test_df))
    y_pred = pipeline.predict(X_test)

    overall = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "n_samples": len(test_df),
    }
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(profession_names))),
        target_names=profession_names,
        output_dict=True,
        zero_division=0,
    )
    logger.info("Overall test metrics: %s", overall)

    fairness_report, gender_mf = compute_fairness_report(test_df, y_pred, config, profession_names)
    fairness_report = {"overall": overall, **fairness_report}

    processed_dir = config.root / config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.root / config.artifacts.metrics_path
    fairness_path = config.root / config.artifacts.fairness_report_path

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_class_report": report_dict}, f, indent=2, ensure_ascii=False)
    with open(fairness_path, "w", encoding="utf-8") as f:
        json.dump(fairness_report, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s and %s", metrics_path, fairness_path)

    logger.info(
        "Top imbalanced professions: %s | mean |TPR gap|=%s | max gap: %s (%s)",
        fairness_report["top_imbalanced_professions"],
        fairness_report["tpr_gap_summary"]["mean_abs_tpr_gap"],
        fairness_report["tpr_gap_summary"]["max_tpr_gap"],
        fairness_report["tpr_gap_summary"]["max_tpr_gap_profession"],
    )

    if push:
        _push_evaluation_artifacts(config, pipeline, metrics_path, fairness_path, overall, fairness_report, gender_mf, hf_token)

    return fairness_report


def _push_evaluation_artifacts(config, pipeline, metrics_path, fairness_path, overall, fairness_report, gender_mf, hf_token):
    from skops import hub_utils

    from src.utils import build_base_model_card

    local_repo_dir = config.root / config.artifacts.local_model_dir
    if not local_repo_dir.exists():
        logger.warning("%s not found; run train.py first. Skipping Hub push of eval artifacts.", local_repo_dir)
        return

    hub_utils.add_files(metrics_path, fairness_path, dst=local_repo_dir, exist_ok=True)

    summary_path = local_repo_dir / "training_summary.json"
    training_summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            training_summary = json.load(f)

    readme_path = local_repo_dir / "README.md"
    model_card = build_base_model_card(
        config,
        pipeline,
        training_summary.get("best_params", {}),
        training_summary.get("dev_metrics", {}),
    )

    model_card.add_metrics(
        section="Model description/Evaluation Results",
        description="Metrics computed on the held-out `test` split of LabHC/bias_in_bios.",
        **overall,
    )

    top_professions = ", ".join(fairness_report["top_imbalanced_professions"])
    worst_gap = fairness_report["tpr_gap_summary"]
    bias_section = (
        "This model was audited for gender bias with "
        "[fairlearn](https://fairlearn.org/), using `gender` as the sensitive feature and the "
        "**True Positive Rate (TPR) gap** — the difference in recall between men and women for "
        "a given predicted profession — as the primary fairness metric, following the "
        "methodology of De-Arteaga et al., *Bias in Bios* (FAccT 2019, arXiv:1901.09451).\n\n"
        f"- Mean absolute TPR gap across all professions: **{worst_gap['mean_abs_tpr_gap']}**\n"
        f"- Largest TPR gap: **{worst_gap['max_tpr_gap']}**, for profession "
        f"**{worst_gap['max_tpr_gap_profession']}**\n"
        f"- Professions with the most gender-imbalanced training data: {top_professions}\n\n"
        "See `fairness_report.json` in this repo for the full per-profession breakdown, and the "
        "project README's Bias Dashboard for a visual comparison.\n\n"
        "**Disclaimer: this is a pedagogical / research project. Do not use this model, or its "
        "predictions, to make real hiring, screening, or employment decisions.**"
    )
    model_card.add(**{"Bias, Risks, and Limitations": bias_section})
    model_card.add_fairlearn_metric_frame(
        gender_mf,
        table_name="Accuracy / recall by gender",
        description="fairlearn MetricFrame computed on the test split with `gender` as the sensitive feature.",
    )
    model_card.save(readme_path)

    hub_utils.push(
        repo_id=config.huggingface.model_repo,
        source=local_repo_dir,
        token=hf_token,
        commit_message="Add evaluation metrics and fairness audit report",
        create_remote=True,
        private=config.huggingface.private,
    )
    logger.info("Pushed evaluation + fairness artifacts to https://huggingface.co/%s", config.huggingface.model_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--source", choices=["local", "hub"], default="local", help="Where to load the pipeline from")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing artifacts to the HF Hub")
    parser.add_argument("--hf-token", default=None, help="HF Hub token (else HF_TOKEN env var or local login)")
    args = parser.parse_args()
    run_evaluation(args.config, source=args.source, push=not args.no_push, hf_token=args.hf_token)
