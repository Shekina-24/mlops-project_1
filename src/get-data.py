"""Download the Bias in Bios dataset from the Hugging Face Hub and save it raw.

Usage:
    python src/get-data.py [--config configs/config.yaml]

Writes one Parquet file per split to `data.raw_dir` (e.g. data/raw/train.parquet)
plus a `label_names.json` mapping the integer `profession` codes to their
human-readable names, read directly off the dataset's ClassLabel feature so we
never have to hardcode (and risk mis-copying) the 28 profession names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)

# LabHC/bias_in_bios stores `profession` and `gender` as plain int64 columns
# (no HF ClassLabel names attached), so we hardcode the mapping from the
# dataset card (https://huggingface.co/datasets/LabHC/bias_in_bios), which
# reproduces the 28-profession label set from De-Arteaga et al. 2019.
PROFESSION_NAMES = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]
GENDER_NAMES = ["male", "female"]


def download_raw_data(config_path: str | None = None) -> None:
    from datasets import load_dataset

    config = load_config(config_path)
    raw_dir = config.root / config.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' from the Hugging Face Hub...", config.data.dataset_repo)
    dataset_dict = load_dataset(config.data.dataset_repo)
    logger.info("Splits found: %s", list(dataset_dict.keys()))

    fallback_names = {
        config.data.target_column: PROFESSION_NAMES,
        config.data.sensitive_column: GENDER_NAMES,
    }
    label_names: dict[str, list[str]] = {}
    for column in (config.data.target_column, config.data.sensitive_column):
        feature = dataset_dict[list(dataset_dict.keys())[0]].features.get(column)
        if feature is not None and hasattr(feature, "names"):
            label_names[column] = list(feature.names)
        elif column in fallback_names:
            label_names[column] = fallback_names[column]
            logger.info("Column '%s' has no embedded ClassLabel names; using hardcoded mapping.", column)

    for split_name, split_dataset in dataset_dict.items():
        out_path = raw_dir / f"{split_name}.parquet"
        split_dataset.to_parquet(str(out_path))
        logger.info("Saved split '%s' (%d rows) -> %s", split_name, len(split_dataset), out_path)

    label_names_path = raw_dir / "label_names.json"
    with open(label_names_path, "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2, ensure_ascii=False)
    logger.info("Saved label names -> %s", label_names_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    download_raw_data(args.config)
