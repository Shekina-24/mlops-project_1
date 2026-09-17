"""Clean the raw Bias in Bios splits and write stratified, sub-sampled Parquet
files to data/processed/.

Usage:
    python src/preprocess.py [--config configs/config.yaml]

The heavy-lifting functions (`clean_dataframe`, `stratified_subsample`,
`preprocess_split`) are pure (no I/O) so they can be unit-tested directly on
small synthetic DataFrames.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.utils import clean_text, get_logger, set_seed

logger = get_logger(__name__)


def clean_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply text cleaning and drop rows that are too short or missing labels."""
    text_col = config.data.text_column
    target_col = config.data.target_column
    sensitive_col = config.data.sensitive_column

    out = df.copy()
    out[text_col] = out[text_col].apply(
        lambda t: clean_text(
            t,
            lowercase=config.preprocessing.lowercase,
            strip_whitespace=config.preprocessing.strip_whitespace,
            collapse_whitespace=config.preprocessing.collapse_whitespace,
        )
    )
    out = out.dropna(subset=[text_col, target_col, sensitive_col])
    out = out[out[text_col].str.len() >= config.preprocessing.min_text_length]
    out = out.reset_index(drop=True)
    return out


def stratified_subsample(
    df: pd.DataFrame,
    stratify_by: list[str],
    frac: float,
    min_samples_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Sample `frac` of `df`, stratified jointly on `stratify_by` columns.

    Groups with `min_samples_per_group` rows or fewer are kept whole rather
    than shrunk further, so rare (profession, gender) combinations don't
    disappear from the training set entirely.
    """
    if frac >= 1.0:
        return df.reset_index(drop=True)

    sampled_groups = []
    for _, group in df.groupby(stratify_by, observed=True):
        if len(group) <= min_samples_per_group:
            sampled_groups.append(group)
        else:
            n = min(len(group), max(1, round(len(group) * frac)))
            sampled_groups.append(group.sample(n=n, random_state=seed))

    return pd.concat(sampled_groups, ignore_index=True)


def preprocess_split(
    df: pd.DataFrame, config: Config, target_rows: int | None
) -> pd.DataFrame:
    """Clean a split and, if configured, stratified-subsample it to `target_rows`."""
    cleaned = clean_dataframe(df, config)

    if not config.sampling.enabled or target_rows is None or target_rows >= len(cleaned):
        return cleaned

    frac = target_rows / len(cleaned)
    return stratified_subsample(
        cleaned,
        stratify_by=config.sampling.stratify_by,
        frac=frac,
        min_samples_per_group=config.sampling.min_samples_per_group,
        seed=config.seed,
    )


def run_preprocessing(config_path: str | None = None) -> None:
    config = load_config(config_path)
    set_seed(config.seed)

    raw_dir = config.root / config.data.raw_dir
    processed_dir = config.root / config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"No raw Parquet files found in {raw_dir}. Run `python src/get-data.py` first."
        )

    raw_splits = {p.stem: pd.read_parquet(p) for p in raw_files}
    total_raw_rows = sum(len(df) for df in raw_splits.values())

    total_samples = config.sampling.total_samples
    for split_name, df in raw_splits.items():
        if config.sampling.enabled and total_samples is not None:
            target_rows = round(total_samples * len(df) / total_raw_rows)
        else:
            target_rows = None

        processed = preprocess_split(df, config, target_rows)
        out_path = processed_dir / f"{split_name}.parquet"
        processed.to_parquet(str(out_path))
        logger.info(
            "Split '%s': %d raw -> %d processed rows -> %s",
            split_name,
            len(df),
            len(processed),
            out_path,
        )

    label_names_src = raw_dir / "label_names.json"
    if label_names_src.exists():
        shutil.copy(label_names_src, processed_dir / "label_names.json")
    else:
        logger.warning("label_names.json not found in %s; skipping copy", raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    run_preprocessing(args.config)
