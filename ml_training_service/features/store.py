"""
Feature store: versioned read/write of processed feature Parquet (local or GCS).
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_features(
    df: pd.DataFrame,
    output_path: str,
    version: str = "v1",
    split: str = "train",
) -> str:
    """Write feature Parquet. Returns the path written."""
    if output_path.startswith("gs://"):
        path = f"{output_path}/{version}/{split}.parquet"
        df.to_parquet(path, index=False)
    else:
        out_dir = Path(output_path) / version
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{split}.parquet")
        df.to_parquet(path, index=False)

    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_features(
    store_path: str,
    version: str = "v1",
    split: str = "train",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read feature Parquet for a given version and split."""
    if store_path.startswith("gs://"):
        path = f"{store_path}/{version}/{split}.parquet"
    else:
        path = str(Path(store_path) / version / f"{split}.parquet")

    df = pd.read_parquet(path, columns=columns)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
