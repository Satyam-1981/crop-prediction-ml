from pathlib import Path
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

from .config import KAGGLE_DATASET, KAGGLE_FILE, FEATURES, TARGET


def load_dataset() -> pd.DataFrame:
    """
    Load the public Kaggle Crop Recommendation Dataset through KaggleHub.
    No CSV needs to be manually downloaded into this project.
    """
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        KAGGLE_FILE,
    )

    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates()
    return df
