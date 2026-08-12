import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
