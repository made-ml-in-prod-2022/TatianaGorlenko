from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, target: pd.Series, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :rtype: object
    """
    train_data, val_data, train_target, val_target = train_test_split(
        data, target, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data, train_target, val_target
