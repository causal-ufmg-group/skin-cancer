import numpy.typing as npt
import pandas as pd
from torch import Tensor


def convert_one_hot_to_integers(
    one_hot_encoded: npt.NDArray | Tensor,
) -> npt.NDArray | Tensor:

    """
    Converts one hot encoding to tensor/array of integers.

    i-th integer will be associated with one_hot_encode_df.column i-th value.

    ----
    Parameters:

        one_hot_encoded: np.typing.NDArray | Tensor

            Each row should be an one-hot encoded value.

    -----
    Returns:

        npt.NDArray | Tensor

            Tensor/array of integers equivalent to one-hot encoding input.
    """

    return one_hot_encoded.argmax(axis=1)


def convert_one_hot_df_to_names(
    one_hot_df: pd.DataFrame, series_name: str
) -> pd.Series:

    """
    Converts one hot encoding pd.DataFrame to a pd.Series
    of strings.

    ----
    Parameters:

        one_hot_df: pd.DataFrame

            DataFrame with one-hot encoding as its only columns.

        series_names: str

            What to call the resulting series.
    -----
    Returns:

        pd.Series

            Series mapping each row to its class.
    """

    return one_hot_df.idxmax(axis=1).rename(series_name)
