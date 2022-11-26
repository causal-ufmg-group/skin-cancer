from typing import NewType

import numpy.typing as npt
from torch import Tensor

ArrayTensor = NewType("ArrayTensor", npt.NDArray, Tensor)


def convert_one_hot_to_integers(one_hot_encoded: ArrayTensor) -> ArrayTensor:

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
