import numpy as np


def map_values_proportionally_to_interval(
    values: np.ndarray, interval: tuple[float, float]
) -> np.ndarray:

    """
    Map values to desired interval.

    ----
    Parameters:

        values: np.ndarray

            Values which should be mapped to interval.

            It should support element wise operations like numpy.

        interval: tuple[float, float]

            Interval to map values to.

            Both endpoints are included.

    ----
    Returns:

        np.ndarray

            Values mapped to interval.

    """

    minimum, maximum = interval
    diff = maximum - minimum

    values_starting_from_zero: np.ndarray = values - np.min(values)
    normalized_values: np.ndarray = values_starting_from_zero / np.max(values)

    mapped_values: np.ndarray = minimum + diff * normalized_values
    return mapped_values.astype(np.int32)
