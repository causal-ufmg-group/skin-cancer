import pandas as pd
from numpy.random import choice


def get_split_train_validation_index(
    labels_index: pd.Index, frac_train: float
) -> tuple[pd.Index, pd.Index]:

    """
    Returns index for train and validation after split.

    -----
    Parameters:

        labels_index: pd.Index

            Index for all labels: train and validation

        frac_train: float

            Percentage of total labels that should be used for train.
            (1 - frac_train) will be used for validation.

    -----
    Returns:

        tuple[pd.Index, pd.Index]

            Respectively, train and validation indexes.
    """

    total_labels = len(labels_index)
    num_labels = int(total_labels * frac_train)

    train_index = pd.Index(choice(labels_index, size=num_labels, replace=False))
    validation_index = labels_index.difference(train_index)

    return train_index.sort_values(), validation_index.sort_values()


def get_only_desired_indexes(
    index: pd.Index, *data_frames: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:

    """
    Filter only desired indexes for all pd.DataFrames passed as arguments.

    Index in resulting pd.DataFrame will be reset, that is, it will be in
    the interval
        [0, len(index)).

    -----
    Parameters:

        index: pd.Index | list

            Index to be used for all pd.DataFrames

        *args: pd.DataFrame

            All pd.DataFrame which should be indexed.

    -----
    Returns:

        tuple[pd.DataFrame]:

            Filtered datasets.
    """

    return tuple(df.loc[index].reset_index(drop=True) for df in data_frames)
