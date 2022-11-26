from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..args_mock import ArgsMock
from .test_dataset import TestDataset


def create_test_dataset(
    args: ArgsMock,
    img_dir: Path,
    int_to_img_names: pd.Series,
    labels_df: pd.DataFrame,
    transform: Optional[Callable] = None,
) -> None:

    r"""
    Creates an instance of TestDataset class.

    This class contains all necessary information required by RobustDG algorithms.

    ------
    RobustDG Parameters:

        args: ArgsMock | argparse.Argument

            Configuration for robustdg.

            See ArgsMock documentation for full list of parameters.

    ------
    General Purpose Parameters:

        img_dir: Path

            Directory containing all images.

        int_to_img_names: pd.Series

            Mapping of integer to image name.

        labels_df: pd.DataFrame

            DataFrame containing classification label of each image.

            Should be sorted in the same order as int_to_img_names.

        transform: Optional[Callable]

            Transformations to be applied to images before returning them.
    """

    return TestDataset(args, img_dir, int_to_img_names, labels_df, transform)
