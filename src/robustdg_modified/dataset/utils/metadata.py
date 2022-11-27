from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from torchvision.io import read_image


def get_one_hot_encoded_names(one_hot_encoded_df: pd.DataFrame) -> np.ndarray:

    """
    Get classification labels.

    ----
    Parameters:

        one_hot_encoded_df: Path

            DataFrame with one-hot encoding information.

            Only columns should be the one-hot encoded ones.
    ----
    Returns:

        np.ndarray -> array of strings

            Arrays with label names.
    """
    return np.array(one_hot_encoded_df.columns.to_list())


def get_image_dimensions(img_dir: Path) -> tuple[int, int, int]:

    """
    Returns image dimension information: (number of channels, height, width).

    -----
    Parameters:

        img_dir: Path

            Path to image directory.

    ----
    Returns:

        tuple[int, int, int]

            Respectively, number of channels, height and width.
    """

    images: Iterator = img_dir.glob("*.jpg")
    first_img = next(images)

    return read_image(str(first_img)).size()
