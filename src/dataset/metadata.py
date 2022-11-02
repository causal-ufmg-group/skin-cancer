from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from torchvision.io import read_image


def get_classes(labels_csv_path: Path) -> np.ndarray:

    """
    Get classification labels.

    ----
    Parameters:

        labels_csv_path: Path

            Path to .csv file with one-hot-encoding information.

    ----
    Returns:

        np.ndarray -> array of strings

            Arrays with label names.
    """

    train_classes = pd.read_csv(labels_csv_path, skiprows=lambda row: row != 0)

    # ignore first column since it only stores image name
    desired_columns = train_classes.columns[1:]

    return np.array(desired_columns.to_list())


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
