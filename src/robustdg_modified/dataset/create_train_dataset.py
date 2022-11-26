from pathlib import Path
from typing import Callable, Optional

import numpy.typing as npt
import pandas as pd

from ..args_mock import ArgsMock
from ..utils import convert_one_hot_to_integers
from .train_dataset import TrainDataset


def _get_num_imgs_for_each_domain(
    one_hot_domains: npt.NDArray, domains: list[str]
) -> int:

    integer_domains = convert_one_hot_to_integers(one_hot_domains)

    return [(integer_domains == domain).sum() for domain in domains]


def _get_base_domain_size(self) -> int:

    r"""
    Base domain size seems to be the sum of max number of elements in a domain
    over all classes, that is,

        \sum_{class} \max_{domain} num_labels(class, domain)

    Not really sure what it is used for.
    """

    # This is basically the same code you could find in robustdg/data/mnist_loader.py

    base_domain_size = 0

    integer_domain = convert_one_hot_to_integers(self.img_one_hot_domain.to_numpy())
    integer_labels = convert_one_hot_to_integers(self.img_one_hot_labels.to_numpy())

    for label_number in range(self.args.out_classes):

        base_class_size = 0

        for domain_number, _ in enumerate(self.domain_names):

            domain_idx: npt.NDArray = integer_domain == domain_number
            labels_this_domain: npt.NDArray = integer_labels[domain_idx]

            class_idx: npt.NDArray = labels_this_domain == label_number
            curr_class_size: int = labels_this_domain[class_idx].shape[0]

            if base_class_size < curr_class_size:
                base_class_size = curr_class_size

        base_domain_size += base_class_size

    return base_domain_size


def create_train_dataset(
    args: ArgsMock,
    img_dir: Path,
    int_to_img_names: pd.Series,
    domain_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    transform: Optional[Callable] = None,
) -> None:

    """
    Creates an instance of TrainDataset class.

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

        domain_df: pd.DataFrame

            DataFrame containing the respective one-hot encoded
            domain of each image.

            Should be sorted in the same order as int_to_img_names.

        transform: Optional[Callable]

            Transformations to be applied to images before returning them.
    """

    # Calculating some required RobustDG parameters
    domain_names = domain_df.columns.to_list()
    base_domain_size = _get_base_domain_size()
    training_list_size = _get_num_imgs_for_each_domain(
        domain_df.to_numpy(), domain_names
    )

    return TrainDataset(
        args,
        domain_names,
        base_domain_size,
        training_list_size,
        img_dir,
        int_to_img_names,
        domain_df,
        labels_df,
        transform,
    )
