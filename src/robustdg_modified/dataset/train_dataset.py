from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from robustdg_modified.config.args_mock import ArgsMock

from .utils.domain_index import get_domain_index


class TrainDataset(Dataset):

    """
    Dataset containing information about skin cancer images.

    Inherits from torch.utils.data.Dataset.
        See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        for more information.

    The main difference between train and test is that test does not provide
    any domain related information.
    """

    def __init__(
        self,
        args: ArgsMock,
        domain_names: list[str],
        base_domain_size: int,
        training_list_size: list[int],
        img_dir: Path,
        int_to_img_names: pd.Series,
        domain_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:

        r"""
        Since it requires less parameters, you should use the function
            .create_test_dataset.create_test_dataset()
        to create an instance of this.

        Initializes TrainDataset class.
        It inherits from torch.utils.data.Dataset.

        Parameters below are divided into three categories:
            - Required RobustDG parameters:
                - Parameters required when using RobustDG algorithms.
            - General purpose parameters
                - Usually related to torch.utils.data.Dataset base class.
            - Removed RobustDG parameters
                - RobustDG parameters not required anymore because of how
                this is implemented.
                - Documented mainly for future reference.

        ------
        RobustDG Parameters:

            args: ArgsMock | argparse.Argument

                Configuration for robustdg.

                See ArgsMock documentation for full list of parameters.

            base_domain_size: int

                Sum of the maximum number of elements in a domain
                over all classes, that is,

                    \sum_{class} \max_{domain} num_labels(class, domain)

            domain_names: list[str]

                Domain names.

                Unnecessary since we can extract it from one-hot encoding DataFrame.

                        training_list_size: list[int]

                Number of images for each domain.

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

        ------
        Removed RobustDG Parameters:

            root: Path

                Parameter used by RobustDG in order to load dataset.

                In this implementation, img_dir does basically that.

            data_case: Literal["train", "test"]

                Seems to be mainly used for loading datasets.

                Not required because training and testing are different classes.

                See robustdg/utils/helper.py for more details.

            match_func: bool

                Mainly used for distinguishing between train and test.

                Not required because training and testing are different classes.

                See robustdg/utils/helper.py and robustdg/data/mnist_loader.py
                for more details.
        """

        # Initializing variables required for robustdg algorithms
        self.args = args
        self.list_domains = domain_names
        self.training_list_size = training_list_size
        self.base_domain_size = base_domain_size

        # Initializing general purpose variables
        self.img_dir = img_dir
        self.int_to_img_names = int_to_img_names
        self.img_one_hot_labels = labels_df
        self.img_one_hot_domain = domain_df
        self.transform = transform

        # Create domain specific index
        self.domain_index = get_domain_index(domain_df)

    def __len__(self) -> int:
        return len(self.int_to_img_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:

        img_filename = self.int_to_img_names.loc[idx]
        img_path = self.img_dir / f"{img_filename}.jpg"
        image = read_image(str(img_path)).float()

        img_label = self.img_one_hot_labels.loc[idx].to_numpy()
        img_domain = self.img_one_hot_domain.loc[idx].to_numpy()
        domain_index = self.domain_index.loc[idx]
        # TODO: Object is the same as the label because we are trying
        #       to identify is its melanoma type, i.e., its class
        img_object = np.argmax(img_label)

        if self.transform:
            image: Tensor = self.transform(image)

        return image, img_label, img_domain, domain_index, img_object
