from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from robustdg_modified.config.args_mock import ArgsMock


class TestDataset(Dataset):

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
        img_dir: Path,
        int_to_img_names: pd.Series,
        labels_df: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:

        r"""
        Even though for test datasets it does not provide any advantage,
        you could use the function
            .create_train_dataset.create_train_dataset()
        to create an instance of this.

        Initializes TrainingSkinCancerDataset class.
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

            domain_names: list[str]

                Domain names.

                Unnecessary since domain information is not provided during test.
        """

        # Initializing variables required for robustdg algorithms
        self.args = args

        # Initializing general purpose variables
        self.img_dir = img_dir
        self.int_to_img_names = int_to_img_names
        self.img_one_hot_labels = labels_df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.int_to_img_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, None, int, Tensor]:

        img_filename = self.int_to_img_names.loc[idx]
        img_path = self.img_dir / f"{img_filename}.jpg"
        image = read_image(str(img_path)).float()

        img_label = self.img_one_hot_labels.loc[idx].to_numpy()

        if self.transform:
            image: Tensor = self.transform(image)

        # TODO: Need to verify whether or not idx must be in
        #       [0, number_in_domain] for each domain
        return image, img_label, None, idx, img_label
