from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from torch import Tensor

from robustdg_modified.config.args_mock import ArgsMock

from .train_dataset import TrainDataset


class HybridTrainDataset(TrainDataset):

    """
    Dataset containing information about skin cancer images.

    This applies data augmentations so that it can be used with
    robustdg_modified/algorithms/hybrid.py algorithm.
    """

    def __init__(
        self,
        augmentation_fn: Callable,
        args: ArgsMock,
        domain_names: list[str],
        base_domain_size: int,
        training_list_size: list[int],
        img_dir: Path,
        int_to_img_names: pd.Series,
        domain_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        transform: Optional[Callable],
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
        Parameters:

            augmentation_fn: Callable

                Data augmentations to be applied to the data.

            All other parameters are required to initialize base_class:
                robustdg_modified.dataset.train_dataset.TrainDataset
        """

        super().__init__(
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

        self.augmentation_fn = augmentation_fn

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int, Tensor]:

        img, label, domain, index, object = super().__getitem__(idx)

        augmented = self.augmentation_fn(img)

        return img, augmented, label, domain, index, object
