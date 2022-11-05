from pathlib import Path
from typing import Callable, Literal, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .skin_cancer_dataset import SkinCancerDataset

StageTypes = Literal["train", "test"]


class SkinCancerDataModule(pl.LightningDataModule):

    """
    Pytorch-Lightning data module for Skin Cancer dataset.

    See https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
    for more information.
    """

    def __init__(
        self,
        ground_truth_map: dict[StageTypes, Path],
        img_dir_map: dict[StageTypes, Path],
        batch_size: int,
        dataloader_num_workers: int,
        transform: Optional[Callable] = None,
        domain_csv: Optional[Path] = None,
        domain_col: Optional[str] = None,
    ) -> None:

        """
        Initialize SkinCancerDataModule.

        ----
        Parameters:

            ground_truth_map: dict[Literal["train", "test"], Path]

                Dictionary mapping each possible stage (train or test)
                to its ground-truth .csv file.

            img_dir_map: dict[Literal["train", "test"], Path]

                Dictionary mapping each possible stage to the respective
                directory containing its images.

            batch_size: int

                How many samples per batch to load.

            transform: Optional[Callable] = None

                Transformations to be applied after loading image tensors.

                If None, no transformation will be applied.

            domain_csv: Optional[Path]

                Path to file containing the respective domain of each image.

                This file should be a .csv with image filepath as the first
                column and its respective classification label as the second.

            domain_col: Optional[str]

                Column name where domain information is stored in domain_csv.
        """
        super().__init__()

        self.ground_truth_map = ground_truth_map
        self.img_dir_map = img_dir_map
        self.batch_size = batch_size
        self.num_workers = dataloader_num_workers
        self.transform = transform

        self.domain_csv = domain_csv
        self.domain_col = domain_col

    def prepare_data(self) -> None:
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:

        if stage in ("fit", None):

            full_train = SkinCancerDataset(
                self.ground_truth_map["train"],
                self.img_dir_map["train"],
                self.transform,
                self.domain_csv,
                self.domain_col,
            )

            num_img = len(full_train)
            train_length = int(0.85 * num_img)
            dataset_lengths = [train_length, num_img - train_length]

            self.train, self.validation = random_split(full_train, dataset_lengths)

        if stage in ("test", None):

            self.test = SkinCancerDataset(
                self.ground_truth_map["test"],
                self.img_dir_map["test"],
                self.transform,
                # no domain information necessary for testing
                domain_csv=None,
                domain_col=None,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
