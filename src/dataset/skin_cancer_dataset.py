from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image


class SkinCancerDataset(Dataset):

    """
    Dataset containing information about skin cancer images.

    Inherits from torch.utils.data.Dataset.
        See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        for more information.
    """

    def __init__(
        self,
        labels_csv: Path,
        img_dir: Path,
        transform: Optional[Callable] = None,
        domain_csv: Optional[Path] = None,
        domain_col: Optional[str] = None,
    ) -> None:

        """
        Parameters:

            labels_csv: Path

                Path to file containing classification labels for all images.

                This file should be a .csv with images as the first column and
                their respective one-hot-encoded class as the rest.

            img_dir: Path

                Directory containing all images.

            transform: Optional[Callable]

                Transformations to be applied to images before returning them.

            domain_csv: Optional[Path]

                Path to file containing the respective domain of each image.

                This file should be sorted in the same order as labels_csv.

            domain_col: Optional[str]

                Column name where domain information is stored in domain_csv.
        """

        self.img_labels = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform

        self.domain_csv = pd.read_csv(domain_csv) if domain_csv else None
        self.domain_col = domain_col

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:

        img_filename, *one_hot_encoding = self.img_labels.iloc[idx]

        img_path = self.img_dir / f"{img_filename}.jpg"
        image = read_image(str(img_path)).float()
        label = Tensor(one_hot_encoding).argmax()

        if self.transform:
            image: Tensor = self.transform(image)

        if self.domain_csv is None:  # if there is no information about domains
            return image, label

        domain = self.domain_csv.iloc[idx][self.domain_col]

        return image, label, domain, idx
