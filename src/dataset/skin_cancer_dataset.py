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
        self, labels_csv: Path, img_dir: Path, transform: Optional[Callable] = None
    ) -> None:

        """
        Parameters:

            labels_csv: Path

                File containing classification labels for all images.

                This file should be a .csv with image filepath as the first
                column and its respective classification label as the second.

            img_dir: Path

                Directory containing all images.

            transform: Optional[Callable]

                Transformations to be applied to images before returning them.
        """

        self.img_labels = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:

        img_filename, *one_hot_encoding = self.img_labels.iloc[idx]

        img_path = self.img_dir / f"{img_filename}.jpg"
        image = read_image(str(img_path)).float()

        if self.transform:
            image: Tensor = self.transform(image)

        return image, Tensor(one_hot_encoding).argmax()
