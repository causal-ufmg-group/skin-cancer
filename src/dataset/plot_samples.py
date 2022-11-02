from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


def plot_some_samples(
    nrows: int, ncols: int, dataset: Dataset, classes: Sequence[str]
) -> None:

    """
    Plot some dataset samples.

    ----
    Parameters:

        nrows, ncols: int, int

            Number of rows/cols for the plot.

            nrows*ncols samples will be shown.

        dataset: torch.utils.data.Dataset

            Dataset where samples will be taken from.

        classes: Sequence[str]

            Sequence containing classes names.

            It should follow the same order as your one-hot-encoding,
            that is, index zero should contain the name for encoding
            [1, 0, ..., 0].
    """

    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 10))

    flat_axs = [ax for row in axs for ax in row]

    for i, ax in enumerate(flat_axs):

        img, label_pos = dataset[i]

        label = classes[label_pos]

        ax.set_title(f"Sample #{i} - {label}")
        ax.imshow(to_pil_image(img.type(torch.uint8)))
        ax.axis("off")

    fig.tight_layout()
