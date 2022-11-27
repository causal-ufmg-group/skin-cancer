from typing import Callable, Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image


def _plot_some_samples(
    nrows: int,
    ncols: int,
    dataset: Dataset,
    classes: Sequence[str],
    domains: Sequence[str],
    adapter_func: Callable,
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

        # adapter_func is a function required as parameters in order to
        # make both train and test datasets available for this template
        img, label_pos, domain, idx, object_ = adapter_func(dataset[i])

        # label_pos and domain are one-hot encoded
        label = classes[label_pos.argmax()]
        domain = domains[domain.argmax()] if domains is not None else ""

        ax.set_title(f"Sample #{i} - {label}. Domain: {domain}")
        ax.imshow(to_pil_image(img.type(torch.uint8)))
        ax.axis("off")

    fig.tight_layout()


def plot_some_train_samples(
    nrows: int,
    ncols: int,
    dataset: Dataset,
    classes: Sequence[str],
    domains: Sequence[str],
) -> None:

    """
    Plot some dataset samples.

    ----
    Parameters:

        nrows, ncols: int, int

            Number of rows/cols for the plot.

            nrows*ncols samples will be shown.

        dataset: torch.utils.data.Dataset

            Train dataset where samples will be taken from.

        classes: Sequence[str]

            Sequence containing classes names.

            It should follow the same order as your one-hot-encoding,
            that is, index zero should contain the name for encoding
            [1, 0, ..., 0].
    """
    _plot_some_samples(nrows, ncols, dataset, classes, domains, lambda x: x)


def plot_some_test_samples(
    nrows: int,
    ncols: int,
    dataset: Dataset,
    classes: Sequence[str],
) -> None:

    """
    Plot some dataset samples.

    ----
    Parameters:

        nrows, ncols: int, int

            Number of rows/cols for the plot.

            nrows*ncols samples will be shown.

        dataset: torch.utils.data.Dataset

            Test dataset where samples will be taken from.

        classes: Sequence[str]

            Sequence containing classes names.

            It should follow the same order as your one-hot-encoding,
            that is, index zero should contain the name for encoding
            [1, 0, ..., 0].
    """

    def _adapter_func(dataset_input):

        img, label_pos, idx, *_ = dataset_input

        return img, label_pos, None, idx, label_pos

    _plot_some_samples(nrows, ncols, dataset, classes, None, _adapter_func)
