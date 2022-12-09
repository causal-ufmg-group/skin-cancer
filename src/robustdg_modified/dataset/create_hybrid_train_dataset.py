from typing import Callable

from .hybrid_train_dataset import HybridTrainDataset
from .train_dataset import TrainDataset


def create_robustdg_hybrid_dataset_from_train_dataset(
    train_dataset: TrainDataset,
    augmentation_fn: Callable,
) -> HybridTrainDataset:

    """
    Creates an instance of HybridTrainDataset class.

    This class contains all necessary information required by Hybrid robustDG algorithm.

    ------
    Parameters:

        train_dataset: robustdg_modified.dataset.train_dataset.TrainDataset

            Train dataset to base Hybrid class off of.

        augmentation_fn: Callable



    """

    return HybridTrainDataset(
        augmentation_fn,
        train_dataset.args,
        train_dataset.list_domains,
        train_dataset.base_domain_size,
        train_dataset.training_list_size,
        train_dataset.img_dir,
        train_dataset.int_to_img_names,
        train_dataset.img_one_hot_domain,
        train_dataset.img_one_hot_labels,
        train_dataset.transform,
    )
