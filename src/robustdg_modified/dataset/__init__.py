"""
Module responsible for dealing with dataset information.

Datasets following this module should be compatible with some RobustDG algorithms.

Necessary information about RobustDG requirements can be found
in TrainDataset and TestDataset classes.
"""

from . import read_dataset as read
from . import utils
from .create_hybrid_train_dataset import (
    create_robustdg_hybrid_dataset_from_train_dataset,
)
from .create_test_dataset import create_robustdg_test_dataset
from .create_train_dataset import create_robustdg_train_dataset
from .hybrid_train_dataset import HybridTrainDataset
from .test_dataset import TestDataset
from .train_dataset import TrainDataset
from .train_val_split import get_only_desired_indexes, get_split_train_validation_index

__all__ = [
    "read",
    "utils",
    "create_robustdg_hybrid_dataset_from_train_dataset",
    "create_robustdg_test_dataset",
    "create_robustdg_train_dataset",
    "HybridTrainDataset",
    "TestDataset",
    "TrainDataset",
    "get_only_desired_indexes",
    "get_split_train_validation_index",
]
