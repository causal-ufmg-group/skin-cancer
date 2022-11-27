"""
Module responsible for dealing with dataset information.

Datasets following this module should be compatible with some RobustDG algorithms.

Necessary information about RobustDG requirements can be found
in TrainDataset and TestDataset classes.
"""

from . import read_dataset as read
from . import utils
from .create_test_dataset import create_robustdg_test_dataset
from .create_train_dataset import create_robustdg_train_dataset
from .test_dataset import TestDataset
from .train_dataset import TrainDataset

__all__ = [
    "read",
    "utils",
    "create_robustdg_test_dataset",
    "create_robustdg_train_dataset",
    "TestDataset",
    "TrainDataset",
]
