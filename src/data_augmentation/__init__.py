"""
Module for applying data augmentation for unbalanced datasets.
"""

from .augmentation import augment_all_imgs, copy_all_imgs
from .domain_information import get_information_per_domain_label
from .values_to_interval import map_values_proportionally_to_interval

__all__ = [
    "augment_all_imgs",
    "copy_all_imgs",
    "get_information_per_domain_label",
    "map_values_proportionally_to_interval",
]
