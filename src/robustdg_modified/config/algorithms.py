"""
Configuration required for different types of algorithms.

robustdg_modified.algorithms.no_domain.NoDomain
    Uses no domain information while training
        NO_DOMAIN_CONFIG: basically only change method_name

robustdg_modified.algorithms.erm_match.ErmMatch
    Can be run using ErmMatch class:
        There are two different types:
            BASELINE_ERM_CONFIG: Just a simple test
            PERFECT_MATCH_CONFIG: Should give best results

robustdg_modified.algorithms.match_dg.MatchDG
    Obtained by running contrastive loss and erm phase
        MATCH_DG_CTR_CONFIG: configurations for ctr phase
        MATCH_DG_ERM_CONFIG: configurations for erm phase
"""

from typing import Any

from .args_mock import ArgsMock


def set_configuration_parameters(args: ArgsMock, key_values: dict[str, Any]) -> None:

    """
    Set configuration parameters into ArgsMock instance.

    -----
    Parameters:

        args: ArgsMock

            ArgsMock instance to which parameters should be set.

        key_values: dict[str, Any]

            Each dataclass variable (key) will be set to its respect value.
    """

    for key, value in key_values.items():
        setattr(args, key, value)


# OUT CONFIGURATIONS
NO_DOMAIN_CONFIG = {"method_name": "no_domain"}

# ROBUSTDG CONFIGURATIONS:
#   robustdg/notebooks/reproduce_results.ipynb
#   robustdg/reproduce_scripts/mnist_run.py
#   robustdg/reproduce_scripts/pacs_run.py
# Most of them were adapted for perfect match.

# ERM

# RandMatch
#   python train.py <...>
#       --img_c 3 --method_name erm_match --penalty_ws 10.0 --match_case 0 --epochs 25

# PerfMatch
#   python train.py <...>
#       --img_c 3 --method_name erm_match --penalty_ws 10.0 --match_case 1 --epochs 25
BASELINE_ERM_CONFIG = {
    "method_name": "erm_match",
    "penalty_ws": 0,
    "match_case": 0,
    "epochs": 25,
}

PERFECT_MATCH_CONFIG = {
    "method_name": "erm_match",
    "penalty_ws": 0.1,
    "match_case": 1,
    "epochs": 40,
}

# DOMAIN GENERALIZATION VIA CAUSAL MATCHING

# Match Function
#   python train.py <...>
#       --method_name matchdg_ctr --match_case 0.0
#       --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos
#       --match_func_aug_case 1
# Classifier regularized on the Match Function
#   python train.py <...>
#       --method_name matchdg_erm --penalty_ws 0.1
#       --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1
#       --ctr_match_interrupt 5 --ctr_model_name resnet18 --epochs 25

MATCH_DG_CTR_CONFIG = {
    "method_name": "matchdg_ctr",
    "match_case": 0,
    "match_flag": 1,
    "epochs": 50,
    "pos_metric": "cos",
    "match_func_aug_case": 1,
}

MATCH_DG_ERM_CONFIG = {
    "method_name": "matchdg_erm",
    "penalty_ws": 0.1,
    "match_case": -1,
    # "ctr_model_name": "", ## this is be set when selecting the model
    "ctr_match_case": 0,
    "ctr_match_flag": 1,
    "ctr_match_interrupt": 5,
    "epochs": 40,
    "weight_decay": 0.001,
}

# HYBRID

HYBRID_CONFIG = {
    "method_name": "hybrid",
    "penalty_ws": 0.1,
    "match_case": 1,
    # "ctr_model_name": "", ## this is be set when selecting the model
    "ctr_match_case": 0,
    "ctr_match_flag": 1,
    "ctr_match_interrupt": 5,
    "epochs": 50,
    "weight_decay": 0.001,
}
