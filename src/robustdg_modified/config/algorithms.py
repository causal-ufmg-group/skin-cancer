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


# No domain configuration
NO_DOMAIN_CONFIG = {"method_name": "no_domain"}

# Configurations based off of:
#   robustdg/notebooks/reproduce_results.ipynb
#   robustdg/reproduce_scripts/mnist_run.py
BASELINE_ERM_CONFIG = {
    "method_name": "erm_match",
    "penalty_ws": 0,
    "match_case": 0,
    "epochs": 25,
}

PERFECT_MATCH_CONFIG = {
    "method_name": "erm_match",
    "penalty_ws": 10,
    "match_case": 1,
    "epochs": 25,
}

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
    "epochs": 25,
}
