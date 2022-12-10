import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from robustdg_modified.config.args_mock import ArgsMock
from robustdg_modified.utils.match_function import get_matched_pairs


class MatchEval:

    """
    Mock class for MatchEval required for MatchDG algorithm.

    Changes:
        - Removed robustdg.evaluation.base_eval.BaseEval base class
          since it is not necessary.
        - For out case, we will only be dealing with validation, which has the same
          domains as train.
    """

    def __init__(
        self,
        args: ArgsMock,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        test_dataset: DataLoader,
        base_res_dir: Path | str,
        run: int,
        cuda: torch.device,
        model: nn.Module,
    ):

        """
        ""
        Initializes an instance of MatchEval.

        Code has (mostly) been extracted from robustdg/evaluation/match_eval.py

        Parameters below are divided into three categories:
            - Required RobustDG parameters:
                - Parameters required when using RobustDG algorithms.
            - General purpose parameters
                - Usually related to torch.utils.data.Dataset base class.
            - Removed RobustDG parameters
                - RobustDG parameters not required anymore because of how
                this is implemented.
                - Documented mainly for future reference.

        -----
        RobustDG Parameters:

            args: ArgsMock | argparse.Argument

                Configuration for robustdg.

                See ArgsMock documentation for full list of parameters.

            <train,val,test>data_loaders: torch.utils.data.DataLoader

                DataLoaders for train, validation and test datasets.

                DataLoaders for train/validation should contain an instance of
                    robustdg_modified.dataset.TrainDataset
                since some variables/methods depend on it.

                As for test it can contain an instance of
                    robustdg_modified.dataset.TestDataset.

            cuda: torch.device

                Device to run algorithms on.

            run: int

                If method is to be run more than once,
                this parameter is which iteration we are at.

                Number of runs are determined by args.n_runs.

            base_res_dir: Path | str

                Directory files will be saved to.

        -----
        General Purpose Parameters:

            model: nn.Module

                Neural network to be evaluated.
        """

        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.base_res_dir = base_res_dir
        self.run = run
        self.cuda = cuda

        self.phi = model

        self.metric_score = dict()

    def get_metric_eval(self):

        STAGE_TO_DATALOADER = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }
        stage = self.args.match_func_data_case

        data_loader = STAGE_TO_DATALOADER[stage]
        total_domains = len(data_loader.dataset.list_domains)
        base_domain_size = data_loader.dataset.base_domain_size
        domain_size_list = data_loader.dataset.training_list_size

        inferred_match = 1

        # Self Augmentation Match Function evaluation will always follow perfect matches
        if self.args.match_func_aug_case:
            perfect_match = 1
        else:
            perfect_match = self.args.perfect_match

        data_matched, domain_data, perfect_match_rank = get_matched_pairs(
            self.args,
            self.cuda,
            data_loader,
            base_domain_size,
            total_domains,
            domain_size_list,
            self.phi,
            self.args.match_case,
            perfect_match,
            inferred_match,
        )

        perfect_match_rank = np.array(perfect_match_rank)

        self.metric_score["Perfect Match Score"] = (
            100 * np.sum(perfect_match_rank < 1) / perfect_match_rank.shape[0]
        )
        self.metric_score["TopK Perfect Match Score"] = (
            100
            * np.sum(perfect_match_rank < self.args.top_k)
            / perfect_match_rank.shape[0]
        )
        self.metric_score["Perfect Match Rank"] = np.mean(perfect_match_rank)

        logging.info(
            f"Perfect Match Score:  {self.metric_score['Perfect Match Score']}"
        )
        logging.info(
            f"TopK Perfect Match Score: {self.metric_score['TopK Perfect Match Score']}"
        )
        logging.info(f"Perfect Match Rank:  {self.metric_score['Perfect Match Rank']}")
