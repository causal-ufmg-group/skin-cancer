import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from more_itertools import chunked
from torch import nn, optim
from torch.utils.data import DataLoader

from robustdg.utils.match_function import get_matched_pairs
from robustdg_modified.config.args_mock import ArgsMock

TrainValTest = Literal["train", "validation", "test"]


class BaseAlgo:

    """
    Base class for algorithms.
    """

    def __init__(
        self,
        args: ArgsMock,
        run: int,
        cuda: torch.device,
        base_res_dir: Path | str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data_loaders: dict[TrainValTest, DataLoader],
    ) -> None:

        """
        Initializes base class.

        Code has (mostly) been extracted from robustdg/algorithms/algo.py

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

                Neural network to be trained.

                In the paper, this is the Phi representation.

            optimizer: optim.Optimizer

                Pytorch optimizer.

            data_loaders: dict[
                Literal["train", "validation", "test"], torch.utils.data.DataLoader
            ]

                DataLoaders for train, validation and test datasets.

                DataLoaders for train/validation should contain an instance of
                    robustdg_modified.dataset.TrainDataset
                since some variables/methods depend on it.

                As for test it can contain an instance of
                    robustdg_modified.dataset.TestDataset.s

        -----
        Removed RobustDG Parameters:

            <train,val,test>_dataset: dict[str | ...]

                Replaced by new parameter: data_loaders.

                Function get_dataloader in robustdg.utils.helper.py
                defines required keys.
        """

        self.args = args

        # Dataset information
        self.train_dataset = data_loaders["train"]

        # TODO: Make sure the if statement below in unnecessary
        # if args.method_name == "matchdg_ctr":
        #     self.val_dataset = val_dataset
        # else:
        #     self.val_dataset = val_dataset["data_loader"]
        self.val_dataset = data_loaders["validation"]
        self.test_dataset = data_loaders["test"]

        self.train_domains = self.train_dataset.dataset.list_domains
        self.total_domains = len(self.train_domains)
        self.domain_size = self.train_dataset.dataset.base_domain_size
        self.training_list_size = self.train_dataset.dataset.training_list_size

        # Neural Network information
        self.phi = model
        self.opt = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)

        # Others parameters
        self.base_res_dir = base_res_dir
        self.run = run
        self.cuda = cuda

        self.post_string = (
            f"{args.penalty_ws}_{args.penalty_diff_ctr}_{args.match_case}"
            f"_{args.match_interrupt}_{args.match_flag}_{run}"
            f"_{args.pos_metric}_{args.model_name}"
        )

        self.final_acc = []
        self.val_acc = []
        self.train_acc = []

    def save_model(self):
        # Store the weights of the model
        torch.save(
            self.phi.state_dict(),
            self.base_res_dir + "/Model_" + self.post_string + ".pth",
        )

        # Store the validation, test loss over the training epochs
        np.save(
            self.base_res_dir + "/Val_Acc_" + self.post_string + ".npy",
            np.array(self.val_acc),
        )
        np.save(
            self.base_res_dir + "/Test_Acc_" + self.post_string + ".npy",
            np.array(self.final_acc),
        )

    def get_match_function(self, inferred_match, phi):

        data_matched, domain_data, _ = get_matched_pairs(
            self.args,
            self.cuda,
            self.train_dataset,
            self.domain_size,
            self.total_domains,
            self.training_list_size,
            phi,
            self.args.match_case,
            self.args.perfect_match,
            inferred_match,
        )

        # Randomly Shuffle list of matched data indices and divide as per batch sizes
        random.shuffle(data_matched)
        data_matched = list(chunked(data_matched, self.args.batch_size))

        return data_matched, domain_data

    def get_match_function_batch(self, batch_idx):
        curr_data_matched = self.data_matched[batch_idx]
        curr_batch_size = len(curr_data_matched)

        data_match_tensor = []
        label_match_tensor = []
        for idx in range(curr_batch_size):
            data_temp = []
            label_temp = []
            for d_i in range(len(curr_data_matched[idx])):
                key = random.choice(curr_data_matched[idx][d_i])
                data_temp.append(self.domain_data[d_i]["data"][key])
                label_temp.append(self.domain_data[d_i]["label"][key])

            data_match_tensor.append(torch.stack(data_temp))
            label_match_tensor.append(torch.stack(label_temp))

        data_match_tensor = torch.stack(data_match_tensor)
        label_match_tensor = torch.stack(label_match_tensor)

        return data_match_tensor, label_match_tensor, curr_batch_size

    def get_test_accuracy(self, case):
        import opacus

        if self.args.dp_noise:
            opacus.autograd_grad_sample.disable_hooks()
            # self.privacy_engine.module.disable_hooks()

        # Test Env Code
        test_acc = 0.0
        test_size = 0
        if case == "val":
            dataset = self.val_dataset
        elif case == "test":
            dataset = self.test_dataset

        for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(dataset):
            with torch.no_grad():

                self.opt.zero_grad()
                x_e = x_e.to(self.cuda)
                y_e = torch.argmax(y_e, dim=1).to(self.cuda)

                # Forward Pass
                out = self.phi(x_e)

                test_acc += torch.sum(torch.argmax(out, dim=1) == y_e).item()
                test_size += y_e.shape[0]

                # To avoid CUDA memory issues
                if self.args.dp_noise:
                    self.opt.zero_grad()

        print(" Accuracy: ", case, 100 * test_acc / test_size)

        # self.privacy_engine.module.enable_hooks()
        opacus.autograd_grad_sample.enable_hooks()
        return 100 * test_acc / test_size
