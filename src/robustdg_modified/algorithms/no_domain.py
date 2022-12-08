import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from robustdg_modified.config.args_mock import ArgsMock

from .base_algo import BaseAlgo, TrainValTest


class NoDomain(BaseAlgo):

    """
    Algorithm for training a neural network without any information about domains.
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

        This class is designed to be similar to the ones used for RobustDG.

        -----
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
                robustdg_modified.dataset.TestDataset.
        """
        super().__init__(
            args,
            f"NO_DOMAIN_{run}_",
            cuda,
            base_res_dir,
            model,
            optimizer,
            data_loaders,
        )

    def _train_one_epoch(self, epoch_index: int) -> torch.Tensor:

        "See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html"

        last_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (x_e, y_e, _, _, _) in enumerate(self.train_dataset):

            # Zero your gradients for every batch!
            self.opt.zero_grad()

            # Make predictions for this batch
            x_e = x_e.to(self.cuda)
            y_e = y_e.to(self.cuda)

            outputs = self.phi(x_e)

            # Compute the loss and its gradients
            loss = nn.functional.cross_entropy(outputs, y_e)
            loss.backward()

            # Adjust learning weights
            self.opt.step()

            train_correct += torch.sum(
                torch.argmax(outputs, dim=1) == torch.argmax(y_e, dim=1)
            ).item()
            train_total += y_e.shape[0]

        epoch_accuracy = train_correct / train_total

        return last_loss, epoch_accuracy

    def train(self):

        self.max_epoch = -1
        self.max_val_acc = 0.0

        for epoch in range(self.args.epochs):

            logging.info(f"Started epoch: {epoch}")

            self.phi.train(True)
            last_loss, epoch_accuracy = self._train_one_epoch(epoch)

            self.phi.train(False)

            logging.info(f"Epoch Final Train Loss: {last_loss}")
            logging.info(f"Epoch Train Acc: {100 * epoch_accuracy}")
            logging.info(f"Done Training for epoch: {epoch}")

            # Train Dataset Accuracy
            self.train_acc.append(100 * epoch_accuracy)

            # Val Dataset Accuracy
            self.val_acc.append(self.get_test_accuracy("val"))

            # Test Dataset Accuracy
            self.final_acc.append(self.get_test_accuracy("test"))

            # Save the model if current best epoch as per validation loss
            if self.val_acc[-1] > self.max_val_acc:
                self.max_val_acc = self.val_acc[-1]
                self.max_epoch = epoch
                self.save_model()

            logging.info(
                f"Current Best Epoch: {self.max_epoch}"
                f" with Test Accuracy: {self.final_acc[self.max_epoch]}"
            )
