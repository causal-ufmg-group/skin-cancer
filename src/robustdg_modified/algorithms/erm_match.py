from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from robustdg.utils.helper import cosine_similarity
from robustdg_modified.config.args_mock import ArgsMock

from .base_algo import BaseAlgo, TrainValTest


class ErmMatch(BaseAlgo):
    def __init__(
        self,
        args: ArgsMock,
        post_string: str,
        cuda: torch.device,
        base_res_dir: Path | str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        data_loaders: dict[TrainValTest, DataLoader],
    ) -> None:

        """
        Initializes ErmMatch class.

        Code has (mostly) been extracted from robustdg/algorithms/erm_match.py

        -----
        RobustDG Parameters:

            args: ArgsMock | argparse.Argument

                Configuration for robustdg.

                See ArgsMock documentation for full list of parameters.

            cuda: torch.device

                Device to run algorithms on.

            post_string: str

                String added to be an identifier for this run.

                Serves a similar purpose as "run" parameter for base class.

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
                    robustdg_modified.dataset.TestDataset.
        """

        super().__init__(
            self,
            args,
            post_string,
            cuda,
            base_res_dir,
            model,
            optimizer,
            data_loaders,
        )

    def train(self):

        self.max_epoch = -1
        self.max_val_acc = 0.0
        for epoch in range(self.args.epochs):

            if epoch == 0:
                inferred_match = 0
                self.data_matched, self.domain_data = self.get_match_function(
                    inferred_match, self.phi
                )
            elif epoch % self.args.match_interrupt == 0 and self.args.match_flag:
                inferred_match = 1
                self.data_matched, self.domain_data = self.get_match_function(
                    inferred_match, self.phi
                )

            penalty_erm = 0
            penalty_ws = 0
            train_acc = 0.0
            train_size = 0

            # Batch iteration over single epoch
            for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(
                self.train_dataset
            ):

                #                 self.opt.zero_grad()
                loss_e = torch.tensor(0.0).to(self.cuda)

                x_e = x_e.to(self.cuda)
                y_e = torch.argmax(y_e, dim=1).to(self.cuda)
                d_e = torch.argmax(d_e, dim=1).numpy()

                wasserstein_loss = torch.tensor(0.0).to(self.cuda)
                erm_loss = torch.tensor(0.0).to(self.cuda)
                if epoch > self.args.penalty_s:
                    # To cover the varying size of the last batch for
                    # data_match_tensor_split, label_match_tensor_split
                    total_batch_size = len(self.data_matched)
                    if batch_idx >= total_batch_size:
                        break

                    # Sample batch from matched data points
                    (
                        data_match_tensor,
                        label_match_tensor,
                        curr_batch_size,
                    ) = self.get_match_function_batch(batch_idx)

                    data_match = data_match_tensor.to(self.cuda)
                    data_match = data_match.flatten(start_dim=0, end_dim=1)
                    feat_match = self.phi(data_match)
                    #                     print(feat_match.shape)

                    label_match = label_match_tensor.to(self.cuda)
                    label_match = torch.squeeze(
                        label_match.flatten(start_dim=0, end_dim=1)
                    )

                    erm_loss += F.cross_entropy(feat_match, label_match.long()).to(
                        self.cuda
                    )
                    penalty_erm += float(erm_loss)
                    loss_e += erm_loss

                    train_acc += torch.sum(
                        torch.argmax(feat_match, dim=1) == label_match
                    ).item()
                    train_size += label_match.shape[0]

                    # Creating tensor of shape ( domain size, total domains, feat size )
                    feat_match = torch.stack(
                        torch.split(feat_match, len(self.train_domains))
                    )

                    # Positive Match Loss
                    pos_match_counter = 0
                    for d_i in range(feat_match.shape[1]):
                        #                 if d_i != base_domain_idx:
                        #                     continue
                        for d_j in range(feat_match.shape[1]):
                            if d_j > d_i:
                                if self.args.pos_metric == "l2":
                                    wasserstein_loss += torch.sum(
                                        torch.sum(
                                            (
                                                feat_match[:, d_i, :]
                                                - feat_match[:, d_j, :]
                                            )
                                            ** 2,
                                            dim=1,
                                        )
                                    )
                                elif self.args.pos_metric == "l1":
                                    wasserstein_loss += torch.sum(
                                        torch.sum(
                                            torch.abs(
                                                feat_match[:, d_i, :]
                                                - feat_match[:, d_j, :]
                                            ),
                                            dim=1,
                                        )
                                    )
                                elif self.args.pos_metric == "cos":
                                    wasserstein_loss += torch.sum(
                                        cosine_similarity(
                                            feat_match[:, d_i, :], feat_match[:, d_j, :]
                                        )
                                    )

                                pos_match_counter += feat_match.shape[0]

                    wasserstein_loss = wasserstein_loss / pos_match_counter
                    penalty_ws += float(wasserstein_loss)

                    if epoch >= self.args.match_interrupt and self.args.match_flag == 1:
                        loss_e += (
                            self.args.penalty_ws
                            * (epoch - self.args.penalty_s - self.args.match_interrupt)
                            / (
                                self.args.epochs
                                - self.args.penalty_s
                                - self.args.match_interrupt
                            )
                        ) * wasserstein_loss
                    else:
                        loss_e += (
                            self.args.penalty_ws
                            * (epoch - self.args.penalty_s)
                            / (self.args.epochs - self.args.penalty_s)
                        ) * wasserstein_loss

                loss_e.backward(retain_graph=False)

                if self.args.dp_noise and self.args.dp_attach_opt:
                    if batch_idx % 10 == 9:
                        self.opt.step()
                        self.opt.zero_grad()
                    else:
                        self.opt.virtual_step()
                else:
                    self.opt.step()
                    self.opt.zero_grad()

                # Removed some comments from here

                del erm_loss
                del wasserstein_loss
                del loss_e
                torch.cuda.empty_cache()

            #             print('Gradient Norm: ', total_grad_norm)

            print("Train Loss Basic : ", penalty_erm, penalty_ws)
            print("Train Acc Env : ", 100 * train_acc / train_size)
            print("Done Training for epoch: ", epoch)

            # Train Dataset Accuracy
            self.train_acc.append(100 * train_acc / train_size)

            # Val Dataset Accuracy
            self.val_acc.append(self.get_test_accuracy("val"))

            # Test Dataset Accuracy
            self.final_acc.append(self.get_test_accuracy("test"))

            # Save the model if current best epoch as per validation loss
            if self.val_acc[-1] > self.max_val_acc:
                self.max_val_acc = self.val_acc[-1]
                self.max_epoch = epoch
                self.save_model()

            print(
                "Current Best Epoch: ",
                self.max_epoch,
                " with Test Accuracy: ",
                self.final_acc[self.max_epoch],
            )
