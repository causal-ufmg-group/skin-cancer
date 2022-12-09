import logging
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from robustdg_modified.config.args_mock import ArgsMock
from robustdg_modified.utils.helper import cosine_similarity

from .base_algo import BaseAlgo, TrainValTest
from .utils.valid_index import get_desired_entries_in_both_columns


class Hybrid(BaseAlgo):
    def __init__(
        self,
        args: ArgsMock,
        post_string: str,
        cuda: torch.device,
        base_res_dir: Path | str,
        model: nn.Module,
        ctr_model: nn.Module,
        optimizer: optim.Optimizer,
        data_loaders: dict[TrainValTest, DataLoader],
    ) -> None:

        """
        Initializes Hybrid class.

        Code has (mostly) been extracted from robustdg/algorithms/hybrid.py

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

            ctr_model: nn.Module

                Neural network to be used after ctr_model.

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
            post_string,
            cuda,
            base_res_dir,
            model,
            optimizer,
            data_loaders,
        )

        self.ctr_phi = ctr_model

        self.ctr_save_post_string = (
            str(self.args.match_case)
            + "_"
            + str(self.args.match_interrupt)
            + "_"
            + str(self.args.match_flag)
            + "_"
            + str(self.run)
            + "_"
            + self.args.model_name
        )
        self.ctr_load_post_string = (
            str(self.args.ctr_match_case)
            + "_"
            + str(self.args.ctr_match_interrupt)
            + "_"
            + str(self.args.ctr_match_flag)
            + "_"
            + str(self.run)
            + "_"
            + self.args.ctr_model_name
        )

    def save_model_erm_phase(self, run):

        if not os.path.exists(self.base_res_dir + "/" + self.ctr_load_post_string):
            os.makedirs(self.base_res_dir + "/" + self.ctr_load_post_string)

        # Store the weights of the model
        torch.save(
            self.phi.state_dict(),
            self.base_res_dir
            + "/"
            + self.ctr_load_post_string
            + "/Model_"
            + self.post_string
            + "_"
            + str(run)
            + ".pth",
        )

    def init_erm_phase(self):

        # Load MatchDG CTR phase model from the saved weights
        base_res_dir = (
            "results/"
            + self.args.dataset_name
            + "/"
            + "matchdg_ctr"
            + "/"
            + self.args.ctr_match_layer
            + "/"
            + "train_"
            + str(self.args.train_domains)
        )

        save_path = base_res_dir + "/Model_" + self.ctr_load_post_string + ".pth"

        if Path(save_path).exists():
            self.ctr_phi.load_state_dict(torch.load(save_path))
            self.ctr_phi.eval()

        # Inferred Match Case
        if self.args.match_case == -1:
            inferred_match = 1
        # x% percentage match initial strategy
        else:
            inferred_match = 0

        data_matched, domain_data = self.get_match_function(
            inferred_match, self.ctr_phi
        )

        return data_matched, domain_data

    def train(self):

        """
        Changed to allow invalid images, that is, to allow that some
        entries from self.get_match_function_batch() do not exist.
        """

        for run_erm in range(self.args.n_runs_matchdg_erm):

            self.max_epoch = -1
            self.max_val_acc = 0.0
            for epoch in range(self.args.epochs):

                if epoch == 0:
                    self.data_matched, self.domain_data = self.init_erm_phase()
                elif epoch % self.args.match_interrupt == 0 and self.args.match_flag:
                    inferred_match = 1
                    (
                        self.data_match_tensor,
                        self.label_match_tensor,
                    ) = self.get_match_function(inferred_match, self.phi)

                penalty_erm = 0
                penalty_erm_extra = 0
                penalty_ws = 0
                penalty_aug = 0
                train_acc = 0.0
                train_size = 0

                # Batch iteration over single epoch
                for batch_idx, (x_e, x_org_e, y_e, d_e, idx_e, obj_e) in enumerate(
                    self.train_dataset
                ):
                    #         logging.info('Batch Idx: ', batch_idx)

                    self.opt.zero_grad()
                    loss_e = torch.tensor(0.0).to(self.cuda)

                    x_e = x_e.to(self.cuda)
                    x_org_e = x_org_e.to(self.cuda)
                    y_e = torch.argmax(y_e, dim=1).to(self.cuda)
                    d_e = torch.argmax(d_e, dim=1).numpy()

                    # Forward Pass
                    out = self.phi(x_e)
                    erm_loss_extra = F.cross_entropy(out, y_e.long()).to(self.cuda)
                    penalty_erm_extra += float(erm_loss_extra)

                    # Perfect Match on Augmentations
                    out_org = self.phi(x_org_e)
                    #                     diff_indices= out != out_org
                    #                     out= out[diff_indices]
                    #                     out_org= out_org[diff_indices]
                    augmentation_loss = torch.tensor(0.0).to(self.cuda)
                    if self.args.pos_metric == "l2":
                        augmentation_loss += torch.sum(
                            torch.sum((out - out_org) ** 2, dim=1)
                        )
                    elif self.args.pos_metric == "l1":
                        augmentation_loss += torch.sum(
                            torch.sum(torch.abs(out - out_org), dim=1)
                        )
                    elif self.args.pos_metric == "cos":
                        augmentation_loss += torch.sum(cosine_similarity(out, out_org))

                    augmentation_loss = augmentation_loss / out.shape[0]
                    penalty_aug += float(augmentation_loss)

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
                        feat_match = self.phi(data_match)
                        #                     logging.info(feat_match.shape)

                        # Filter valid labels
                        valid_labels = label_match_tensor >= 0
                        label_match = label_match_tensor[valid_labels].to(self.cuda)

                        erm_loss += F.cross_entropy(feat_match, label_match.long()).to(
                            self.cuda
                        )
                        penalty_erm += float(erm_loss)
                        loss_e += erm_loss

                        train_acc += torch.sum(
                            torch.argmax(feat_match, dim=1) == label_match
                        ).item()
                        train_size += label_match.shape[0]

                        train_size += label_match.shape[0]

                        # Positive Match Loss
                        pos_match_counter = 0
                        for d_i in range(valid_labels.shape[1]):
                            #                 if d_i != base_domain_idx:
                            #                     continue
                            for d_j in range(valid_labels.shape[1]):

                                # Use valid labels to detect which
                                # images should be used
                                mask_i, mask_j = get_desired_entries_in_both_columns(
                                    valid_labels, d_i, d_j
                                )

                                feat_i = feat_match[mask_i]
                                feat_j = feat_match[mask_j]

                                if d_j > d_i:
                                    if self.args.pos_metric == "l2":
                                        wasserstein_loss += torch.sum(
                                            (feat_i - feat_j) ** 2
                                        )
                                    elif self.args.pos_metric == "l1":
                                        wasserstein_loss += torch.sum(
                                            torch.abs(feat_i - feat_j)
                                        )
                                    elif self.args.pos_metric == "cos":
                                        wasserstein_loss += torch.sum(
                                            cosine_similarity(feat_i, feat_j)
                                        )

                                    pos_match_counter += feat_i.shape[0]

                        wasserstein_loss = wasserstein_loss / pos_match_counter
                        penalty_ws += float(wasserstein_loss)

                        loss_e += (
                            self.args.penalty_ws
                            * (epoch - self.args.penalty_s)
                            / (self.args.epochs - self.args.penalty_s)
                        ) * wasserstein_loss
                        loss_e += self.args.penalty_aug * augmentation_loss
                        loss_e += erm_loss
                        loss_e += erm_loss_extra

                    loss_e.backward(retain_graph=False)
                    self.opt.step()

                    del erm_loss_extra
                    del erm_loss
                    del wasserstein_loss
                    del loss_e
                    torch.cuda.empty_cache()

                logging.info(
                    f"Train Loss Basic : "
                    f"{penalty_erm_extra},"
                    f"{penalty_aug},"
                    f"{penalty_erm},"
                    f"{penalty_ws}"
                )
                logging.info(f"Train Acc Env :  {100 * train_acc / train_size}")
                logging.info(f"Done Training for epoch:  {epoch}")

                # Val Dataset Accuracy
                self.val_acc.append(self.get_test_accuracy("val"))

                # Test Dataset Accuracy
                # Only do test in the end to reduce computational cost
                # self.final_acc.append(self.get_test_accuracy("test"))
                self.final_acc.append([])

                # Save the model if current best epoch as per validation loss
                if self.val_acc[-1] > self.max_val_acc:
                    self.max_val_acc = self.val_acc[-1]
                    self.max_epoch = epoch
                    self.save_model_erm_phase(run_erm)

                logging.info(
                    f"Current Best Epoch: {self.max_epoch}"
                    f" with Test Accuracy: {self.final_acc[self.max_epoch]}"
                )
