import logging
import os
import sys
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from robustdg_modified.config.args_mock import ArgsMock
from robustdg_modified.evaluation.match_eval import MatchEval
from robustdg_modified.utils.helper import cosine_similarity, embedding_dist

from .base_algo import BaseAlgo, TrainValTest
from .utils.valid_index import (
    get_desired_entries_for_column,
    get_desired_entries_in_both_columns,
)


class MatchDG(BaseAlgo):
    def __init__(
        self,
        args: ArgsMock,
        post_string: str,
        cuda: torch.device,
        base_res_dir: Path | str,
        ctr_phase: Literal[0, 1],
        model: nn.Module,
        ctr_model: nn.Module,
        optimizer: optim.Optimizer,
        data_loaders: dict[TrainValTest, DataLoader],
    ) -> None:

        """
        Initializes MatchDG class.

        Code has (mostly) been extracted from robustdg/algorithms/match_dg.py

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

            ctr_phase: Literal[0, 1]

                Indicates which phase should be run.
                    1 -> contrastive phase (ctr_phase)
                    0 -> erm phase (erm_phase)

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
        self.ctr_phase = ctr_phase

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

    def train(self):
        # Initialise and call train functions depending on the method's phase
        if self.ctr_phase:
            self.train_ctr_phase()
        else:
            self.train_erm_phase()

    def save_model_ctr_phase(self, epoch):
        # Store the weights of the model
        torch.save(
            self.phi.state_dict(),
            self.base_res_dir + "/Model_" + self.ctr_save_post_string + ".pth",
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

        save_path = self.base_res_dir + "/Model_" + self.ctr_save_post_string + ".pth"

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

    def train_ctr_phase(self):

        """
        Changed to allow invalid images, that is, to allow that some
        entries from self.get_match_function_batch() do not exist.
        """

        self.max_epoch = -1
        self.max_val_score = 0.0
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

            penalty_same_ctr = 0
            penalty_diff_ctr = 0
            penalty_same_hinge = 0
            penalty_diff_hinge = 0

            # Batch iteration over single epoch
            for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(
                self.train_dataset
            ):
                #         logging.info('Batch Idx: ', batch_idx)

                self.opt.zero_grad()
                loss_e = torch.tensor(0.0).to(self.cuda)

                x_e = x_e.to(self.cuda)
                y_e = torch.argmax(y_e, dim=1).to(self.cuda)
                d_e = torch.argmax(d_e, dim=1).numpy()

                same_ctr_loss = torch.tensor(0.0).to(self.cuda)
                diff_ctr_loss = torch.tensor(0.0).to(self.cuda)
                same_hinge_loss = torch.tensor(0.0).to(self.cuda)
                diff_hinge_loss = torch.tensor(0.0).to(self.cuda)

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

                    # Filter valid labels
                    valid_labels = label_match_tensor >= 0

                    # Contrastive Loss
                    same_neg_counter = 1
                    diff_neg_counter = 1
                    for y_c in range(self.args.out_classes):

                        pos_indices = label_match_tensor == y_c
                        neg_indices = label_match_tensor != y_c
                        # pos_indices = label_match[:, 0] == y_c
                        # neg_indices = label_match[:, 0] != y_c
                        # pos_feat_match = feat_match[pos_indices]
                        # neg_feat_match = feat_match[neg_indices]

                        # If no instances of label y_c in the current batch, continue
                        logging.debug(f"Label: {y_c}")
                        logging.debug(
                            f"Num positive matches: {pos_indices.shape[0]},"
                            f"Num negative matches: {neg_indices.shape[0]}"
                        )

                        if pos_indices.shape[0] == 0 or neg_indices.shape[0] == 0:
                            continue

                        # Iterating over anchors from different domains
                        for d_i in range(pos_indices.shape[1]):

                            domain_pos_indices_i = get_desired_entries_for_column(
                                valid_labels, d_i, pos_indices[:, d_i]
                            )
                            domain_neg_indices_i = get_desired_entries_for_column(
                                valid_labels, d_i, neg_indices[:, d_i]
                            )

                            pos_feat_i = feat_match[domain_pos_indices_i].to(self.cuda)
                            neg_feat_i = feat_match[domain_neg_indices_i].to(self.cuda)

                            if torch.sum(torch.isnan(neg_feat_i)):
                                logging.error("Non Reshaped X2 is Nan")
                                sys.exit()

                            # If I understood this function correctly, it will
                            # broadcast all positive matches to all negatives ones
                            # so it should work correctly
                            neg_dist = embedding_dist(
                                pos_feat_i,
                                neg_feat_i,
                                self.args.pos_metric,
                                self.args.tau,
                                xent=True,
                            )

                            del pos_feat_i
                            del neg_feat_i

                            if torch.sum(torch.isnan(neg_dist)):
                                logging.error("Neg Dist Nan")
                                sys.exit()

                            # Iterating pos dist for current anchor
                            for d_j in range(pos_indices.shape[1]):

                                mask_i, mask_j = get_desired_entries_in_both_columns(
                                    valid_labels, d_i, d_j
                                )
                                # We only only positive matches
                                # so we should filter out negative ones
                                valid_pos_indices = pos_indices[valid_labels]
                                mask_i = mask_i & valid_pos_indices
                                mask_j = mask_j & valid_pos_indices

                                feat_i = feat_match[mask_i].to(self.cuda)
                                feat_j = feat_match[mask_j].to(self.cuda)

                                if d_i != d_j:
                                    pos_dist = 1.0 - embedding_dist(
                                        feat_i,
                                        # pos_feat_match[:, d_i, :],
                                        feat_j,
                                        # pos_feat_match[:, d_j, :],
                                        self.args.pos_metric,
                                    )

                                    del feat_i
                                    del feat_j

                                    pos_dist = pos_dist / self.args.tau
                                    # TODO: Verify if setting to zero is ideal or
                                    # if we should ignore it
                                    pos_dist_inputted = (
                                        pos_dist
                                        if pos_dist.numel()
                                        else torch.Tensor([0]).to(self.cuda)
                                    )
                                    exp_pos = torch.exp(pos_dist_inputted)

                                    if torch.sum(torch.isnan(neg_dist)):
                                        logging.error("Pos Dist Nan")
                                        sys.exit()

                                    if torch.sum(
                                        torch.isnan(torch.log(exp_pos + neg_dist))
                                    ):
                                        logging.error("Xent Nan")
                                        sys.exit()

                                    diff_hinge_loss += -1 * torch.sum(
                                        pos_dist_inputted
                                        - torch.log(exp_pos + neg_dist)
                                    )

                                    diff_ctr_loss += torch.sum(neg_dist)
                                    diff_neg_counter += pos_dist.shape[0]

                    same_ctr_loss = same_ctr_loss / same_neg_counter
                    diff_ctr_loss = diff_ctr_loss / diff_neg_counter
                    same_hinge_loss = same_hinge_loss / same_neg_counter
                    diff_hinge_loss = diff_hinge_loss / diff_neg_counter

                    penalty_same_ctr += float(same_ctr_loss)
                    penalty_diff_ctr += float(diff_ctr_loss)
                    penalty_same_hinge += float(same_hinge_loss)
                    penalty_diff_hinge += float(diff_hinge_loss)

                    loss_e += (
                        (epoch - self.args.penalty_s)
                        / (self.args.epochs - self.args.penalty_s)
                    ) * diff_hinge_loss

                if not loss_e.requires_grad:
                    continue

                loss_e.backward(retain_graph=False)
                self.opt.step()

                del same_ctr_loss
                del diff_ctr_loss
                del same_hinge_loss
                del diff_hinge_loss
                torch.cuda.empty_cache()

            logging.info(
                f"Train Loss Ctr : "
                f"{penalty_same_ctr},"
                f"{penalty_diff_ctr},"
                f"{penalty_same_hinge},"
                f"{penalty_diff_hinge}"
            )
            logging.info(f"Done Training for epoch:  {epoch}")

            if (epoch + 1) % 5 == 0:

                test_method = MatchEval(
                    self.args,
                    self.train_dataset,
                    self.val_dataset,
                    self.test_dataset,
                    self.base_res_dir,
                    self.run,
                    self.cuda,
                    self.phi,  # added for simpler match_eval
                )
                # Compute test metrics: Mean Rank
                test_method.phi = self.phi
                test_method.get_metric_eval()

                # Save the model's weights post training
                if (
                    test_method.metric_score["TopK Perfect Match Score"]
                    >= self.max_val_score
                ):
                    self.max_val_score = test_method.metric_score[
                        "TopK Perfect Match Score"
                    ]
                    self.max_epoch = epoch
                    self.save_model_ctr_phase(epoch)

                logging.info(
                    f"Current Best Epoch: {self.max_epoch}"
                    f" with TopK Overlap: {self.max_val_score}"
                )

    def train_erm_phase(self):

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
                train_acc = 0.0
                train_size = 0

                # Batch iteration over single epoch
                for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(
                    self.train_dataset
                ):
                    #         logging.info('Batch Idx: ', batch_idx)

                    self.opt.zero_grad()
                    loss_e = torch.tensor(0.0).to(self.cuda)

                    x_e = x_e.to(self.cuda)
                    y_e = torch.argmax(y_e, dim=1).to(self.cuda)
                    d_e = torch.argmax(d_e, dim=1).numpy()

                    # Forward Pass
                    out = self.phi(x_e)
                    erm_loss_extra = F.cross_entropy(out, y_e.long()).to(self.cuda)
                    penalty_erm_extra += float(erm_loss_extra)

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
                    f"{penalty_erm},"
                    f"{penalty_ws}"
                )
                logging.info(f"Train Acc Env :  {100 * train_acc / train_size}")
                logging.info(f"Done Training for epoch:  {epoch}")

                # Train Dataset Accuracy
                self.train_acc.append(100 * train_acc / train_size)

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
