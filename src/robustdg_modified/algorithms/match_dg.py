import os
import sys
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from robustdg.utils.helper import cosine_similarity, embedding_dist
from robustdg_modified.config.args_mock import ArgsMock

from .base_algo import BaseAlgo, TrainValTest


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
            self,
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
                #         print('Batch Idx: ', batch_idx)

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
                    data_match = data_match.flatten(start_dim=0, end_dim=1)
                    feat_match = self.phi(data_match)

                    label_match = label_match_tensor.to(self.cuda)
                    label_match = torch.squeeze(
                        label_match.flatten(start_dim=0, end_dim=1)
                    )

                    # Creating tensor of shape ( domain size, total domains, feat size )
                    feat_match = torch.stack(
                        torch.split(feat_match, len(self.train_domains))
                    )
                    label_match = torch.stack(
                        torch.split(label_match, len(self.train_domains))
                    )

                    # Contrastive Loss
                    same_neg_counter = 1
                    diff_neg_counter = 1
                    for y_c in range(self.args.out_classes):

                        pos_indices = label_match[:, 0] == y_c
                        neg_indices = label_match[:, 0] != y_c
                        pos_feat_match = feat_match[pos_indices]
                        neg_feat_match = feat_match[neg_indices]

                        # If no instances of label y_c in the current batch, continue
                        print(pos_feat_match.shape[0], neg_feat_match.shape[0], y_c)

                        if pos_feat_match.shape[0] == 0 or neg_feat_match.shape[0] == 0:
                            continue

                        # Iterating over anchors from different domains
                        for d_i in range(pos_feat_match.shape[1]):
                            if torch.sum(torch.isnan(neg_feat_match)):
                                print("Non Reshaped X2 is Nan")
                                sys.exit()

                            diff_neg_feat_match = neg_feat_match.view(
                                neg_feat_match.shape[0] * neg_feat_match.shape[1],
                                neg_feat_match.shape[2],
                            )

                            if torch.sum(torch.isnan(diff_neg_feat_match)):
                                print("Reshaped X2 is Nan")
                                sys.exit()

                            neg_dist = embedding_dist(
                                pos_feat_match[:, d_i, :],
                                diff_neg_feat_match[:, :],
                                self.args.pos_metric,
                                self.args.tau,
                                xent=True,
                            )
                            if torch.sum(torch.isnan(neg_dist)):
                                print("Neg Dist Nan")
                                sys.exit()

                            # Iterating pos dist for current anchor
                            for d_j in range(pos_feat_match.shape[1]):
                                if d_i != d_j:
                                    pos_dist = 1.0 - embedding_dist(
                                        pos_feat_match[:, d_i, :],
                                        pos_feat_match[:, d_j, :],
                                        self.args.pos_metric,
                                    )
                                    pos_dist = pos_dist / self.args.tau
                                    if torch.sum(torch.isnan(neg_dist)):
                                        print("Pos Dist Nan")
                                        sys.exit()

                                    if torch.sum(
                                        torch.isnan(
                                            torch.log(torch.exp(pos_dist) + neg_dist)
                                        )
                                    ):
                                        print("Xent Nan")
                                        sys.exit()

                                    diff_hinge_loss += -1 * torch.sum(
                                        pos_dist
                                        - torch.log(torch.exp(pos_dist) + neg_dist)
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

            print(
                "Train Loss Ctr : ",
                penalty_same_ctr,
                penalty_diff_ctr,
                penalty_same_hinge,
                penalty_diff_hinge,
            )
            print("Done Training for epoch: ", epoch)

            if (epoch + 1) % 5 == 0:

                from robustdg.evaluation.match_eval import MatchEval

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
                    > self.max_val_score
                ):
                    self.max_val_score = test_method.metric_score[
                        "TopK Perfect Match Score"
                    ]
                    self.max_epoch = epoch
                    self.save_model_ctr_phase(epoch)

                print(
                    "Current Best Epoch: ",
                    self.max_epoch,
                    " with TopK Overlap: ",
                    self.max_val_score,
                )

    def train_erm_phase(self):

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
                    #         print('Batch Idx: ', batch_idx)

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
                        data_match = data_match.flatten(start_dim=0, end_dim=1)
                        feat_match = self.phi(data_match)

                        label_match = label_match_tensor.to(self.cuda)
                        label_match = torch.squeeze(
                            label_match.flatten(start_dim=0, end_dim=1)
                        )

                        erm_loss += F.cross_entropy(feat_match, label_match.long()).to(
                            self.cuda
                        )
                        penalty_erm += float(erm_loss)

                        train_acc += torch.sum(
                            torch.argmax(feat_match, dim=1) == label_match
                        ).item()
                        train_size += label_match.shape[0]

                        # Creating tensor of shape
                        #   ( domain size, total domains, feat size )
                        feat_match = torch.stack(
                            torch.split(feat_match, len(self.train_domains))
                        )
                        label_match = torch.stack(
                            torch.split(label_match, len(self.train_domains))
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
                                                feat_match[:, d_i, :],
                                                feat_match[:, d_j, :],
                                            )
                                        )

                                    pos_match_counter += feat_match.shape[0]

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

                print("Train Loss Basic : ", penalty_erm_extra, penalty_erm, penalty_ws)
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
                    self.save_model_erm_phase(run_erm)

                print(
                    "Current Best Epoch: ",
                    self.max_epoch,
                    " with Test Accuracy: ",
                    self.final_acc[self.max_epoch],
                )
