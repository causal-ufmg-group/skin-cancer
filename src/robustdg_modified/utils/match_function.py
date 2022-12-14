"""
Changes:

    Loading all images into RAM is not viable, so it only loads them when necessary now.
        These changes can be found in comments prefixed by "Changed to ..."

        They can be found in:
            1) this file
            2) robustdg_modified/algorithms/base_algo.py

        Separated domain index from image index. The latter is necessary for
        loading images only when necessary.

    Created helper function to load images from indexes.
        function: load_images_from_indexes
"""
import logging
from typing import Iterable

import numpy as np
import torch
from scipy.stats import bernoulli
from torch import nn
from torch.utils.data import DataLoader, Dataset

from robustdg_modified.config.args_mock import ArgsMock


def init_data_match_dict(args: ArgsMock, keys, vals, variation):

    data = {}

    for key in keys:
        data[key] = {}
        if variation:
            val_dim = vals[key]
        else:
            val_dim = vals

        # Changed to ignore it
        # data[key]["data"] = torch.rand((val_dim, args.img_c, args.img_w, args.img_h))
        data[key]["data_idx"] = torch.randint(0, 1, (val_dim, 1))

        data[key]["label"] = torch.rand((val_dim, 1))
        data[key]["idx"] = torch.randint(0, 1, (val_dim, 1))
        data[key]["obj"] = torch.randint(0, 1, (val_dim, 1))

    return data


def load_images_from_indexes(
    dataset: Dataset, indexes: Iterable[torch.Tensor | int]
) -> torch.Tensor:

    """
    Load various images from their indexes.

    Helper function to avoid loading all images into memory at once.

    ----
    Parameters:

        dataloader: torch.utils.data.Dataset

            Dataset from which to load images.

        indexes: Iterable[torch.Tensor | int]
            List of indexes (each index can be Tensor containing only a number)

    ----
    Returns:

        torch.Tensor

            Tensor with all images stacked.

            Shape will be: [len(indexes), num_channels, img_height, img_width]
    """

    return torch.stack([dataset[int(i)][0] for i in indexes])


def get_matched_pairs(
    args: ArgsMock,
    cuda: torch.device,
    train_dataset: DataLoader,
    domain_size: int,
    total_domains: int,
    training_list_size: list[int],
    phi: nn.Module,
    match_case: float,
    perfect_match: int,
    inferred_match: int,
):
    """ """
    logging.info(f"Total Domains {total_domains}")
    logging.info(f"Domain Size {domain_size}")
    logging.info(f"Training List Size {training_list_size}")

    # TODOs: Move the domain_data dictionary to data_loader class (saves time by
    # not computing it again and again)

    # Making Data Matched pairs
    domain_data = init_data_match_dict(
        args, range(total_domains), training_list_size, 1
    )

    # TODO: Make a common initialization
    data_matched = []
    for key in range(domain_size):
        temp = []
        for domain_idx in range(total_domains):
            temp.append([])
        data_matched.append(temp)

    perfect_match_rank = []

    domain_count = {}
    for domain in range(total_domains):
        domain_count[domain] = 0

    # Create dictionary: class label -> list of ordered indices
    if args.method_name == "hybrid" and args.match_func_aug_case == 0:
        for batch_idx, (x_e, _, y_e, d_e, idx_e, obj_e) in enumerate(train_dataset):
            x_e = x_e
            y_e = torch.argmax(y_e, dim=1)
            d_e = torch.argmax(d_e, dim=1).numpy()

            # Changed to have both image and domain indexes
            data_idx = idx_e[:, 0]  # image index
            idx_e = idx_e[:, 1]  # domain index

            domain_indices = np.unique(d_e)
            for domain_idx in domain_indices:

                indices = d_e == domain_idx
                ordered_indices = idx_e[indices]

                for idx in range(ordered_indices.shape[0]):
                    # Matching points across domains
                    perfect_indice = ordered_indices[idx].item()

                    # Changed to not need to store it into memory, save idx instead
                    # domain_data[domain_idx]["data"][
                    #   perfect_indice
                    # ] = x_e[indices][idx]
                    img_index = data_idx[indices][idx]
                    domain_data[domain_idx]["data_idx"][perfect_indice] = img_index

                    domain_data[domain_idx]["label"][perfect_indice] = y_e[indices][idx]
                    domain_data[domain_idx]["idx"][perfect_indice] = idx_e[indices][idx]
                    domain_data[domain_idx]["obj"][perfect_indice] = obj_e[indices][idx]
                    domain_count[domain_idx] += 1
    else:
        for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(train_dataset):
            x_e = x_e
            y_e = torch.argmax(y_e, dim=1)
            d_e = torch.argmax(d_e, dim=1).numpy()

            # Changed to have both image and domain indexes
            data_idx = idx_e[:, 0]  # image index
            idx_e = idx_e[:, 1]  # domain index

            domain_indices = np.unique(d_e)
            for domain_idx in domain_indices:

                indices = d_e == domain_idx
                ordered_indices = idx_e[indices]

                for idx in range(ordered_indices.shape[0]):
                    # Matching points across domains
                    perfect_indice = ordered_indices[idx].item()

                    # Changed to not need to store it into memory, save idx instead
                    # domain_data[domain_idx]["data"][
                    #   perfect_indice
                    # ] = x_e[indices][idx]
                    img_index = data_idx[indices][idx]
                    domain_data[domain_idx]["data_idx"][perfect_indice] = img_index

                    domain_data[domain_idx]["label"][perfect_indice] = y_e[indices][idx]
                    domain_data[domain_idx]["idx"][perfect_indice] = idx_e[indices][idx]
                    domain_data[domain_idx]["obj"][perfect_indice] = obj_e[indices][idx]
                    domain_count[domain_idx] += 1

    # Sanity Check: To check if the domain_data was updated for all the data points
    for domain in range(total_domains):
        if domain_count[domain] != training_list_size[domain]:
            logging.warning(
                "Issue: Some data points are missing from domain_data dictionary"
            )

    # Creating the random permutation tensor for each domain
    # TODO: Perm Prob might become 2.0 in case of matchdg_erm, handle that case
    if match_case == -1:
        perm_prob = 1.0
    else:
        perm_prob = 1.0 - match_case
    logging.info(f"Perm prob:  {perm_prob}")
    total_matches_per_point = args.total_matches_per_point

    # Determine the base_domain_idx as the one with the max samples of current class
    base_domain_dict = {}
    for y_c in range(args.out_classes):
        base_domain_size = 0
        base_domain_idx = -1
        for domain_idx in range(total_domains):
            class_idx = domain_data[domain_idx]["label"] == y_c
            curr_size = domain_data[domain_idx]["label"][class_idx].shape[0]
            if base_domain_size < curr_size:
                base_domain_size = curr_size
                base_domain_idx = domain_idx

        base_domain_dict[y_c] = base_domain_idx
        logging.info(f"Base Domain:  {base_domain_size, base_domain_idx, y_c}")

    # Finding the match
    for domain_idx in range(total_domains):

        total_data_idx = 0
        perf_match_mistakes = 0

        for y_c in range(args.out_classes):

            #             logging.info(domain_idx, y_c)
            if base_domain_dict[y_c] < 0:
                continue

            base_domain_idx = base_domain_dict[y_c]
            indices_base = domain_data[base_domain_idx]["label"] == y_c
            indices_base = indices_base[:, 0]
            ordered_base_indices = domain_data[base_domain_idx]["idx"][indices_base]
            obj_base = domain_data[base_domain_idx]["obj"][indices_base]

            if domain_idx == base_domain_idx:
                #                 logging.info('base domain idx')
                # Then its simple, the index if same as ordered-base-indice
                for idx in range(ordered_base_indices.shape[0]):
                    perfect_indice = ordered_base_indices[idx].item()
                    data_matched[total_data_idx][domain_idx].append(perfect_indice)
                    total_data_idx += 1
                continue

            indices_curr = domain_data[domain_idx]["label"] == y_c
            indices_curr = indices_curr[:, 0]
            ordered_curr_indices = domain_data[domain_idx]["idx"][indices_curr]
            obj_curr = domain_data[domain_idx]["obj"][indices_curr]
            curr_size = ordered_curr_indices.shape[0]

            if inferred_match == 1:

                # Changed to indexes
                # base_feat_data = domain_data[base_domain_idx]["data"][indices_base]

                base_feat_data = domain_data[base_domain_idx]["data_idx"][indices_base]
                base_feat_data_split = torch.split(
                    base_feat_data, args.batch_size, dim=0
                )
                base_feat = []

                # Changed to only load images when needed
                for batch_feat in base_feat_data_split:

                    if batch_feat.numel() == 0:
                        continue

                    with torch.no_grad():
                        imgs_feat = load_images_from_indexes(
                            train_dataset.dataset, batch_feat
                        ).to(cuda)
                        out = phi(imgs_feat)
                        base_feat.append(out.cpu())

                    del imgs_feat

                base_feat = torch.cat(base_feat)

                # Changed to indexes
                # feat_x_data = domain_data[domain_idx]["data"][indices_curr]

                feat_x_data = domain_data[domain_idx]["data_idx"][indices_curr]
                feat_x_data_split = torch.split(feat_x_data, args.batch_size, dim=0)
                feat_x = []

                # Changed to only load images when needed
                for batch_feat in feat_x_data_split:

                    if batch_feat.numel() == 0:
                        continue

                    with torch.no_grad():
                        imgs_feat = load_images_from_indexes(
                            train_dataset.dataset, batch_feat
                        ).to(cuda)
                        out = phi(imgs_feat)
                        feat_x.append(out.cpu())

                    del imgs_feat

                # Changed to accept empty feat_x
                feat_x = torch.cat(feat_x) if len(feat_x) else None
                size_base = ordered_base_indices.shape[0]

                for idx in range(size_base):

                    sort_idx = list(range(feat_x_data.shape[0]))
                    np.random.shuffle(sort_idx)

                    if feat_x is not None:
                        ws_dist = torch.sum((feat_x - base_feat[idx]) ** 2, dim=1)
                        sort_val, sort_idx = torch.sort(ws_dist, dim=0)
                        del ws_dist

                    perfect_indice = ordered_base_indices[idx].item()
                    curr_indices = ordered_curr_indices[
                        sort_idx[:total_matches_per_point]
                    ]

                    for _, curr_indice in enumerate(curr_indices):
                        data_matched[total_data_idx][domain_idx].append(
                            curr_indice.item()
                        )

                    total_data_idx += 1

                    if perfect_match == 1:
                        # Find all instances among the curr_domain with same object
                        # as obj_base[idx]

                        # .nonzero() converts True matches to match indexes; [0, 0]
                        # takes into the first match of same base object in the
                        # curr domain

                        if obj_base[idx] in obj_curr[sort_idx]:
                            perfect_match_rank.append(
                                (obj_curr[sort_idx] == obj_base[idx])
                                .nonzero()[0, 0]
                                .item()
                            )
            #       logging.info('Time Taken in CTR Loop: ', time.time()-start_time)

            elif inferred_match == 0 and perfect_match == 1:

                rand_vars = bernoulli.rvs(perm_prob, size=ordered_base_indices.shape[0])

                for idx in range(ordered_base_indices.shape[0]):
                    perfect_indice = ordered_base_indices[idx].item()

                    # Select random matches with perm_prob probability
                    if rand_vars[idx]:

                        rand_indices = np.arange(ordered_curr_indices.size()[0])
                        np.random.shuffle(rand_indices)
                        curr_indices = ordered_curr_indices[rand_indices][
                            :total_matches_per_point
                        ]
                        for _, curr_indice in enumerate(curr_indices):
                            data_matched[total_data_idx][domain_idx].append(
                                curr_indice.item()
                            )
                            if curr_indice.item() != perfect_indice:
                                perf_match_mistakes += 1

                    # Sample perfect matches
                    else:
                        base_object = obj_base[idx]
                        match_obj_indices = obj_curr == base_object
                        curr_indices = ordered_curr_indices[match_obj_indices]
                        for _, curr_indice in enumerate(curr_indices):
                            data_matched[total_data_idx][domain_idx].append(
                                curr_indice.item()
                            )
                            if curr_indice.item() != perfect_indice:
                                perf_match_mistakes += 1

                    total_data_idx += 1

            elif inferred_match == 0 and perfect_match == 0:

                for idx in range(ordered_base_indices.shape[0]):
                    perfect_indice = ordered_base_indices[idx].item()
                    rand_indices = np.arange(ordered_curr_indices.size()[0])
                    np.random.shuffle(rand_indices)
                    curr_indices = ordered_curr_indices[rand_indices][
                        :total_matches_per_point
                    ]
                    for _, curr_indice in enumerate(curr_indices):
                        data_matched[total_data_idx][domain_idx].append(
                            curr_indice.item()
                        )
                    total_data_idx += 1

        logging.info(f"Perfect Match Mistakes:  {perf_match_mistakes}")

        if total_data_idx != domain_size:
            logging.warning(
                "Issue: Some data points left from data_matched dictionary",
                total_data_idx,
                domain_size,
            )

    # Sanity Check:  N keys; K vals per key
    for idx in range(len(data_matched)):
        if len(data_matched[idx]) != total_domains:
            logging.warning("Issue with data matching")

    if inferred_match:
        logging.info(f"Perfect Match rank: {np.mean(np.array(perfect_match_rank))}")

    logging.info(f"Data matches lenght: {len(data_matched)}")
    return data_matched, domain_data, perfect_match_rank
