import sys

import torch


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return 1.0 - cos(x1, x2)


def l1_dist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    # Broadcasting
    if len(x1.shape) == len(x2.shape) - 1:
        x1 = x1.unsqueeze(1)
    if len(x2.shape) == len(x1.shape) - 1:
        x2 = x2.unsqueeze(1)

    if len(x1.shape) == 3 and len(x2.shape) == 3:
        # Tensor shapes: (N,1,D) and (N,K,D) so x1-x2 would result in (N,K,D)
        return torch.sum(torch.sum(torch.abs(x1 - x2), dim=2), dim=1)
    elif len(x1.shape) == 2 and len(x2.shape) == 2:
        return torch.sum(torch.abs(x1 - x2), dim=1)
    elif len(x1.shape) == 1 and len(x2.shape) == 1:
        return torch.sum(torch.abs(x1 - x2), dim=0)
    else:
        print("Error: Expect 1, 2 or 3 rank torch.tensors to compute L1 Norm")
        return


def l2_dist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    # Broadcasting
    if len(x1.shape) == len(x2.shape) - 1:
        x1 = x1.unsqueeze(1)
    if len(x2.shape) == len(x1.shape) - 1:
        x2 = x2.unsqueeze(1)

    if len(x1.shape) == 3 and len(x2.shape) == 3:
        # Tensor shapes: (N,1,D) and (N,K,D) so x1-x2 would result in (N,K,D)
        return torch.sum(torch.sum((x1 - x2) ** 2, dim=2), dim=1)
    elif len(x1.shape) == 2 and len(x2.shape) == 2:
        return torch.sum((x1 - x2) ** 2, dim=1)
    elif len(x1.shape) == 1 and len(x2.shape) == 1:
        return torch.sum((x1 - x2) ** 2, dim=0)
    else:
        print("Error: Expect 1, 2 or 3 rank torch.tensors to compute L2 Norm")
        return


def embedding_dist(
    x1: torch.Tensor,
    x2: torch.Tensor,
    pos_metric: str,
    tau: float = 0.05,
    xent: bool = False,
) -> torch.Tensor:

    if xent:
        # X1 denotes the batch of anchors while X2 denotes all the negative matches
        # Broadcasting to compute loss for each anchor over all the negative matches

        # Only implemnted if x1, x2 are 2 rank torch.tensors
        if len(x1.shape) != 2 or len(x2.shape) != 2:
            print(
                "Error: both should be rank 2 tensors for NT-Xent loss computation"
            )

        # Normalizing each vector
        #   Take care to reshape the norm: For a (N*D) vector; the norm would be (N)
        #   which needs to be shaped to (N,1) to ensure row wise l2 normalization
        #   takes place
        if torch.sum(torch.isnan(x1)):
            print("X1 is nan")
            sys.exit()

        if torch.sum(torch.isnan(x2)):
            print("X1 is nan")
            sys.exit()

        eps = 1e-8

        norm = x1.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x1 = x1 / torch.max(norm, temp)

        if torch.sum(torch.isnan(x1)):
            print("X1 Norm is nan")
            sys.exit()

        norm = x2.norm(dim=1)
        norm = norm.view(norm.shape[0], 1)
        temp = eps * torch.ones_like(norm)

        x2 = x2 / torch.max(norm, temp)

        if torch.sum(torch.isnan(x2)):
            print("Norm: ", norm, x2)
            print("X2 Norm is nan")
            sys.exit()

        # Boradcasting the anchors vector to compute loss over all negative matches
        x1 = x1.unsqueeze(1)
        cos_sim = torch.sum(x1 * x2, dim=2)
        cos_sim = cos_sim / tau

        if torch.sum(torch.isnan(cos_sim)):
            print("Cos is nan")
            sys.exit()

        loss = torch.sum(torch.exp(cos_sim), dim=1)

        if torch.sum(torch.isnan(loss)):
            print("Loss is nan")
            sys.exit()

        return loss

    else:
        if pos_metric == "l1":
            return l1_dist(x1, x2)
        elif pos_metric == "l2":
            return l2_dist(x1, x2)
        elif pos_metric == "cos":
            return cosine_similarity(x1, x2)
