import random

import numpy as np
import torch

# See https://pytorch.org/docs/stable/notes/randomness.html


def seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int, data_loader_generator: torch.Generator) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    data_loader_generator.manual_seed(seed)
