import os
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


def set_env_variable_for_deterministic_algorithm():

    """
    - Operation uses CuBLAS and CUDA >= 10.2
    - To enable deterministic behavior, you must set an environment variable
      before running your PyTorch application:
        CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.

    For more information, go to
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
