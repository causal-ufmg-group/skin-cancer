from dataclasses import dataclass


@dataclass
class ArgsMock:

    # TODO: Write parameters.
    """
    Class to mock args parameters.
    Effectively, this works as a configuration class.

    It is required to run the code in a notebook,
    rather than through shell commands.

    -------
    Parameters:


    """

    method_name: str = "erm_match"
    match_layer: str = "logit_match"
    pos_metric: str = "l2"

    # probably necessary
    out_classes: int = 10
    img_c: int = 1
    img_h: int = 224
    img_w: int = 224

    rep_dim: int = 250
    perfect_match: int = 1

    penalty_s: int = -1
    penalty_irm: float = 0.0
    penalty_aug: float = 1.0
    penalty_ws: float = 0.1
    penalty_diff_ctr: float = 1.0

    tau: float = 0.05

    match_flag: int = 0
    match_case: float = 1.0
    match_interrupt: int = 5

    ctr_abl: int = 0
    match_abl: int = 0

    n_runs: int = 3
    n_runs_matchdg_erm: int = 1

    ctr_model_name: str = "resnet18"
    ctr_match_layer: str = "logit_match"
    ctr_match_flag: int = 1
    ctr_match_case: float = 0.01
    ctr_match_interrupt: int = 5

    retain: float = 0

    # ------------------------------
    # MAYBE UNNECESSARY PARAMETERS
    # ------------------------------

    lr: float = 0.01
    epochs: int = 15
    batch_size: int = 16
    fc_layer: int = 1

    model_name: str = "resnet18"
    pre_trained: int = 0
    opt: str = "sgd"
    weight_decay: float = 5e-4

    dataset_name: str = "rot_mnist"

    # ------------------------------
    # UNNECESSARY PARAMETERS
    # ------------------------------

    train_domains: list[str] = ["15", "30", "45", "60", "75"]
    test_domains: list[str] = ["0", "90"]

    mnist_seed: int = 0
    cuda_device: int = 0
    os_env: int = 0

    # Differential Privacy
    dp_noise: int = 0
    dp_epsilon: float = 1.0

    # Special case when you want to check results with the dp setting
    # for the infinite epsilon case
    dp_attach_opt: int = 1

    # MMD, DANN
    d_steps_per_g_step: int = 1
    grad_penalty: float = 0.0
    conditional: int = 1
    gaussian: int = 1

    # Slab Dataset
    slab_data_dim: int = 2
    slab_total_slabs: int = 7
    slab_num_samples: int = 1000
    slab_noise: float = 0.1

    # Differentiate between resnet, lenet, domainbed cases of mnist
    mnist_case: str = "resnet18"
    mnist_aug: int = 0

    # Multiple random matches
    total_matches_per_point: int = 1

    # Evaluation specific
    test_metric: str = "match_score"
    acc_data_case: str = "test"
    top_k: int = 10
    match_func_aug_case: int = 0
    match_func_data_case: str = "val"
