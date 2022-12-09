from dataclasses import dataclass, field


@dataclass
class ArgsMock:

    """
    Class to mock args parameters.
    Effectively, this works as a configuration class.

    It is required to run the code in a notebook,
    rather than through shell commands.

    -------
    General Parameters:

        out_classes: int = 10

            Number of classification labels.

        img_c: int = 1

            Number of image channels.

        img_h: int = 224

            Image height.

        img_w: int = 224

            Image width.

        epochs: int = 15

            Number of epochs.

        batch_size: int = 16

            Training batch-size.

    -----
    Model Parameters:

        <ctr_>model_name: str = "resnet18"

            Neural network architecture used.

            There is a ctr_model_name for contrastive loss in MatchDG.

        opt: str = "sgd"

            Optimizer used.

        lr: float = 0.01

            Learning rate.

        weight_decay: float = 5e-4

            Weight decay for optimizer.

    -----
    Algorithm Parameters:

        method_name: str = "erm_match"

            Algorithm used.

            Given the modification made, only changes some file names.

        <ctr_>match_flag: int = 0

            0 -> no update to match strategy

            1 -> update to match strategy

            There is a ctr_match_flag for contrastive loss in MatchDG.

        <ctr_>match_case: float = 1.0

            0 -> Random Match

            1 -> Perfect Match

            There is a ctr_match_case for contrastive loss in MatchDG.

        <ctr_>match_interrupt: int = 5

            Number of epochs before inferring the match strategy

            There is a ctr_match_interrupt for contrastive loss in MatchDG.

        penalty_ws: float = 0.1

            Penalty weight for Matching Loss

        penalty_diff_ctr: float = 1.0

            Penalty weight for Contrastive Loss

        tau: float = 0.05

            Temperature hyper param for NTXent contrastive loss.

    ----
    Other parameters:

        pos_metric: str = "l2"

            Cost to function to evaluate distance between two representations.

            Options: l1, l2, cos

        <ctr_>match_layer: str = "logit_match"

            "rep_match" -> Matching at an intermediate representation level.

            "logit_match" -> Matching at the logit level.

            There is a ctr_match_layer for contrastive loss in MatchDG.

        perfect_match: int = 1

            0 -> No perfect match known.

            1 -> Perfect match known.

        rep_dim: int = 250

            Representation dimension for contrastive learning.

        penalty_s: int = -1

            Epoch threshold over which Matching Loss to be optimized

        retain: int = 0

            Whether or not should train from scratch in MatchDG phase 2.

            0 -> train from scratch.

            2 -> fine-tune from phase 1

        match_func_aug_case: int = 0

            0 -> Evaluate match func on train domains

            1 -> Evaluate match func on self augmentations

        match_func_data_case: str = "val"

            MatchDG contrastive loss validation dataset.

    """

    # ------------------------------
    # GENERAL PARAMETERS
    # ------------------------------

    out_classes: int = 10
    img_c: int = 1
    img_h: int = 224
    img_w: int = 224

    batch_size: int = 16
    epochs: int = 15

    # ------------------------------
    # MODEL SPECIFIC
    # ------------------------------

    model_name: str = "resnet18"
    opt: str = "sgd"
    lr: float = 0.01
    weight_decay: float = 5e-4

    # ------------------------------
    # ALGORITHM SPECIFIC
    # ------------------------------

    method_name: str = "erm_match"
    pos_metric: str = "l2"

    match_flag: int = 0
    match_case: float = 1.0
    match_interrupt: int = 5

    penalty_ws: float = 0.1
    penalty_diff_ctr: float = 1.0

    tau: float = 0.05

    # Contrastive Parameters
    ctr_model_name: str = "resnet18"
    ctr_match_layer: str = "logit_match"
    ctr_match_flag: int = 1
    ctr_match_case: float = 0.01
    ctr_match_interrupt: int = 5

    # ------------------------------
    # OTHER PARAMETERS
    # ------------------------------

    dataset_name: str = "skin-cancer"  # it was "rot_mnist"
    perfect_match: int = 1  # in our case it does exist

    retain: float = 0
    match_layer: str = "logit_match"
    rep_dim: int = 250

    #    Evaluation specific
    test_metric: str = "acc"  # it was "match_score"
    acc_data_case: str = "test"
    top_k: int = 10
    match_func_aug_case: int = 0
    match_func_data_case: str = "val"

    #   Penalties
    penalty_s: int = -1
    penalty_irm: float = 0.0  # we don't use irm
    penalty_aug: float = 1.0

    #   Not sure what it does
    ctr_abl: int = 0
    match_abl: int = 0

    # ------------------------------
    # UNNECESSARY AFTER CHANGES
    # ------------------------------

    # Everything is run locally, so paths are absolute
    os_env: int = 0

    # Device is defined in notebook
    cuda_device: int = 0

    # Model is passed as a parameter in robustdg modified
    fc_layer: int = 1
    pre_trained: int = 0

    # Only running once anyway
    n_runs: int = 3
    n_runs_matchdg_erm: int = 1

    # ------------------------------
    # SPECIFIC FOR ORIGINAL PAPER
    # ------------------------------

    # MNIST-specific parameters
    mnist_aug: int = 0
    mnist_seed: int = 0

    train_domains: list[str] = field(
        default_factory=lambda: ["15", "30", "45", "60", "75"]
    )
    test_domains: list[str] = field(default_factory=lambda: ["0", "90"])
    mnist_case: str = (
        "resnet18"  # Differentiate between resnet, lenet, domainbed cases of mnist
    )

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

    # Multiple random matches
    total_matches_per_point: int = 1
