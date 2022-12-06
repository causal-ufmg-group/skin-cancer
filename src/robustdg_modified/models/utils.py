from typing import Iterator

from torch.nn import Module, Parameter


def set_requires_grad_for_all_parameters(model: Module, value: bool = False) -> None:

    """
    Set requires grad for all parameters of a given torch.nn.Module.

    ----
    Parameters:

        model: nn.Module

            Model with parameters to be set.

        value: bool

            Value that parameters should be set to.
    """

    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    for param in model.parameters():
        param.requires_grad = value


def find_parameters_to_be_trained(model: Module) -> Iterator[Parameter]:

    """
    Returns an iterator with all parameters to be updated in training.

    ----
    Parameters:

        model: torch.nn.Module (or subclass)

            Model with parameters to be checked.

    ----
    Returns

        Iterator[nn.Parameter]

            Parameter which should be updated.

            This iterator can be passed to torch.optim.Optimizer.
    """

    return (parameter for parameter in model.parameters() if parameter.requires_grad)
