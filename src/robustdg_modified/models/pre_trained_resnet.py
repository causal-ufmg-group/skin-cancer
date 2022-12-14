from torch import Tensor, nn
from torchvision import models

from .utils import set_requires_grad_for_all_parameters


class PreTrainedResNet18(nn.Module):

    """
    Pre-trained ResNet18.

    For more information see:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    def __init__(self, num_classes: int) -> None:

        super().__init__()

        pre_trained = models.resnet18(weights="DEFAULT")

        set_requires_grad_for_all_parameters(pre_trained, value=False)

        # fine-tune last convolutional layer
        pre_trained.layer4.requires_grad_(True)

        # fine-tune last fc layer with correct number of classes
        in_features_fc = pre_trained.fc.in_features
        pre_trained.fc = nn.Linear(in_features_fc, num_classes)

        self.model = pre_trained

    def forward(self, x: Tensor) -> Tensor:

        x = self.model(x)

        return nn.functional.softmax(x, dim=1)
