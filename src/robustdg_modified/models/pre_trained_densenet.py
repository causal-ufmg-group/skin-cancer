from torch import Tensor, nn
from torchvision import models

from .utils import set_requires_grad_for_all_parameters


class PreTrainedDenseNet121(nn.Module):

    """
    Pre-trained DenseNet121.

    For more information see:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    def __init__(self, num_classes: int) -> None:

        super().__init__()

        pre_trained = models.densenet121(weights="DEFAULT")

        set_requires_grad_for_all_parameters(pre_trained, value=False)

        # adding last fc layer with correct number of classes
        in_features_fc = pre_trained.classifier.in_features
        pre_trained.classifier = nn.Linear(in_features_fc, num_classes)

        self.model = pre_trained

    def forward(self, x: Tensor) -> Tensor:

        x = self.model(x)

        return nn.functional.softmax(x, dim=1)
