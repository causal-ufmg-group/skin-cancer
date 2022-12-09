import torch.nn as nn
from torch import Tensor


class ConvNetwork(nn.Module):

    """
    Neural network architecture.
    """

    def __init__(self, num_classes: int, dropout_rate: float) -> None:

        super().__init__()

        self.first_conv: nn.Sequential = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # using default MaxPool parameter from keras
            nn.MaxPool2d(kernel_size=2),
        )

        self.second_conv: nn.Sequential = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2),
        )

        self.third_conv: nn.Sequential = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fully_connected: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            # using default BatchNorm parameters from keras
            nn.LazyBatchNorm1d(eps=0.001, momentum=0.99),
            nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:

        x: Tensor = self.first_conv(x)
        x: Tensor = self.second_conv(x)
        x: Tensor = self.third_conv(x)
        x: Tensor = self.fully_connected(x)

        return x
