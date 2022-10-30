import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, argmax, optim
from torchmetrics.functional import accuracy


class NetworkModule(pl.LightningModule):

    """
    Pytorch-Lightning neural network module..

    See https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html
    for more information.
    """

    def __init__(
        self,
        model: nn.Module,
        channels: int,
        height: int,
        width: int,
        num_classes: int,
        learning_rate: float,
    ) -> None:

        super().__init__()

        self.model = model

        self.channels = channels
        self.height = height
        self.width = width

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.test_expected = []
        self.test_prediction = []

    def forward(self, x: Tensor) -> Tensor:

        x = self.model(x)

        return F.softmax(x, dim=1)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:

        x, y = batch
        logits = self(x)
        y_predictions = argmax(logits, dim=1)

        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, logger=True, on_epoch=True, prog_bar=True)

        acc = accuracy(y_predictions, y)
        self.log("train_acc", acc, logger=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:

        x, y = batch
        logits = self(x)
        y_predictions = argmax(logits, dim=1)

        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss, logger=True, on_epoch=True, prog_bar=True)

        acc = accuracy(y_predictions, y)
        self.log("val_acc", acc, logger=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:

        x, y = batch
        logits = self(x)
        y_predictions = argmax(logits, dim=1)

        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss, logger=True, on_epoch=True, prog_bar=True)

        acc = accuracy(y_predictions, y)
        self.log("test_acc", acc, logger=True, on_epoch=True, prog_bar=True)

        self.test_expected.append(y)
        self.test_prediction.append(y_predictions)

    def clear_test_predictions_variables(self) -> None:

        """
        Empties self.test_expected and self.test_prediction.
        """

        del self.test_expected
        del self.test_prediction

        self.test_expected = []
        self.test_prediction = []

    def configure_optimizers(self) -> optim.Adam:

        return optim.Adam(self.parameters(), lr=self.learning_rate)
