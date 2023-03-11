import lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10
from torchvision.models import MobileNetV2 as MV2
from torchvision.transforms import Compose, RandAugment, Resize, ToTensor

import TensorState as ts


class AccuracyCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.accuracy.reset()
        ts.reset_efficiency_model(pl_module)

    def on_validation_start(self, trainer, pl_module):
        entropy = sum(layer.entropy() for layer in pl_module.efficiency_layers)
        pl_module.log("entropy/train", entropy)
        pl_module.accuracy.reset()
        ts.reset_efficiency_model(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        entropy = sum(layer.entropy() for layer in pl_module.efficiency_layers)
        pl_module.log("entropy/val", entropy)


class MobileNetV2(pl.LightningModule, MV2):
    loss = torch.nn.CrossEntropyLoss()

    def __init__(self, num_classes=1000, **kwargs):
        """Stock prediction model"""
        super().__init__()

        model = ts.models.mobilenet_v2(num_classes=num_classes, **kwargs)

        self.accuracy = Accuracy("multiclass", num_classes=num_classes)

        self.features = model.features

        self.classifier = model.classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, target = batch

        z = self(data)

        # Main loss
        loss = self.loss(z, target)

        self.accuracy.update(z, target)

        self.log("loss/train", loss)
        self.log("accuracy/train", self.accuracy.compute())

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        data, target = batch

        z = self(data)

        # Main loss
        loss = self.loss(z, target)

        self.accuracy.update(z, target)

        self.log("loss/val", loss)
        self.log("accuracy/val", self.accuracy.compute())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    dev = "cuda"

    # sm = StockModel().to(dev)
    model = MobileNetV2(weights="IMAGENET1K_V1", num_classes=10)
    model = ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])

    # Create the augmentation transform
    compose = Compose(
        [
            Resize((64, 64)),
            RandAugment(),
            ToTensor(),
        ]
    )
    train_dl = DataLoader(
        CIFAR10(".data", transform=compose, train=True, download=True),
        batch_size=200,
        shuffle=True,
        num_workers=16,
        persistent_workers=True,
    )
    test_dl = DataLoader(
        CIFAR10(".data", transform=compose, train=False, download=True),
        batch_size=200,
        num_workers=16,
        persistent_workers=True,
    )

    for x, y in test_dl:
        model(x)
        break

    trainer: pl.Trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        # log_every_n_steps=1,
        # accumulate_grad_batches=25,
        callbacks=[AccuracyCallback()],
        val_check_interval=1.0,
    )
    # sm = torch.compile(sm)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=test_dl)
