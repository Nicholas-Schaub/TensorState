import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import TensorState as ts

# Set the device to run the model on (gpu if available, cpu otherwise)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

""" Load MNIST and transform it """
# Set up the directories
DATA_PATH = Path("data")
train_ds = MNIST(DATA_PATH, transform=ToTensor(), train=True, download=True)
valid_ds = MNIST(DATA_PATH, transform=ToTensor(), train=False, download=True)

train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=200)

""" Create a LeNet-5 model """
# Set the random seed for reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True


# Build the layers
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Unit 1
        self.conv_1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_1.weight)
        torch.nn.init.zeros_(self.conv_1.bias)
        self.elu_1 = nn.ELU()
        self.norm_1 = nn.BatchNorm2d(20, eps=0.00001, momentum=0.9)
        self.maxp_1 = nn.MaxPool2d(2, stride=2)

        # Unit 2
        self.conv_2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_2.weight)
        torch.nn.init.zeros_(self.conv_2.bias)
        self.elu_2 = nn.ELU()
        self.norm_2 = nn.BatchNorm2d(50, eps=0.00001, momentum=0.9)
        self.maxp_2 = nn.MaxPool2d(2, stride=2)

        # Fully Connected
        self.conv_3 = nn.Conv2d(50, 100, kernel_size=4, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_3.weight)
        torch.nn.init.zeros_(self.conv_3.bias)
        self.elu_3 = nn.ELU()
        self.norm_3 = nn.BatchNorm2d(100, eps=0.00001, momentum=0.9)

        # Prediction
        self.flatten = nn.Flatten()
        self.pred = nn.Linear(100, 10)
        torch.nn.init.kaiming_normal_(self.pred.weight)
        torch.nn.init.zeros_(self.pred.bias)

    def forward(self, data):
        x = data.view(-1, 1, 28, 28)
        x = self.conv_1(x)
        x = self.maxp_1(self.elu_1(self.norm_1(x)))
        x = self.conv_2(x)
        x = self.maxp_2(self.elu_2(self.norm_2(x)))
        x = self.conv_3(x)
        x = self.elu_3(self.norm_3(x))
        x = self.pred(self.flatten(x))
        return x.view(-1, x.size(1))


# Create the Keras model, attach efficiency layers
model = LeNet5().to(dev)
model.eval()
model = ts.build_efficiency_model(model, attach_to=["Conv2d"], method="after")

""" Train the model """
num_epochs = 100
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True
)
last_valid_accuracy = 0
last_valid_eff = np.inf
val_count = 0
eff_val_count = 0
patience = 5


def epoch_func(x, y, train=False):
    predictions = model(x)
    num = len(x)
    accuracy = (torch.argmax(predictions, axis=1) == y).float().sum() / num
    loss = loss_func(predictions, y)

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss, accuracy, num


efficiencies = []
early_stop_epoch = 0
eff_stop_epoch = 0
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), True) for xb, yb in train_dl]
    )

    train_loss = sum(loss * n for loss, n in zip(losses, nums)) / sum(nums)
    train_accuracy = sum(acc * n for acc, n in zip(accuracies, nums)) / sum(nums)
    te_start = time.time()
    efficiencies.append(
        {
            "train": {
                "loss": train_loss.cpu().item(),
                "accuracy": train_accuracy.cpu().item(),
                "efficiency": [e.efficiency() for e in model.efficiency_layers],
                "net_efficiency": ts.network_efficiency(model),
            },
            "test": {},
        }
    )
    te = time.time() - te_start
    # for e in model.efficiency_layers:
    #     print(e._raw_states.info)
    ts.reset_efficiency_model(model)

    model.eval()
    with torch.no_grad():
        losses, accuracies, nums = zip(
            *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
        )

    valid_loss = sum(loss * n for loss, n in zip(losses, nums)) / sum(nums)
    valid_accuracy = sum(acc * n for acc, n in zip(accuracies, nums)) / sum(nums)
    ve_start = time.time()
    efficiencies[-1]["test"]["loss"] = valid_loss.cpu().item()
    efficiencies[-1]["test"]["accuracy"] = valid_accuracy.cpu().item()
    efficiencies[-1]["test"]["efficiency"] = [
        e.efficiency() for e in model.efficiency_layers
    ]
    efficiencies[-1]["test"]["net_efficiency"] = ts.network_efficiency(model)
    ve = time.time() - ve_start

    print(
        "Epoch {}/{} ({:.2f}s,{:.2f}s,{:.2f}s): TrainLoss={:.4f}, TrainAccuracy={:.2f}%, ValidLoss={:.4f}, ValidAccuracy={:.2f}%".format(
            str(epoch + 1).zfill(3),
            num_epochs,
            time.time() - start,
            te,
            ve,
            train_loss,
            100 * train_accuracy,
            valid_loss,
            100 * valid_accuracy,
        )
    )

    # Early stopping criteria
    if not early_stop_epoch and valid_accuracy > last_valid_accuracy:
        val_count = 0
        last_valid_accuracy = valid_accuracy
    else:
        val_count += 1

    if (
        not eff_stop_epoch
        and efficiencies[-1]["train"]["net_efficiency"] < last_valid_eff
    ):
        eff_val_count = 0
        last_valid_eff = efficiencies[-1]["train"]["net_efficiency"]
    else:
        eff_val_count += 1

    if eff_val_count >= patience and not eff_stop_epoch:
        eff_stop_epoch = epoch

    if val_count >= patience and not early_stop_epoch:
        early_stop_epoch = epoch
        break

    ts.reset_efficiency_model(model)

""" Evaluate model efficiency """
fig, ax = plt.subplots(1, 3)
X = list(range(early_stop_epoch + 1))
ax[0].plot(
    X, [d["train"]["accuracy"] for d in efficiencies], "b-", label="TrainAccuracy"
)
ax[0].plot(X, [d["test"]["accuracy"] for d in efficiencies], "r-", label="TestAccuracy")
ax[0].plot(
    [early_stop_epoch - 5],
    efficiencies[early_stop_epoch - 5]["test"]["accuracy"],
    "ko",
    label="AccuracyEarlyStop",
)
ax[0].plot(
    [eff_stop_epoch - 5],
    efficiencies[eff_stop_epoch - 5]["test"]["accuracy"],
    "kx",
    label="EfficiencyEarlyStop",
)
ax[0].legend()


ax[1].plot(
    X, [d["train"]["efficiency"][0] for d in efficiencies], "b-", label="TrainEff_1"
)
ax[1].plot(
    X, [d["train"]["efficiency"][1] for d in efficiencies], "g-", label="TrainEff_2"
)
ax[1].plot(
    X, [d["train"]["efficiency"][2] for d in efficiencies], "r-", label="TrainEff_3"
)
ax[1].plot(
    X,
    [d["train"]["net_efficiency"] for d in efficiencies],
    "m-",
    label="TrainNetEfficiency",
)
ax[1].plot(
    [early_stop_epoch - 5],
    efficiencies[early_stop_epoch - 5]["train"]["net_efficiency"],
    "ko",
    label="AccuracyEarlyStop",
)
ax[1].plot(
    [eff_stop_epoch - 5],
    efficiencies[eff_stop_epoch - 5]["train"]["net_efficiency"],
    "kx",
    label="EfficiencyEarlyStop",
)
ax[1].legend()

ax[2].plot(
    X, [d["test"]["efficiency"][0] for d in efficiencies], "b-", label="TestEff_1"
)
ax[2].plot(
    X, [d["test"]["efficiency"][1] for d in efficiencies], "g-", label="TestEff_2"
)
ax[2].plot(
    X, [d["test"]["efficiency"][2] for d in efficiencies], "r-", label="TestEff_3"
)
ax[2].plot(
    X,
    [d["test"]["net_efficiency"] for d in efficiencies],
    "m-",
    label="TestNetEfficiency",
)
ax[2].plot(
    [early_stop_epoch - 5],
    efficiencies[early_stop_epoch - 5]["test"]["net_efficiency"],
    "ko",
    label="AccuracyEarlyStop",
)
ax[2].plot(
    [eff_stop_epoch - 5],
    efficiencies[eff_stop_epoch - 5]["test"]["net_efficiency"],
    "kx",
    label="EfficiencyEarlyStop",
)
ax[2].legend()

plt.show()
