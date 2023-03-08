import gzip
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import TensorState as ts

# Set the device to run the model on (gpu if available, cpu otherwise)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

""" Load MNIST and transform it """
# Set up the directories
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

# Download the data if it doesn't exist
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# Load the data
train_rand = 0.1  # percentage of training data to randomize
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    y_train[: int(50000 * train_rand)] = np.random.randint(
        10, size=(int(50000 * train_rand),)
    )

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
    valid_ds = TensorDataset(x_valid, y_valid)
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
num_epochs = 1000
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
eff_model = Path(".").joinpath("EffLeNet5.ptm")
acc_model = Path(".").joinpath("AccLeNet5.ptm")
eff_stop_epoch = 0
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), True) for xb, yb in train_dl]
    )

    train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    train_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
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
    ts.reset_efficiency_model(model)

    model.eval()
    with torch.no_grad():
        losses, accuracies, nums = zip(
            *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
        )

    valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    valid_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
    efficiencies[-1]["test"]["loss"] = valid_loss.cpu().item()
    efficiencies[-1]["test"]["accuracy"] = valid_accuracy.cpu().item()
    efficiencies[-1]["test"]["efficiency"] = [
        e.efficiency() for e in model.efficiency_layers
    ]
    efficiencies[-1]["test"]["net_efficiency"] = ts.network_efficiency(model)

    print(
        "Epoch {}/{} ({:.2f}s): TrainLoss={:.4f}, TrainAccuracy={:.2f}%, ValidLoss={:.4f}, ValidAccuracy={:.2f}%".format(
            str(epoch + 1).zfill(3),
            num_epochs,
            time.time() - start,
            train_loss,
            100 * train_accuracy,
            valid_loss,
            100 * valid_accuracy,
        )
    )

    # Early stopping criteria
    if not early_stop_epoch and train_accuracy > last_valid_accuracy:
        val_count = 0
        last_valid_accuracy = train_accuracy
        torch.save(model.state_dict(), str(acc_model))
    else:
        val_count += 1

    if (
        not eff_stop_epoch
        and efficiencies[-1]["train"]["net_efficiency"] < last_valid_eff
    ):
        eff_val_count = 0
        last_valid_eff = efficiencies[-1]["train"]["net_efficiency"]
        torch.save(model.state_dict(), str(eff_model))
    else:
        eff_val_count += 1

    if eff_val_count >= patience and not eff_stop_epoch:
        eff_stop_epoch = epoch

    if val_count >= patience and not early_stop_epoch:
        early_stop_epoch = epoch

    if early_stop_epoch and eff_stop_epoch:
        break

    ts.reset_efficiency_model(model)

""" Evaluate models on training data """

# Reload the data so it's not randomized anymore
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=200)

# Load the efficiency model and evaluate
model.load_state_dict(torch.load(str(eff_model)))
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in train_dl]
    )
train_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
print(f"Efficiency Stopping Epoch: {eff_stop_epoch}")
print(f"Efficiency Stopping Criteria Accuracy: {100 * train_accuracy:.2f}")

# Load the accuracy model and evaluate
model.load_state_dict(torch.load(str(acc_model)))
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in train_dl]
    )
train_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
print(f"Accuracy Stopping Epoch: {early_stop_epoch}")
print(f"Accuracy Stopping Criteria Accuracy: {100 * train_accuracy:.2f}")
