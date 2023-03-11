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
torch.manual_seed(0)


# Build the layers
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Unit 1
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_1.weight)
        torch.nn.init.zeros_(self.conv_1.bias)
        self.elu_1 = nn.ELU()
        self.norm_1 = nn.BatchNorm2d(8, eps=0.00001, momentum=0.9)
        self.maxp_1 = nn.MaxPool2d(2, stride=2)

        # Unit 2
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_2.weight)
        torch.nn.init.zeros_(self.conv_2.bias)
        self.elu_2 = nn.ELU()
        self.norm_2 = nn.BatchNorm2d(8, eps=0.00001, momentum=0.9)
        self.maxp_2 = nn.MaxPool2d(2, stride=2)

        # Fully Connected
        self.conv_3 = nn.Conv2d(8, 8, kernel_size=4, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_3.weight)
        torch.nn.init.zeros_(self.conv_3.bias)
        self.elu_3 = nn.ELU()
        self.norm_3 = nn.BatchNorm2d(8, eps=0.00001, momentum=0.9)

        # Prediction
        self.flatten = nn.Flatten()
        self.pred = nn.Linear(8, 10)
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


# Create the Keras model
model = LeNet5().to(dev)

""" Train the model, or load model if it already exists """
model_path = Path(".").joinpath("LeNet5")

num_epochs = 200
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True
)
last_valid_accuracy = 0
val_count = 0
patience = 10


def epoch_func(model, x, y, train=False):
    predictions = model(x)
    num = len(x)
    accuracy = (torch.argmax(predictions, axis=1) == y).float().sum() / num
    loss = loss_func(predictions, y)

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.detach().cpu(), accuracy.detach().cpu(), num


if not model_path.exists():
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        losses, accuracies, nums = zip(
            *[epoch_func(model, xb.to(dev), yb.to(dev), True) for xb, yb in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        train_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)

        model.eval()
        with torch.no_grad():
            losses, accuracies, nums = zip(
                *[
                    epoch_func(model, xb.to(dev), yb.to(dev), False)
                    for xb, yb in valid_dl
                ]
            )
        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        valid_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)

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
        if valid_accuracy > last_valid_accuracy:
            val_count = 0
            last_valid_accuracy = valid_accuracy
        else:
            val_count += 1

        if val_count >= patience:
            break

    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))

""" Capture state space for each class """
# Attach StateCapture layers to the model
efficiency_model = ts.build_efficiency_model(
    model, attach_to=["BatchNorm2d"], method="after"
)
efficiency_model.eval()

# Calculate per class information
Y = valid_ds.data.numpy()
X = valid_ds.targets.numpy()
total_samples = 0
state_dict = {}
for c in range(10):
    print()
    print(f"** Class {c} **")
    ind = np.argwhere(Y == c).squeeze()
    print(f"# samples: {ind.size}")

    class_dl = DataLoader(valid_ds, batch_size=200)

    with torch.no_grad():
        losses, accuracies, nums = zip(
            *[
                epoch_func(efficiency_model, xb.to(dev), yb.to(dev), False)
                for xb, yb in class_dl
            ]
        )

    class_dict = {}
    for layer in efficiency_model.efficiency_layers:
        print(f"Layer {layer.name} efficiency: {100 * layer.efficiency():.2f}%")

        # Store the states for each class
        start = time.time()
        class_dict[layer.name] = {
            state_id: count
            for state_id, count in zip(layer.state_ids(), layer.counts())
        }
        print(
            "Got {} state ids for layer {} in {:.3f}s...".format(
                len(class_dict[layer.name]), layer.name, time.time() - start
            )
        )

    state_dict[c] = class_dict

    ts.reset_efficiency_model(model)

""" Bin data and generate 2d histogram of states """

# Bin data for states 1 and 8
fig, ax = plt.subplots(3, 3)
for rind, c in enumerate([0, 2, 3]):
    for cind, layer_id in enumerate(state_dict[0].keys()):
        X = []
        Y = []
        keys = set(state_dict[1][layer_id].keys())
        keys.update(set(state_dict[c][layer_id].keys()))
        for key in keys:
            X.append(state_dict[1][layer_id].get(key, 0))
            Y.append(state_dict[c][layer_id].get(key, 0))
        ax[rind, cind].plot(X, Y, "rx")
        ax[rind, cind].set_xscale("symlog", linthreshx=10)
        ax[rind, cind].set_yscale("symlog", linthreshy=10)
        ax[rind, cind].set_title(layer_id)
        ax[rind, cind].set_xlabel("Class 1 Counts")
        ax[rind, cind].set_ylabel(f"Class {c} Counts")
        ax[rind, cind].set_ylim([0, max(Y)])
        ax[rind, cind].set_xlim([0, max(X)])
plt.tight_layout()
plt.show()
