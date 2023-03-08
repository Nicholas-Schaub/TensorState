import time
from pathlib import Path

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
print(f"device_count = {torch.cuda.device_count()}")
print(f"device: {dev}")

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
        x = self.maxp_1(self.norm_1(self.elu_1(x)))
        x = self.conv_2(x)
        x = self.maxp_2(self.norm_2(self.elu_2(x)))
        x = self.conv_3(x)
        x = self.norm_3(self.elu_3(x))
        x = self.pred(self.flatten(x))
        return x.view(-1, x.size(1))


# Create the Keras model
model = LeNet5().to(dev)

""" Train the model """
num_epochs = 200
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True
)
last_valid_accuracy = 0
val_count = 0
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

    return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), num


for epoch in range(num_epochs):
    start = time.time()
    model.train()
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), True) for xb, yb in train_dl]
    )
    train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    train_accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)

    model.eval()
    with torch.no_grad():
        losses, accuracies, nums = zip(
            *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
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

""" Evaluate model efficiency """
# Attach StateCapture layers to the model
efficiency_model = ts.build_efficiency_model(
    model, attach_to=["Conv2d"], method="after"
)

# Collect the states for each layer
print()
print("Running model predictions to capture states...")
start = time.time()
model.eval()
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
    )
print(f"Finished in {time.time() - start:.3f}s!")

# Count the number of states in each layer
print()
print("Getting the number of states in each layer...")
for layer in efficiency_model.efficiency_layers:
    print(f"Layer {layer.name} number of states: {layer.state_count}")

# Calculate each layers efficiency
print()
print("Evaluating efficiency of each layer...")
for layer in efficiency_model.efficiency_layers:
    start = time.time()
    print(
        "Layer {} efficiency: {:.1f}% ({:.3f}s)".format(
            layer.name, 100 * layer.efficiency(), time.time() - start
        )
    )

# Calculate the aIQ
beta = 2  # fudge factor giving a slight bias toward accuracy over efficiency

print()
print("Network metrics...")
print(f"Beta: {beta}")

network_efficiency = ts.network_efficiency(efficiency_model)
print(f"Network efficiency: {100 * network_efficiency:.1f}%")

accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
print(f"Network accuracy: {100 * accuracy:.1f}%")

aIQ = ts.aIQ(network_efficiency, accuracy.cpu().item(), beta)
print(f"aIQ: {100 * aIQ:.1f}%")
