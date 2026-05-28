# A Simple PyTorch Tutorial

## Introduction

The core ideas behind this package were originally described in our paper,
[Assessing Intelligence in Artificial Neural
Networks](https://arxiv.org/abs/2006.02909).

This package simplifies the capture of neural layer states, and provides
some utility functions to assist in analyzing the state space of neural
layers.

In this tutorial, we are going to build a classic convolutional neural
network, LeNet-5. Then we are going to use TensorState to evaluate the
network architecture, getting the efficiency of each layer and calculating
the artificial intelligence quotient.

## Build and train LeNet-5

### Get MNIST

To train this model, we need to get the MNIST data set. The data comes
already rescaled to floats ranging between 0–1, so no rescaling is required.

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set up the directories
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

# Download MNIST if it doesn't exist; ToTensor() rescales pixels into [0, 1].
train_ds = datasets.MNIST(PATH, train=True, download=True, transform=transforms.ToTensor())
valid_ds = datasets.MNIST(PATH, train=False, download=True, transform=transforms.ToTensor())

train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=200)
```

### Create a LeNet-5 model

The general structure of our LeNet-5 model will roughly follow the
structure of the original network described by
[LeCun et al.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
However, we are going to make a few modifications to make it more modern
relative to the original architecture, such as addition of L2 weight
regularization, exponential linear units, and batch normalization. So, we
start off by setting the PyTorch random seed (to make the results
reproducible). Then we build the LeNet-5 class to define our network.

LeNet-5 has 2 convolutional layers and 2 fully connected layers. We will use
max pooling layers after each convolutional layer, and we will add a batch
normalization layer to all but the last fully connected layer.

```python
import torch.nn as nn

# Set the random seed for reproducibility
torch.manual_seed(0)

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

# Set the device to run the model on (gpu if available, cpu otherwise)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Create the PyTorch model
model = LeNet5().to(dev)
```

### Train the LeNet-5 model

First, set up the parameters used for training. This will be set up to run
with early stopping, similar to how Keras has an early stopping callback.
The `patience` parameter determines how many epochs to let pass after the
highest accuracy value is observed.

```python
import torch.optim as optim

num_epochs = 200
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
)
last_valid_accuracy = 0
val_count = 0
patience = 5
```

Next, create the function to process training and evaluation for all samples.

```python
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
```

Finally, run the training and evaluation loop.

```python
import time
import numpy as np

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
        "Epoch {}/{} ({:.2f}s): TrainLoss={:.4f}, TrainAccuracy={:.2f}%, "
        "ValidLoss={:.4f}, ValidAccuracy={:.2f}%".format(
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
```

## Use TensorState to evaluate LeNet-5

To calculate neural layer efficiency, we need to capture the various states
each layer takes on as the network processes data. This functionality is
built into the `StateCaptureHook` class, which is a hook that can be called
before or after the designated layers to automate the capturing of
information passing through the network. The `StateCaptureHook` acts like a
probe that can be placed anywhere in the network: it records the
information without modifying it, and passes it on to subsequent layers.

While `StateCaptureHook`'s can be placed manually, `ts.attach` adds hooks
at the layers selected by a matcher. For example, we can attach a
`StateCaptureHook` to all convolutional layers.

```python
import TensorState as ts

efficiency_model = ts.attach(model, ts.match(types=nn.Conv2d), when="after")
```

In the above code, we feed the trained LeNet-5 model into `ts.attach`,
designate via `ts.match(types=nn.Conv2d)` that we want to attach
`StateCaptureHook`'s to all 2D convolutional layers, and we want to
capture the states `after` the layer. We could also capture the inputs
going into and out of the layer by using `when='both'`. The matcher is
composable (`|`, `&`, `~`) so you can target by type, name regex, or any
custom check. For more information on `attach` and additional settings,
please see the [TensorState reference](../reference/tensorstate.md).

!!! note
    The older `ts.build_efficiency_model(model, attach_to=["Conv2d"])`
    call still works and dispatches to `ts.attach` internally. New code
    should prefer `ts.attach`; `build_efficiency_model` will be removed
    in a future release.

Now that the `efficiency_model` has been created, the `StateCaptureHook`'s
will collect all states of the network as images are fed to the network.
Thus, to generate all possible states the network contains for the test
data, we only need to evaluate the test data. Then we can look at how many
states were collected for each layer.

```python
model.eval()
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
    )

for layer in ts.layers(efficiency_model).values():
    print("Layer {} number of states: {}".format(layer.name, layer.state_count))
```

`ts.layers(efficiency_model)` returns the attached probes keyed by layer
name. The output of the above code should look something like this:

```text
Layer conv_1_post_states number of states: 5760000
Layer conv_2_post_states number of states: 640000
Layer conv_3_post_states number of states: 10000
```

Since there are 10,000 images in the validation data set, it is expected
that the fully connected layer (`conv_3_post_states`) has 10,000 states
recorded, since exactly one state will be recorded per image. The other
layers are convolutional, generating multiple states per image. The number
of states can be checked by determining the number of locations the
convolutional operator is applied per image then multiplying by 10,000. For
example, in a 28×28 image with a 5×5 convolutional operation performed on
it, the dimensions of the output would be 24×24. Thus, the number of
states for all 10,000 images would be 24·24·10,000 = 5,760,000 states,
which is the number of states observed by `conv_1_post_states`.

!!! note
The `state_count` is the raw number of states observed, and there are
likely states that occur multiple times.

Now that the states of each layer have been captured, let's analyze the
state space using the efficiency metric originally described by
[Schaub et al.](https://arxiv.org/abs/2006.02909). The efficiency metric
calculates the entropy of the state space and divides by the number of
neurons in the layer, giving an efficiency value in the range 0.00–1.00.

```python
for layer in ts.layers(efficiency_model).values():
    layer_efficiency = layer.efficiency()
    print("Layer {} efficiency: {:.1f}%".format(layer.name, 100 * layer_efficiency))
```

Next, we can calculate the artificial intelligence quotient (aIQ). Since
things like neural network efficiency and aIQ are metrics calculated over
the entire network, the `StateCaptureHook`'s do not have built-in methods
to calculate these values.

```python
beta = 2  # fudge factor giving a slight bias toward accuracy over efficiency

print()
print("Network metrics...")
print("Beta: {}".format(beta))

network_efficiency = ts.network_efficiency(efficiency_model)
print("Network efficiency: {:.1f}%".format(100 * network_efficiency))

accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
print("Network accuracy: {:.1f}%".format(100 * accuracy))

aIQ = ts.aIQ(network_efficiency, accuracy.cpu().item(), beta)
print("aIQ: {:.1f}%".format(100 * aIQ))
```

## Complete example

```python
import time

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import TensorState as ts

# Set the device to run the model on (gpu if available, cpu otherwise)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

""" Load MNIST and transform it """
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
PATH.mkdir(parents=True, exist_ok=True)

train_ds = datasets.MNIST(PATH, train=True, download=True, transform=transforms.ToTensor())
valid_ds = datasets.MNIST(PATH, train=False, download=True, transform=transforms.ToTensor())

train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=200)

""" Create a LeNet-5 model """
torch.manual_seed(0)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_1.weight)
        torch.nn.init.zeros_(self.conv_1.bias)
        self.elu_1 = nn.ELU()
        self.norm_1 = nn.BatchNorm2d(20, eps=0.00001, momentum=0.9)
        self.maxp_1 = nn.MaxPool2d(2, stride=2)

        self.conv_2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_2.weight)
        torch.nn.init.zeros_(self.conv_2.bias)
        self.elu_2 = nn.ELU()
        self.norm_2 = nn.BatchNorm2d(50, eps=0.00001, momentum=0.9)
        self.maxp_2 = nn.MaxPool2d(2, stride=2)

        self.conv_3 = nn.Conv2d(50, 100, kernel_size=4, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_3.weight)
        torch.nn.init.zeros_(self.conv_3.bias)
        self.elu_3 = nn.ELU()
        self.norm_3 = nn.BatchNorm2d(100, eps=0.00001, momentum=0.9)

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

model = LeNet5().to(dev)

""" Train the model """
num_epochs = 200
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
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

    return loss, accuracy, num

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
        "Epoch {}/{} ({:.2f}s): TrainLoss={:.4f}, TrainAccuracy={:.2f}%, "
        "ValidLoss={:.4f}, ValidAccuracy={:.2f}%".format(
            str(epoch + 1).zfill(3),
            num_epochs,
            time.time() - start,
            train_loss,
            100 * train_accuracy,
            valid_loss,
            100 * valid_accuracy,
        )
    )

    if valid_accuracy > last_valid_accuracy:
        val_count = 0
        last_valid_accuracy = valid_accuracy
    else:
        val_count += 1

    if val_count >= patience:
        break

""" Evaluate model efficiency """
efficiency_model = ts.attach(model, ts.match(types=nn.Conv2d), when="after")

print()
print("Running model predictions to capture states...")
start = time.time()
model.eval()
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[epoch_func(xb.to(dev), yb.to(dev), False) for xb, yb in valid_dl]
    )
print("Finished in {:.3f}s!".format(time.time() - start))

print()
print("Getting the number of states in each layer...")
for layer in ts.layers(efficiency_model).values():
    print("Layer {} number of states: {}".format(layer.name, layer.state_count))

print()
print("Evaluating efficiency of each layer...")
for layer in ts.layers(efficiency_model).values():
    start = time.time()
    print(
        "Layer {} efficiency: {:.1f}% ({:.3f}s)".format(
            layer.name, 100 * layer.efficiency(), time.time() - start
        )
    )

beta = 2  # fudge factor giving a slight bias toward accuracy over efficiency

print()
print("Network metrics...")
print("Beta: {}".format(beta))

network_efficiency = ts.network_efficiency(efficiency_model)
print("Network efficiency: {:.1f}%".format(100 * network_efficiency))

accuracy = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
print("Network accuracy: {:.1f}%".format(100 * accuracy))

aIQ = ts.aIQ(network_efficiency, accuracy.cpu().item(), beta)
print("aIQ: {:.1f}%".format(100 * aIQ))
```
