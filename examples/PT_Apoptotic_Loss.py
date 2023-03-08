import gzip
import pickle
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
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=200, shuffle=True)
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, batch_size=200)

""" Create a LeNet-5 model """
# Set the random seed for reproducibility
torch.manual_seed(0)


# Build the layers
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Unit 1
        self.conv_1 = nn.Conv2d(1, 3, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_1.weight)
        torch.nn.init.zeros_(self.conv_1.bias)
        self.elu_1 = nn.ELU()
        self.norm_1 = nn.BatchNorm2d(3, eps=0.00001, momentum=0.9)
        self.maxp_1 = nn.MaxPool2d(2, stride=2)

        # Unit 2
        self.conv_2 = nn.Conv2d(3, 9, kernel_size=5, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_2.weight)
        torch.nn.init.zeros_(self.conv_2.bias)
        self.elu_2 = nn.ELU()
        self.norm_2 = nn.BatchNorm2d(9, eps=0.00001, momentum=0.9)
        self.maxp_2 = nn.MaxPool2d(2, stride=2)

        # Fully Connected
        self.conv_3 = nn.Conv2d(9, 4, kernel_size=4, stride=1)
        torch.nn.init.kaiming_normal_(self.conv_3.weight)
        torch.nn.init.zeros_(self.conv_3.bias)
        self.elu_3 = nn.ELU()
        self.norm_3 = nn.BatchNorm2d(4, eps=0.00001, momentum=0.9)

        # Prediction
        self.flatten = nn.Flatten()
        self.pred = nn.Linear(4, 10)
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

""" Train the model, or load model if it already exists """
model_path = Path(".").joinpath("LeNet5")

num_epochs = 200
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True
)
last_valid_accuracy = 0
val_count = 0
patience = 5


def epoch_func(model, x, y, train=False):
    predictions = model(x)
    num = len(x)
    accuracy = (torch.argmax(predictions, axis=1) == y).float().sum() / num
    loss = loss_func(predictions, y)

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss, accuracy, num


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

        # if val_count >= patience:
        #     break

    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))

""" Create a function to calculate entropic loss from removing a neuron """


def apoptotic_loss(layer, neuron):
    # Get layer information
    counts = layer.counts()
    num_neurons = layer.max_entropy()

    # Delete a neuron and recompute states
    states = np.delete(layer.states, neuron, 1)
    compressed_states = ts.compress_states(states)

    # Determine new state counts
    edges, index = ts.sort_states(states, states.shape[0])
    new_counts = []
    for i in range(len(edges) - 1):
        new_counts.append(sum(counts[index[j]] for j in range(edges[i], edges[i + 1])))
    new_counts = np.array(new_counts, dtype=np.int64)

    return (layer.entropy(10) - ts.entropy(new_counts, alpha=10)) / (
        layer.entropy(0.1) - ts.entropy(new_counts, alpha=0.1)
    )


""" Evaluate model efficiency """
# Attach StateCapture layers to the model
efficiency_model = ts.build_efficiency_model(
    model, attach_to=["Conv2d"], method="after"
)

# Collect the states for each layer
print()
print("Running model predictions to capture states...")
start = time.time()
efficiency_model.eval()
with torch.no_grad():
    losses, accuracies, nums = zip(
        *[
            epoch_func(efficiency_model, xb.to(dev), yb.to(dev), False)
            for xb, yb in train_dl
        ]
    )
print(f"Finished in {time.time() - start:.3f}s!")

# fig, ax = plt.subplots(1,3,sharey=True)
# for i,layer in enumerate(efficiency_model.efficiency_layers):
#     ax[i].plot(list(range(len(layer.counts()))),np.flip(np.sort(layer.counts())))
#     ax[i].set_yscale('log')

# final_state = efficiency_model.efficiency_layers[-1].states
# plt.figure()
# plt.imshow(np.transpose(final_state.astype(np.uint8)),aspect='auto')

print()
for layer in efficiency_model.efficiency_layers:
    print(f"Layer {layer.name} unique states: {len(layer.counts())})")

for layer in efficiency_model.efficiency_layers:
    print()
    print(
        "Layer {} Efficiency: {:.2f}%".format(
            layer.name, 100 * layer.efficiency(alpha1=1, alpha2=None)
        )
    )
    print(f"Layer {layer.name} Entropy_4: {layer.entropy(alpha=10):.2f}%")
    print(f"Layer {layer.name} Entropy_1/4: {layer.entropy(alpha=0.1):.2f}%")
    for i in range(int(layer.max_entropy())):
        start = time.time()
        print(
            "Apoptotic Loss (#{}): {:.2f} ({:.3f}s)".format(
                i, apoptotic_loss(layer, i), time.time() - start
            )
        )

# plt.show()

# print()
# print('** All Classes **')
# for layer in efficiency_model.efficiency_layers:
#     print('Layer {} efficiency: {:.1f}%)'.format(layer.name,100*layer.efficiency()))

# # Calculate per class information
# Y = y_valid.numpy()
# X = x_valid.numpy()
# total_samples = 0
# for c in range(10):
#     print()
#     print('** Class {} **'.format(c))
#     ind = np.flatnonzero(Y==c)
#     print('# samples: {}'.format(ind.size))

#     ts.reset_efficiency_model(model)

#     class_ds = TensorDataset(x_valid[ind],y_valid[ind])
#     class_dl = DataLoader(class_ds,batch_size=200)

#     with torch.no_grad():
#         losses, accuracies, nums = zip(
#             *[epoch_func(efficiency_model,xb.to(dev), yb.to(dev), False) for xb, yb in class_dl]
#         )

#     for layer in efficiency_model.efficiency_layers:
#         print('Layer {} efficiency: {:.1f}%)'.format(layer.name,100*layer.efficiency()))
