###########################################################################
# To run this example with Distributed Data Parallel (DDP):
#   Use the following command to run on 2 GPUs (or more if available):
#
#   torchrun --standalone --nnodes=1 --nproc_per_node=2 examples/mnist.py
#
#   Explanation:
#   --standalone: Runs a single-node distributed job
#   --nnodes=1: Specifies that only one node is used
#   --nproc_per_node=2: Runs 2 processes, usually matching the number of GPUs
#
#   Adjust `nproc_per_node` based on the number of GPUs available on your machine.
###########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from ddp_trainer import Trainer


# Define Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    return train_dataset, test_dataset


def visualize_predictions(model, test_loader):
    device = next(model.parameters()).device
    data, _ = next(iter(test_loader))
    data = data[:16].to(device)

    model.eval()
    with torch.no_grad():
        output = model(data)
        preds = output.argmax(dim=1, keepdim=True).cpu().numpy()

    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    for i in range(16):
        x = i // 4
        y = i % 4
        axs[x][y].imshow(data[i].cpu().numpy().squeeze(), cmap="gray")
        axs[x][y].set_title(f"Pred: {preds[i][0]}")
        axs[x][y].axis("off")
    fig.suptitle("MNIST Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig("./mnist-pred.png")


if __name__ == "__main__":
    Trainer.init()

    # Hyperparameter
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # Load data
    train_dataset, test_dataset = load_dataset()
    train_loader = Trainer.get_loader(train_dataset, batch_size)
    test_loader = Trainer.get_loader(test_dataset, batch_size)

    # Create Trainer
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, loss_fn, optimizer)

    # Train
    model = trainer.fit(epochs, train_loader, test_loader)

    # Test
    if Trainer.rank == 0:
        visualize_predictions(model, test_loader)
