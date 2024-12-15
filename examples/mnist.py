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

import ddp_trainer
from ddp_trainer import Trainer
from ddp_trainer.utils import evaluate


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


# Load mnist dataset
def load_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Initialize distributed processes
    ddp_trainer.init()

    # Hyperparameter
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # Load data
    train_dataset, test_dataset = load_dataset()
    train_loader = ddp_trainer.to_ddp_loader(train_dataset, batch_size)
    test_loader = ddp_trainer.to_ddp_loader(test_dataset, batch_size)

    # Create Trainer
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    eval_fn = evaluate.multi_class_accuracy
    trainer = Trainer(model, loss_fn, optimizer, eval_fn=eval_fn)

    # Train
    trainer.train(epochs, train_loader, test_loader)
