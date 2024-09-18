import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# check cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# set random seed
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    if device != "cpu":
        torch.cuda.random.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        scheduler=None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if scheduler is not None:
            self.scheduler = scheduler

    def fit(self, epochs, train_loader: DataLoader, val_loader=None) -> nn.Module:
        for epoch in range(epochs):
            train_loss = 0.0
            self.model.train()
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x, y = x.to(device), y.to(device)

                # forward propagation
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)

                # backward propagation
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)

            msg = f"Epoch [{epoch + 1}/{epochs}] train loss: {train_loss}"
            if val_loader is not None:
                val_loss = self.validation(val_loader)
                msg += f", validation loss: {val_loss}"
            print(msg)
        return self.model

    def validation(self, val_loader: DataLoader):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # forward propagation
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        return val_loss
