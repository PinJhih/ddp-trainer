import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


# set random seed
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)


def get_ddp_loader(ds: Dataset, batch_size, num_workers=4, seed=42):
    sampler = DistributedSampler(ds, shuffle=True, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    return loader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        scheduler=None,
    ) -> None:
        # Init distributed
        if "RANK" in os.environ:
            self.is_distributed = True
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        else:
            self.is_distributed = False
            self.rank = 0
            self.local_rank = 0
            self.world_size = 0
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, epochs, train_loader: DataLoader, val_loader=None) -> nn.Module:
        progress = range(epochs)
        if self.local_rank == 0:
            progress = tqdm(progress, desc="Train")

        for _ in progress:
            train_loss = self.train(train_loader)
            loss = {"train_loss": train_loss}

            if val_loader is not None:
                val_loss = self.validation(val_loader)
                loss["val_loss"] = val_loss

            if self.local_rank == 0:
                progress.set_postfix(loss)

            if self.scheduler is not None:
                self.scheduler.step()
        return self.model

    def train(self, train_loader: DataLoader):
        train_loss = 0.0
        self.model.train()
        for x, y in train_loader:
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)

            # forward propagation
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            # backward propagation
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss

    def validation(self, val_loader: DataLoader):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # forward propagation
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        return val_loss
