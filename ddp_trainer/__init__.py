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


is_distributed = False
rank = 0
local_rank = 0
world_size = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)


def init(seed=42, backend="nccl"):
    global is_distributed
    global rank
    global local_rank
    global world_size
    global device

    set_random_seed(seed)
    if "RANK" in os.environ:
        # Save environ variable
        is_distributed = True
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Config dist and device
        dist.init_process_group(backend)
        torch.cuda.set_device(local_rank)
        Trainer.device = torch.device("cuda", local_rank)


def get_loader(ds: Dataset, batch_size=64, num_workers=4, seed=42):
    if is_distributed:
        sampler = DistributedSampler(ds, shuffle=True, seed=seed)
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler
        )
    else:
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return loader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        eval_fn=None,
        scheduler=None,
    ) -> None:
        if is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(device)
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, epochs, train_loader: DataLoader, val_loader=None) -> nn.Module:
        progress = range(epochs)
        if rank == 0:
            progress = tqdm(progress, desc="Train")

        for _ in progress:
            # Train
            train_loss = self.train(train_loader)
            metrics = {"train_loss": train_loss}

            # Validate
            if val_loader is not None:
                val_loss, score = self.validation(val_loader)
                metrics["val_loss"] = val_loss

                if self.eval_fn is not None:
                    metrics["eval"] = score

            # Update progress bar
            if rank == 0:
                progress.set_postfix(metrics)

            if self.scheduler is not None:
                self.scheduler.step()
        return self.model

    def train(self, train_loader: DataLoader):
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
        return train_loss

    def validation(self, val_loader: DataLoader):
        val_loss = 0.0
        score = 0.0

        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # forward propagation
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                val_loss += loss.item()

                # evaluate
                if self.eval_fn is not None:
                    score += self.eval_fn(y_pred, y)
        val_loss /= len(val_loader)
        score /= len(val_loader)
        return val_loss, score
