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
import matplotlib.pyplot as plt


rank = 0
local_rank = 0
world_size = 0
is_distributed = False
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
        # Extract environ variable
        is_distributed = True
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Config torch.dist and device
        dist.init_process_group(backend)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)


def to_ddp_loader(ds: Dataset, batch_size=64, num_workers=4, seed=42):
    if is_distributed:
        sampler = DistributedSampler(ds, shuffle=True, seed=seed)
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler
        )
    else:
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return loader


class TrainHistory:
    metric_names = ["train_loss", "val_loss", "eval"]

    def __init__(self):
        self.history = {"train_loss": [], "val_loss": [], "eval": []}

    def append(self, metrics: dict):
        for name in TrainHistory.metric_names:
            if name in metrics:
                self.history[name].append(metrics[name])

    def get_train_losses(self):
        return self.history["train_loss"]

    def get_val_losses(self):
        return self.history["val_loss"]

    def get_evaluations(self):
        return self.history["eval"]

    def plot(self, path=None):
        train_losses = self.get_train_losses()
        val_losses = self.get_val_losses()
        evaluations = self.get_evaluations()

        plt.figure(figsize=(4, 6))
        fig, ax1 = plt.subplots()

        ax1.set_title("Training History")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(train_losses, label="train loss")

        if len(val_losses) != 0:
            ax1.plot(val_losses, label="val loss")

        if len(evaluations) != 0:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Eval")
            ax2.plot(evaluations, label="eval", c="green")
        fig.legend()
        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        eval_fn=None,
        scheduler=None,
    ) -> None:
        # Init DDP model
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
        self.history = TrainHistory()

    def train(self, epochs, train_loader: DataLoader, val_loader=None) -> nn.Module:
        progress = range(epochs)
        if rank == 0:
            progress = tqdm(progress, desc="Train")

        # Training loop
        for _ in progress:
            # Fit (forward/backward)
            train_loss = self.fit(train_loader)
            metrics = {"train_loss": train_loss}

            # Validate
            if val_loader is not None:
                val_loss, val_eval = self.validation(val_loader)
                metrics["val_loss"] = val_loss

                if self.eval_fn is not None:
                    metrics["eval"] = val_eval

            # Update progress bar
            if rank == 0:
                progress.set_postfix(metrics)
            self.history.append(metrics)

            # Update LR
            if self.scheduler is not None:
                self.scheduler.step()

    def fit(self, train_loader: DataLoader):
        train_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
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

            train_loss += loss
        train_loss /= len(train_loader)

        # compute the global training loss
        if is_distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        return train_loss.item()

    def validation(self, val_loader: DataLoader):
        val_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
        val_eval = torch.tensor(0.0, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # forward propagation
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                val_loss += loss

                # evaluate
                if self.eval_fn is not None:
                    val_eval += self.eval_fn(y_pred, y)

        val_loss /= len(val_loader)
        val_eval /= len(val_loader)

        # compute the global loss/metric
        if is_distributed:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_eval, op=dist.ReduceOp.AVG)
        return val_loss.item(), val_eval.item()

    def get_history(self):
        return self.history
