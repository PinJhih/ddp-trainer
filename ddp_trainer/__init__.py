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


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)


class Trainer:
    def init(seed=42, backend="nccl"):
        set_random_seed(seed)

        if "RANK" in os.environ:
            # Save environ variable
            Trainer.is_distributed = True
            Trainer.rank = int(os.environ["RANK"])
            Trainer.local_rank = int(os.environ["LOCAL_RANK"])
            Trainer.world_size = int(os.environ["WORLD_SIZE"])

            # Config dist and device
            dist.init_process_group(backend)
            torch.cuda.set_device(Trainer.local_rank)
            Trainer.device = torch.device("cuda", Trainer.local_rank)
        else:
            Trainer.is_distributed = False
            Trainer.rank = 0
            Trainer.local_rank = 0
            Trainer.world_size = 0
            Trainer.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_loader(ds: Dataset, batch_size=64, num_workers=4, seed=42):
        if Trainer.is_distributed:
            sampler = DistributedSampler(ds, shuffle=True, seed=seed)
            loader = DataLoader(
                ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler
            )
        else:
            loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        return loader

    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        eval_fn=None,
        scheduler=None,
    ) -> None:
        if Trainer.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[Trainer.local_rank],
                output_device=Trainer.local_rank,
            )
        else:
            self.model = model.to(Trainer.device)
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, epochs, train_loader: DataLoader, val_loader=None) -> nn.Module:
        progress = range(epochs)
        if Trainer.rank == 0:
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
            if Trainer.rank == 0:
                progress.set_postfix(metrics)

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
        score = 0.0

        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

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
