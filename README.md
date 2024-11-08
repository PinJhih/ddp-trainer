# DDP Trainer: A Python Package Simplify the Process of ML Model Training and Distributed Data Parallel in PyTorch

DDP Trainer is a Python package designed to simplify the process of machine learning models training and Distributed Data Parallel (DDP) in PyTorch.

## Install

To install the ddp-trainer, clone this repository and install it using pip:
```bash
git clone https://github.com/PinJhih/ddp-trainer.git && cd ./ddp-trainer
pip install .
```

## Usage 

### The Basic Programming Model

```python
# Import ddp_trainer
import ddp_trainer
from ddp_trainer import Trainer

# Initialize distributed processes
ddp_trainer.init()

# Load the dataset
...

# Convert into distributed dataloader
train_loader = ddp_trainer.get_loader(train_set)
test_loader = ddp_trainer.get_loader(test_set)

# Create the trainer
model = ...
loss_fn = ...
optimizer = ...
trainer = Trainer(model, loss_fn, optimizer)

# Train the model
trainer.train(epochs, train_loader, test_loader)
```

### Run the Training Script

Run on single GPU:
```bash
python /path/to/your/script.py
```

Run on single host with multiple GPUs (2 for example):
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 /path/to/your/script.py
```
- `--standalone`: Runs a single-node distributed job
- `--nnodes=1`: Specifies that only one node is used
- `--nproc_per_node=`: Specifies the number of processes, usually matching the number of GPUs

## Examples

Full examples can be found in the [examples](./examples).
