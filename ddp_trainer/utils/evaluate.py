import torch


def multi_class_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    # if one-hot, convert into labels
    if output.ndimension() > 1 and output.size(1) > 1:
        pred = output.argmax(dim=1)
    if target.ndimension() > 1 and target.size(1) > 1:
        target = target.argmax(dim=1)

    # calculus accuracy
    correct = pred.eq(target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


def binary_class_accuracy(output: torch.Tensor, target: torch.Tensor, threshold=0.5):
    # convert probability into 0/1
    pred = (output > threshold).float()

    # calculus accuracy
    correct = pred.eq(target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


def regression_accuracy(output: torch.Tensor, target: torch.Tensor):
    diff = output - target
    error_rate = torch.abs(diff) / target
    return 1 - torch.mean(error_rate)
