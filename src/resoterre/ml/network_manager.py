"""Neural network manager module."""

from torch import nn


def nb_of_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Calculate the number of parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model for which to calculate the number of parameters.
    only_trainable : bool
        If True, only count trainable parameters. If False, count all parameters.

    Returns
    -------
    int
        The total number of parameters in the model.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
