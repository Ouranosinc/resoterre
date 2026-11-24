"""
Losses of the CRCM emulator.

Notes
-----
Adding new losses:

1. Define a class inheriting from ``Loss`` with a ``forward`` returning the scalar loss over all variables, and a
    ``per_variable`` method returning the loss of each variable (shape ``(variables,)``), which is what the
    trainer uses.
2. Add the class to ``AVAILABLE_LOSSES`` under its name.
3. Its weight is then set by name in ``loss.default`` or ``loss.variables`` of the configuration.

Logging and reduction:

The total loss and each term are logged with the metrics under the names in ``LOSS_METRIC_NAMES`` (``loss`` and
``<name>_loss``, e.g. ``mse_loss``), plus ``<variable>/<name>_loss`` per variable. These names follow
``AVAILABLE_LOSSES``, so a new loss is logged without further registration.

Across batches, every logged loss is averaged with the sample-mean reducer (``SAMPLE_MEAN_REDUCER`` in
``resoterre.pipelines.crcm_emulator.metrics``), which weights each batch by its number of samples so that a partial
last batch counts less. ``reducer_for_key`` in that module selects it for any key whose last part (after
``<variable>/``) is in ``LOSS_METRIC_NAMES``. There is currently no per-loss choice of reducer: to change it for all
losses, edit ``reducer_for_key``.
"""

import torch
import torch.nn as nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class Loss(nn.Module):  # type: ignore[misc]
    """Base class of the losses of the CRCM emulator."""

    def per_variable(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of each variable.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Loss of each variable, of shape (variables,).
        """
        raise NotImplementedError


class MSELoss(Loss):
    """Mean squared error."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error over all variables.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return nn.functional.mse_loss(output, target)

    def per_variable(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error of each variable.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Loss of each variable, of shape (variables,).
        """
        return ((output - target) ** 2).mean(dim=(0, 2, 3))


class MAELoss(Loss):
    """Mean absolute error."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean absolute error over all variables.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return nn.functional.l1_loss(output, target)

    def per_variable(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean absolute error of each variable.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Loss of each variable, of shape (variables,).
        """
        return (output - target).abs().mean(dim=(0, 2, 3))


class SSIMLoss(Loss):
    """
    One minus the multi-scale structural similarity index, for targets in [-1, 1].

    Parameters
    ----------
    data_range : tuple[float, float]
        Range of the target values.
    """

    def __init__(self, data_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        super().__init__()
        self.ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss over all variables at once.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        loss: torch.Tensor = 1 - self.ssim(output, target)
        return loss

    def per_variable(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of each variable.

        Parameters
        ----------
        output : torch.Tensor
            Model output, of shape (batch, variables, height, width).
        target : torch.Tensor
            Ground truth, same shape as ``output``.

        Returns
        -------
        torch.Tensor
            Loss of each variable, of shape (variables,).
        """
        return torch.stack([self(output[:, i : i + 1], target[:, i : i + 1]) for i in range(output.shape[1])])


# Losses that can be combined in ``training_loss_computation``, by name.
AVAILABLE_LOSSES: dict[str, type[Loss]] = {
    "mse": MSELoss,
    "mae": MAELoss,
    "ssim": SSIMLoss,
}

# Names under which the total loss and each loss term are logged with the metrics (e.g. ``mse_loss``).
LOSS_METRIC_NAMES: tuple[str, ...] = ("loss", *(f"{name}_loss" for name in AVAILABLE_LOSSES))
