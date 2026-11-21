"""Basic neural network building blocks."""

from typing import Any

import torch
from torch import nn
from torch.nn.functional import log_softmax


class Linear(nn.Linear):  # type: ignore[misc]
    """
    A linear layer that tracks its initialization function.

    Parameters
    ----------
    init_weight_fn_name : str, optional
        The name of the weight initialization function, by default None.
    init_weight_fn_kwargs : dict, optional
        Keyword arguments for the weight initialization function, by default None.
    **kwargs : dict
        Parameters for the torch.nn.Linear layer.
    """

    def __init__(
        self, init_weight_fn_name: str | None = None, init_weight_fn_kwargs: dict[Any, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.init_weight_fn_name = init_weight_fn_name
        self.init_weight_fn_kwargs = init_weight_fn_kwargs


class Conv2d(nn.Conv2d):  # type: ignore[misc]
    """
    A 2d convolution layer that tracks its initialization function.

    Parameters
    ----------
    init_weight_fn_name : str, optional
        The name of the weight initialization function, by default None.
    init_weight_fn_kwargs : dict, optional
        Keyword arguments for the weight initialization function, by default None.
    **kwargs : dict
        Parameters for the torch.nn.Conv2d layer.
    """

    def __init__(
        self, init_weight_fn_name: str | None = None, init_weight_fn_kwargs: dict[Any, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.init_weight_fn_name = init_weight_fn_name
        self.init_weight_fn_kwargs = init_weight_fn_kwargs


def init_weights(module: nn.Module) -> None:
    """
    Initialize weights of a module based on its initialization function name.

    Parameters
    ----------
    module : nn.Module
        The module whose weights are to be initialized.
    """
    for submodule in module.modules():
        if hasattr(submodule, "init_weight_fn_name") and submodule.init_weight_fn_name is not None:
            init_fn = getattr(nn.init, submodule.init_weight_fn_name, None)
            if init_fn is not None:
                init_kwargs = submodule.init_weight_fn_kwargs or {}
                init_fn(submodule.weight, **init_kwargs)
            else:
                raise ValueError(f"Initialization function {submodule.init_weight_fn_name} not found.")


class LinearReLU(nn.Module):  # type: ignore[misc]
    """
    A simple feedforward neural network with linear layers followed by ReLU activations.

    Parameters
    ----------
    input_size : int
        The size of the input features.
    hidden_sizes : list[int]
        A list of sizes for the hidden layers.
    output_size : int
        The size of the output features.
    final_activation : str, optional
        The type of activation function to apply to the output layer.
    """

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int, final_activation: str | None = None
    ) -> None:
        super().__init__()
        self.final_activation = final_activation
        connections = []
        for i in range(len(hidden_sizes)):
            if i != 0:
                input_size = hidden_sizes[i - 1]
            connections.append(
                Linear(in_features=input_size, out_features=hidden_sizes[i], init_weight_fn_name="kaiming_uniform_")
            )
            connections.append(nn.ReLU())
        if self.final_activation == "softmax":
            connections.append(Linear(in_features=hidden_sizes[-1], out_features=output_size))
        elif self.final_activation is not None:
            raise ValueError(f"Unsupported final activation: {self.final_activation}")
        self.sequential_block = nn.Sequential(*connections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LinearReLU block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the LinearReLU block.
        """
        x = self.sequential_block(x)
        if self.final_activation == "softmax":
            x = log_softmax(x, dim=1)
        elif self.final_activation is not None:
            raise ValueError(f"Unsupported final activation: {self.final_activation}")
        return x


class LinearEmbedding(nn.Module):  # type: ignore[misc]
    """
    Small dense network embedding a vector of scalar inputs.

    Parameters
    ----------
    in_features : int
        Number of input scalars.
    hidden_sizes : list[int]
        Output size of each dense layer, in order. A ReLU follows every layer.
    """

    def __init__(self, in_features: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size.")
        layers: list[nn.Module] = []
        for hidden_size in hidden_sizes:
            layers.append(Linear(in_features=in_features, out_features=hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        self.out_features = hidden_sizes[-1]
        self.sequential_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of scalar vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Embedding of shape (batch_size, out_features).
        """
        return self.sequential_block(x)


class CyclicalDateEncoding(nn.Module):  # type: ignore[misc]
    """
    Encode a calendar date (month and day) as a point on the unit circle.

    The day of the year is computed on a 366-day (leap) calendar and mapped to ``(sin, cos)`` of its angle, so that
    December 31 and January 1 are neighbours.
    """

    def __init__(self) -> None:
        super().__init__()
        days_in_month = torch.tensor([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=torch.float32)
        cumulative_days = torch.cumsum(days_in_month, dim=0) - days_in_month
        self.register_buffer("days_before_month", cumulative_days, persistent=False)
        self.register_buffer("days_in_month", days_in_month, persistent=False)

    def forward(self, month: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """
        Encode months and days.

        Parameters
        ----------
        month : torch.Tensor
            Month numbers (1 to 12) of shape (batch_size,).
        day : torch.Tensor
            Day of the month (1 to 31) of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 2) holding ``(sin, cos)`` of the day of the year angle.
        """
        month_idx = month.round().long() - 1
        day_in_month = torch.minimum(day, self.days_in_month[month_idx])
        day_of_year = self.days_before_month[month_idx] + day_in_month - 1
        angle = 2 * torch.pi * day_of_year / 366.0
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


class SEBlock(nn.Module):  # type: ignore[misc]
    """
    Squeeze and Excitation Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    reduction_ratio : int
        Reduction ratio for the number of channels in the bottleneck layer.
    min_reduced_channels : int
        Minimum number of reduced channels.

    Notes
    -----
    In the paper [1]_, they try reduction ratio 2, 4, 8, 16, and 32. See Table 10.

    References
    ----------
    .. [1] Hu, J., et al. (2017). Squeeze-and-Excitation Networks
       arXiv:1709.01507v4
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, min_reduced_channels: int = 2) -> None:
        super().__init__()
        reduced_channel = max(in_channels // reduction_ratio, min_reduced_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sequential_block = nn.Sequential(
            Linear(
                init_weight_fn_name="kaiming_uniform_",
                init_weight_fn_kwargs={"nonlinearity": "relu"},
                in_features=in_channels,
                out_features=reduced_channel,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channel, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Squeeze and Excitation block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the Squeeze and Excitation block.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.sequential_block(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
