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
