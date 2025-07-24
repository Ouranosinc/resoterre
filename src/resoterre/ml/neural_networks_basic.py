"""Basic neural network building blocks."""

import torch
from torch import nn


class ModuleWithInitTracker:
    """
    A module that tracks its initialization functions.

    Parameters
    ----------
    module : nn.Module
        The module to track.
    init_weight_fn_name : str | None, optional
        The name of the weight initialization function, by default None.
    init_weight_fn_kwargs : dict[str, str] | None, optional
        Keyword arguments for the weight initialization function, by default None.
    init_bias_fn_name : str | None, optional
        The name of the bias initialization function, by default None.
    """

    def __init__(
        self,
        module: nn.Module,
        init_weight_fn_name: str | None = None,
        init_weight_fn_kwargs: dict[str, str] | None = None,
        init_bias_fn_name: str | None = None,
    ) -> None:
        self.module = module
        self.init_weight_fn_name = init_weight_fn_name
        self.init_weight_fn_kwargs = {} if init_weight_fn_kwargs is None else init_weight_fn_kwargs
        self.init_bias_fn_name = init_bias_fn_name


class ModuleInitFnTracker:
    """
    Tracker for initialization functions of modules.

    Attributes
    ----------
    init_fn_tracker : list[ModuleWithInitTracker]
        A list that tracks the initialization functions of modules.
    """

    def __init__(self) -> None:
        super().__init__()
        self.init_fn_tracker: list[ModuleWithInitTracker] = []

    def track_init_fn(self, module: nn.Module | ModuleWithInitTracker) -> nn.Module:
        """
        Track the initialization function of a module.

        Parameters
        ----------
        module : nn.ModuleWithInitTracker
            The module to track.

        Returns
        -------
        nn.Module
            The original module.
        """
        if hasattr(module, "init_fn_tracker"):
            self.init_fn_tracker.extend(module.init_fn_tracker)
        elif isinstance(module, ModuleWithInitTracker):
            self.init_fn_tracker.append(module)
            return module.module
        return module

    def init_weights(self) -> None:
        """Initialize the weights and biases of the tracked modules using the specified initialization functions."""
        for tracker in self.init_fn_tracker:
            if tracker.init_weight_fn_name is not None:
                getattr(nn.init, tracker.init_weight_fn_name)(tracker.module.weight, **tracker.init_weight_fn_kwargs)
            if tracker.init_bias_fn_name is not None:
                getattr(nn.init, tracker.init_bias_fn_name)(tracker.module.module.bias)


class ModuleListInitTracker(nn.ModuleList):  # type: ignore[misc]
    """
    A module list that tracks initialization functions of its modules.

    Parameters
    ----------
    init_fn_tracker : list[ModuleWithInitTracker]
        A list that tracks the initialization functions of modules.
    modules : list[nn.Module] | None, optional
        An optional list of modules to initialize the ModuleList with, by default None.
    """

    def __init__(self, init_fn_tracker: list[ModuleWithInitTracker], modules: list[nn.Module] | None = None) -> None:
        self.init_fn_tracker = init_fn_tracker
        if modules is not None:
            for module in modules:
                if hasattr(module, "init_fn_tracker"):
                    self.init_fn_tracker.extend(module.init_fn_tracker)
        super().__init__(modules=modules)

    def append(self, module: nn.Module) -> None:
        """
        Append a module to the list and track its initialization function.

        Parameters
        ----------
        module : nn.Module
            The module to append.
        """
        if hasattr(module, "init_fn_tracker"):
            self.init_fn_tracker.extend(module.init_fn_tracker)
        super().append(module)


class SequentialInitTracker(nn.Sequential):  # type: ignore[misc]
    """
    A sequential container that tracks initialization functions of its modules.

    Parameters
    ----------
    init_tracker : list[ModuleWithInitTracker]
        A list that tracks the initialization functions of modules.
    *args : nn.Module
        Variable length argument list of modules to initialize the Sequential with.
    """

    def __init__(self, init_tracker: list[ModuleWithInitTracker], *args: nn.Module) -> None:
        for arg in args:
            if hasattr(arg, "init_fn_tracker"):
                init_tracker.extend(arg.init_fn_tracker)
        super().__init__(*args)


class SEBlock(ModuleInitFnTracker, nn.Module):  # type: ignore[misc]
    """
    Squeeze and Excitation Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    reduction_ratio : int, default=16
        Reduction ratio for the number of channels in the bottleneck layer.
    min_reduced_channels : int, default=2
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
            self.track_init_fn(
                ModuleWithInitTracker(
                    nn.Linear(in_channels, reduced_channel, bias=False),
                    init_weight_fn_name="kaiming_uniform_",
                    init_weight_fn_kwargs={"nonlinearity": "relu"},
                )
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
