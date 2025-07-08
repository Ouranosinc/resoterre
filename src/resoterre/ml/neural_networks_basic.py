"""Basic neural network building blocks."""

import torch
from torch import nn


class ModuleListWithInitFnTracker(nn.ModuleList):  # type: ignore[misc]
    """
    Tracker for initialization functions of modules in a ModuleList.

    Attributes
    ----------
    init_fn_tracker : dict[str, list[nn.Module]]
        A dictionary that maps initialization function names to lists of modules
        initialized with those functions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.init_fn_tracker: dict[str, list[nn.Module]] = {}

    def append_with_init_fn(
        self,
        module_list_obj: nn.ModuleList,
        module: nn.Module,
        init_fn_str: str | None = None,
    ) -> None:
        """
        Append a module to the ModuleList and track its initialization function.

        Parameters
        ----------
        module_list_obj : nn.ModuleList
            The ModuleList to which the module will be appended.
        module : nn.Module
            The module to be appended.
        init_fn_str : str, optional
            The name of the initialization function used for the module.
            If provided, it will be used to track the initialization function.

        Raises
        ------
        ValueError
            If the module does not have an 'init_fn_tracker' attribute and no init_fn_str is provided.
        """
        module_list_obj.append(module)
        if hasattr(module, "init_fn_tracker"):
            for key, value in module.init_fn_tracker.items():
                if key not in self.init_fn_tracker:
                    self.init_fn_tracker[key] = []
                self.init_fn_tracker[key].extend(value)
        elif init_fn_str is not None:
            if init_fn_str not in self.init_fn_tracker:
                self.init_fn_tracker[init_fn_str] = []
            self.init_fn_tracker[init_fn_str].append(module_list_obj[-1])
        else:
            raise ValueError(f"Module {module} does not have 'init_fn_tracker' attribute and no init_fn_str provided.")


class SEBlock(ModuleListWithInitFnTracker):
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
        self.connections = nn.ModuleList()
        self.append_with_init_fn(
            self.connections,
            nn.Linear(in_channels, reduced_channel, bias=False),
            "kaiming_uniform_",
        )
        self.connections.append(nn.ReLU(inplace=True))
        self.connections.append(nn.Linear(reduced_channel, in_channels, bias=False))
        self.connections.append(nn.Sigmoid())
        self.sequential = nn.Sequential(*self.connections)

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
        y = self.sequential(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
