"""Neural network manager module."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ToDo: remove typing_extensions when phasing out Python < 3.11
try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from resoterre.config_utils import register_config
from resoterre.io_utils import purge_files
from resoterre.utils import TemplateStore


save_defaults = [
    "network_name",
    "experiment_name",
    "class",
    "model_state_dict",
    "optimizer_state_dict",
    "pth_files_history",
]


@dataclass(frozen=True, slots=True)
class PurgeModelFilesConfig:
    """
    Configuration for purging model files.

    Attributes
    ----------
    path : Path | str | None
        The directory path where model files are located. If None, the path will be inferred from
        the saved model files.
    older_than : float, optional
        Age in seconds; model files older than this will be purged.
    more_than : int, optional
        If the number of model files exceeds this number, the oldest files will be purged.
    must_both_be_true : bool
        If True, model files must meet both criteria to be purged; if False, meeting either criterion is sufficient.
    protected_files : list[str], optional
        List of model file names to protect from deletion.
    """

    path: Path | str | None = None  # ToDo: can be retrieved from templates?
    older_than: float | None = None  # in seconds
    more_than: int | None = None
    must_both_be_true: bool = False
    protected_files: list[str] | None = field(default_factory=list)  # List of file names to protect from deletion


@dataclass(frozen=True, slots=True)
class NeuralNetworksManagerConfig:
    """
    Configuration for the Neural Networks Manager.

    Attributes
    ----------
    device : str
        The device to use for the neural networks (e.g., 'cpu', 'cuda').
    purge_model_files : PurgeModelFilesConfig
        Configuration for purging model files.
    """

    device: str = "cpu"
    purge_model_files: PurgeModelFilesConfig = PurgeModelFilesConfig()


@register_config("Adam")
@dataclass(frozen=True, slots=True)
class AdamConfig:
    """
    Configuration for the Adam optimizer.

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay (L2 penalty) for the optimizer.
    """

    optimizer_name: str = field(default="Adam", metadata={"is_hyperparameter": True})
    lr: float = field(default=0.001, metadata={"is_hyperparameter": True})
    weight_decay: float = field(default=0.0, metadata={"is_hyperparameter": True})


@register_config("AdamW")
@dataclass(frozen=True, slots=True)
class AdamWConfig:
    """
    Configuration for the AdamW optimizer.

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer.
    lr : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay (L2 penalty) for the optimizer.
    beta_1 : float
        The beta_1 parameter for the AdamW optimizer.
    beta_2 : float
        The beta_2 parameter for the AdamW optimizer.
    """

    optimizer_name: str = field(default="AdamW", metadata={"is_hyperparameter": True})
    lr: float = field(default=0.001, metadata={"is_hyperparameter": True})
    weight_decay: float = field(default=0.0, metadata={"is_hyperparameter": True})
    beta_1: float = field(default=0.9, metadata={"is_hyperparameter": True})
    beta_2: float = field(default=0.999, metadata={"is_hyperparameter": True})


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


def neural_networks_pth_file_from_path(path_models: Path | str, experiment_name: str | None = None) -> dict[str, str]:
    """
    Find the latest neural network .pth files in a given directory.

    Parameters
    ----------
    path_models : Path | str
        The directory path where model files are located.
    experiment_name : str, optional
        The experiment name to filter model files.

    Returns
    -------
    dict[str, str]
        A dictionary mapping network names to their latest .pth file paths.
    """
    # Get the latest pth file in the directory, if it has info about other networks saved at the same time, include them
    if experiment_name is None:
        pattern = "*.pth"
    else:
        pattern = f"*{experiment_name}*.pth"
    pth_files = list(Path(path_models).glob(pattern))
    if not pth_files:
        return {}
    latest_file = max(pth_files, key=lambda x: x.stat().st_mtime)
    save_state = torch.load(latest_file, weights_only=False)
    # ToDo: review this use of last_saved_networks
    last_saved_networks = save_state.get(
        "pth_files_last_network_manager_save", [(save_state.get("network_name", ""), latest_file)]
    )
    neural_networks_pth_file = {k: v for k, v in last_saved_networks.items()}
    return neural_networks_pth_file


class NeuralNetworksManager:
    """
    Neural Networks Manager class to handle multiple neural networks, their optimizers, and learning rate schedulers.

    Parameters
    ----------
    networks : dict[str, nn.Module], optional
        A dictionary of neural network instances.
    optimizers : dict[str, torch.optim.Optimizer], optional
        A dictionary of optimizers for the neural networks.
    schedulers : dict[str, torch.optim.lr_scheduler._LRScheduler | None], optional
        A dictionary of learning rate schedulers for the optimizers.
    neural_network_classes : dict[str, type[nn.Module]], optional
        A dictionary of neural network classes.
    config : NeuralNetworksManagerConfig, optional
        Configuration for the Neural Networks Manager.
    """

    def __init__(
        self,
        networks: dict[str, nn.Module] | None = None,
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
        schedulers: dict[str, torch.optim.lr_scheduler._LRScheduler] | None = None,
        neural_network_classes: dict[str, type[nn.Module]] | None = None,
        config: NeuralNetworksManagerConfig | None = None,
    ):
        self.config = config or NeuralNetworksManagerConfig()
        self.networks = {} if networks is None else networks
        self.optimizers = {} if optimizers is None else optimizers
        self.lr_schedulers = {} if schedulers is None else schedulers
        self.neural_network_classes = {} if neural_network_classes is None else neural_network_classes
        self.pth_files_history: dict[str, list[str]] = {}
        self.to_device(self.config.device)

    @classmethod
    def from_neural_networks_info(cls: type[Self], neural_networks_info: dict[str, Any], runner_config: Any) -> Self:
        """
        Create a NeuralNetworksManager instance from neural network information and runner configuration.

        Parameters
        ----------
        neural_networks_info : dict[str, dict]
            A dictionary containing information about neural networks, including their classes.
        runner_config : any
            The configuration for the runner, which includes network configurations.

        Returns
        -------
        Self
            An instance of NeuralNetworksManager.
        """
        # ToDo: document what runner_config must be, and possibly provide a default???
        # No loading of pth files
        networks = {}
        for key, neural_network_info in neural_networks_info.items():
            kwargs = asdict(runner_config.networks.get(key)) if (key in runner_config.networks) else {}
            networks[key] = neural_network_info["class"](**kwargs)
        networks_manager_obj = cls(networks=networks, config=runner_config.networks_manager)
        for key in neural_networks_info.keys():
            networks_manager_obj.pth_files_history[key] = []
            networks_manager_obj.optimizers[key] = networks_manager_obj.optimizer_setup(
                network_parameters=networks_manager_obj.networks[key].parameters(), config=runner_config.optimizers[key]
            )
            networks_manager_obj.lr_schedulers[key] = networks_manager_obj.lr_scheduler_setup(
                networks_manager_obj.optimizers[key], config=runner_config.lr_schedulers.get(key, None)
            )
            networks_manager_obj.init_network_weights(key)
        return networks_manager_obj

    @classmethod
    def from_path_models(
        cls: type[Self],
        path_models: Path | str,
        neural_networks_classes: dict[str, type[nn.Module]],
        experiment_name: str | None = None,
        config: NeuralNetworksManagerConfig | None = None,
    ) -> tuple[Self, dict[str, Any]]:
        """
        Create a NeuralNetworksManager instance by loading neural networks from saved .pth files.

        Parameters
        ----------
        path_models : Path | str
            The directory path where model files are located.
        neural_networks_classes : dict[str, type[nn.Module]]
            A dictionary of neural network classes.
        experiment_name : str, optional
            The experiment name to filter model files.
        config : NeuralNetworksManagerConfig, optional
            Configuration for the Neural Networks Manager.

        Returns
        -------
        Self
            An instance of NeuralNetworksManager.
        dict[str, Any]
            A dictionary containing supplemental information for each network.
        """
        # ToDo: clarify what is a manager config and a runner config, perhaps rename things a little?
        neural_networks_pth_file = neural_networks_pth_file_from_path(path_models, experiment_name=experiment_name)
        networks_manager_obj = cls(config=config)
        networks_supplemental_info_dict = {}
        for key, pth_file in neural_networks_pth_file.items():
            save_state = torch.load(pth_file, weights_only=False)
            networks_supplemental_info_dict[key] = {k: v for k, v in save_state.items() if k not in save_defaults}
            networks_manager_obj.pth_files_history[key] = save_state.get("pth_files_history", [pth_file])
            network_class = save_state["class"]
            # ToDo: need to verify that the config match? present in multiple model saves!
            global_config = save_state.get("config", None)
            networks_manager_obj.config = global_config.networks_manager
            kwargs = asdict(global_config.networks[key]) if global_config and hasattr(global_config, "networks") else {}
            networks_manager_obj.networks[key] = neural_networks_classes[network_class](**kwargs)
            networks_manager_obj.optimizers[key] = networks_manager_obj.optimizer_setup(
                network_parameters=networks_manager_obj.networks[key].parameters(), config=global_config.optimizers[key]
            )
            networks_manager_obj.lr_schedulers[key] = networks_manager_obj.lr_scheduler_setup(
                networks_manager_obj.optimizers[key], config=global_config.lr_schedulers.get(key, None)
            )
            networks_manager_obj.load_network(
                key, save_state=save_state, return_state_dict=False, device=networks_manager_obj.config.device
            )
        networks_manager_obj.to_device(networks_manager_obj.config.device)
        return networks_manager_obj, networks_supplemental_info_dict

    @staticmethod
    def optimizer_setup(network_parameters: Any, config: Any) -> torch.optim.Optimizer:
        """
        Set up an optimizer for the neural network.

        Parameters
        ----------
        network_parameters : Any
            The parameters of the neural network to optimize.
        config : Any
            The configuration for the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The configured optimizer.
        """
        kwargs = {k: v for k, v in asdict(config).items() if k != "optimizer_name"}
        if isinstance(config, AdamWConfig):
            kwargs["betas"] = (kwargs.pop("beta_1"), kwargs.pop("beta_2"))
        optimizer = getattr(torch.optim, config.optimizer_name)(network_parameters, **kwargs)
        return optimizer

    @staticmethod
    def lr_scheduler_setup(
        optimizer: torch.optim.Optimizer, config: Any
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Set up a learning rate scheduler for the optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer for which to set up the learning rate scheduler.
        config : Any
            The configuration for the learning rate scheduler.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler | None
            The configured learning rate scheduler, or None if no configuration is provided.
        """
        if config is None:
            return None
        kwargs = {k: v for k, v in asdict(config).items() if k != "scheduler_name"}
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler_name)(optimizer, **kwargs)
        return scheduler

    def init_network_weights(self, network_name: str, **kwargs: Any) -> None:
        """
        Initialize the weights of a neural network.

        Parameters
        ----------
        network_name : str
            The name of the neural network to initialize.
        **kwargs
            Additional keyword arguments for the weight initialization function.
        """
        if hasattr(self.networks[network_name], "init_fn_tracker"):
            for key, layers in self.networks[network_name].init_fn_tracker.items():
                for layer in layers:
                    getattr(torch.nn.init, key)(layer.weight, **kwargs)
        # torch.nn.init.xavier_uniform_(self.network.layer.weight)  # For tanh, sigmoid and variants, already default?

    def save_network(
        self,
        network_name: str,
        path: str,
        experiment_name: str | None = None,
        pth_files_last_network_manager_save: dict[str, str] | None = None,
        supplemental_info_dict: dict[str, Any] | None = None,
    ) -> None:
        """
        Save the state of a neural network to a .pth file.

        Parameters
        ----------
        network_name : str
            The name of the neural network to save.
        path : str
            The file path where the network state will be saved.
        experiment_name : str, optional
            The experiment name associated with the network.
        pth_files_last_network_manager_save : dict[str, str], optional
            A dictionary of the last saved .pth files for each network.
        supplemental_info_dict : dict[str, Any], optional
            Additional information to include in the saved state.
        """
        supplemental_info_dict = {} if supplemental_info_dict is None else supplemental_info_dict
        save_dict = {
            "network_name": network_name,
            "experiment_name": experiment_name,
            "class": self.networks[network_name].__class__.__name__,
            "model_state_dict": self.networks[network_name].state_dict(),
            "optimizer_state_dict": self.optimizers[network_name].state_dict(),
            "pth_files_history": self.pth_files_history.get(network_name, [path]),
            "pth_files_last_network_manager_save": pth_files_last_network_manager_save,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({**supplemental_info_dict, **save_dict}, path)

    def optimizer_to_device(self, network_name: str, device: str) -> None:
        """
        Move the optimizer state to a specified device.

        Parameters
        ----------
        network_name : str
            The name of the neural network whose optimizer will be moved.
        device : str
            The device to which the optimizer state will be moved.
        """
        for state in self.optimizers[network_name].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def load_network(
        self,
        network_name: str,
        save_state: dict[str, Any] | None = None,
        pth_file: str | None = None,
        return_state_dict: bool = False,
        device: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Load a network.

        Parameters
        ----------
        network_name : str
            The name of the neural network to load.
        save_state : dict[str, Any], optional
            The state dictionary to load. If None, it will be loaded from the specified .pth file.
        pth_file : str, optional
            The file path from which to load the network state if save_state is not provided.
        return_state_dict : bool
            If True, return the full state dictionary after loading. If False, return None.
        device : str, optional
            The device to which the network and optimizer will be moved after loading.

        Returns
        -------
        dict[str, Any] | None
            The loaded state dictionary if return_state_dict is True; otherwise, None.
        """
        if device is None:
            device = self.config.device
        if save_state is None:
            if pth_file is None:
                raise RuntimeError("No save state or pth file provided.")
            save_state = torch.load(pth_file, weights_only=False)
        self.networks[network_name].load_state_dict(save_state["model_state_dict"])
        self.optimizers[network_name].load_state_dict(save_state["optimizer_state_dict"])
        self.optimizer_to_device(network_name, device)
        self.pth_files_history[network_name] = save_state.get("pth_files_history", [pth_file])
        if not return_state_dict:
            del save_state["model_state_dict"]
            del save_state["optimizer_state_dict"]
        return save_state

    def init_weights(self, **kwargs: Any) -> None:
        """
        Initialize the weights of all networks.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the weight initialization function.
        """
        for key in self.networks.keys():
            self.init_network_weights(key, **kwargs)

    def load_weights(self) -> None:
        """Load the latest weights for all networks from their respective .pth files."""
        for key in self.networks.keys():
            if self.pth_files_history[key]:
                self.load_network(network_name=key, pth_file=self.pth_files_history[key][-1])

    def to_device(self, device: str) -> None:
        """
        Move all networks to a specified device.

        Parameters
        ----------
        device : str
            The device to which the networks will be moved.
        """
        for network_specific in self.networks.values():
            network_specific.to(device)

    def train(self, mode: bool = True) -> None:
        """
        Perform training mode for all networks.

        Parameters
        ----------
        mode : bool
            If True, set the networks to training mode; if False, set them to evaluation mode.
        """
        for network_specific in self.networks.values():
            network_specific.train(mode=mode)  # ToDo: what is mode here?

    def __repr__(self) -> str:
        """
        String representation of the NeuralNetworksManager.

        Returns
        -------
        str
            A string representation of the NeuralNetworksManager.
        """
        network_str = ""
        for key, network in self.networks.items():
            nb_trainable_parameters = nb_of_parameters(network, only_trainable=True)
            nb_total_parameters = nb_of_parameters(network, only_trainable=False)
            network_str += (
                f"{key} ({nb_trainable_parameters} trainable parameters, "
                f"{nb_total_parameters} total parameters): \n{network!s}"
            )
        return network_str

    def total_nb_of_trainable_parameters(self) -> int:
        """
        Calculate the total number of trainable parameters across all networks.

        Returns
        -------
        int
            The total number of trainable parameters.
        """
        return sum([nb_of_parameters(network, only_trainable=True) for network in self.networks.values()])

    def save(
        self,
        path: str | None = None,
        experiment_name: str | None = None,
        supplemental_info_dict: dict[str, Any] | None = None,
        save_supplemental_info_dict_only_once: bool = False,
        templates: TemplateStore | None = None,
    ) -> dict[str, str]:
        """
        Save all networks.

        Parameters
        ----------
        path : str, optional
            The directory path where the networks will be saved. If None, templates must be provided.
        experiment_name : str, optional
            The experiment name associated with the networks.
        supplemental_info_dict : dict, optional
            Additional information to include in the saved state.
        save_supplemental_info_dict_only_once : bool
            If True, save the supplemental information dictionary only for the first network.
        templates : dict, optional
            A dictionary of templates for generating file paths.

        Returns
        -------
        dict[str, str]
            A dictionary mapping network names to their saved .pth file paths.
        """
        supplemental_info_dict = {} if supplemental_info_dict is None else supplemental_info_dict
        saved_pth_files = {}
        for key in self.networks.keys():
            if (path is None) and (templates is None) and (not self.pth_files_history[key]):
                raise RuntimeError("No path provided.")
            if path is not None:
                if experiment_name is None:
                    experiment_str = ""
                else:
                    experiment_str = f"{experiment_name}_"
                self.pth_files_history[key].append(str(Path(path, f"{experiment_str}{key}.pth")))
            elif templates is not None:
                templates.add_substitutes(model_name=key)
                if experiment_name is not None:
                    templates.add_substitutes(experiment_name=experiment_name)
                self.pth_files_history[key].append(templates["model_file"])
            saved_pth_files[key] = self.pth_files_history[key][-1]

        for i, key in enumerate(self.networks.keys()):
            if save_supplemental_info_dict_only_once and (i != 0):
                supplemental_info_dict = {}
            self.save_network(
                network_name=key,
                path=self.pth_files_history[key][-1],
                experiment_name=experiment_name,
                pth_files_last_network_manager_save=saved_pth_files,
                supplemental_info_dict=supplemental_info_dict,
            )
        return saved_pth_files

    def last_saved_models(self) -> list[str]:
        """
        Get the list of the last saved .pth files for all networks.

        Returns
        -------
        list[str]
            A list of the last saved .pth file paths.
        """
        return [x[-1] for x in self.pth_files_history.values() if x]

    def purge_model_files(self, protected_files: list[str] | None = None) -> list[str]:
        """
        Purge old model files based on the configuration.

        Parameters
        ----------
        protected_files : list[str], optional
            Additional model file names to protect from deletion.

        Returns
        -------
        list[str]
            A list of paths to the purged model files.
        """
        if protected_files is None:
            protected_files = []
        if self.config.purge_model_files.protected_files is not None:
            protected_files += self.config.purge_model_files.protected_files
        path = self.config.purge_model_files.path
        if path is None:
            for list_of_pth_files in self.pth_files_history.values():
                for pth_file in list_of_pth_files:
                    if pth_file is not None:
                        if path is None:
                            path = Path(pth_file).parent
                        elif path != Path(pth_file).parent:
                            raise RuntimeError("Ambiguous path for model files purge.")
        if path is not None:
            return purge_files(
                path=path,
                pattern="*.pth",
                older_than=self.config.purge_model_files.older_than,
                more_than=self.config.purge_model_files.more_than,
                must_both_be_true=self.config.purge_model_files.must_both_be_true,
                recursive=False,
                safe=True,
                excludes=protected_files,
            )
        return []
