"""Module managing dataset managers for machine learning tasks."""

import logging
from typing import Any

from torch.utils import data as td

from resoterre.logging_utils import CustomLogging
from resoterre.memory_utils import check_over_memory


logger = CustomLogging(caller=logging.getLogger(__name__))

known_dataset_managers: dict[str, Any] = {}


def register_dataset_manager(name: str, known_dataset_managers_dict: dict[str, Any] | None = None) -> Any:
    """
    Decorator to register a dataset manager class with a given name.

    Parameters
    ----------
    name : str
        The name to register the dataset manager class under.
    known_dataset_managers_dict : dict[str, type], optional
        The dictionary to register the dataset manager class in. Defaults to the global known_dataset_managers.

    Returns
    -------
    decorator
        A decorator that registers the dataset manager class.
    """
    if known_dataset_managers_dict is None:
        known_dataset_managers_dict = known_dataset_managers

    def decorator(cls: Any) -> Any:
        """
        Decorator function to register the class.

        Parameters
        ----------
        cls : type
            The dataset manager class to register.

        Returns
        -------
        cls
            The original class, unmodified.
        """
        if name in known_dataset_managers_dict:
            raise ValueError(f"Dataset with name '{name}' already exists.")
        known_dataset_managers_dict[name] = cls
        return cls

    return decorator


def get_data_loader_item(data_loader_iterator: Any) -> Any:
    """
    Retrieve the next item from a data loader iterator.

    Parameters
    ----------
    data_loader_iterator : iterator
        An iterator over a data loader.

    Returns
    -------
    Any
        The next item from the data loader iterator.
    """
    return next(data_loader_iterator)


class DatasetManager:
    """
    Base class for managing datasets and data loaders.

    Parameters
    ----------
    data_loaders : dict[str, td.DataLoader], optional
        A dictionary mapping data loader names to their corresponding data loader objects.

    Notes
    -----
    This class should be subclassed to provide specific implementations for resetting data loaders.
    """

    def __init__(self, data_loaders: dict[str, td.DataLoader] | None = None) -> None:
        self.data_loaders: dict[str, td.DataLoader] = data_loaders or {}

    def __contains__(self, item: str) -> bool:
        """
        Whether the dataset manager contains a specific dataset.

        Parameters
        ----------
        item : str
            The name of the dataset to check.

        Returns
        -------
        bool
            True if the dataset manager contains the specified dataset, False otherwise.

        Notes
        -----
        This method should be overridden in subclasses to return True if the item will be created in reset_data_loader.
        """
        return False

    def reset_data_loader(
        self, data_loader_name: str, dataset_config: Any, data_loader_kwargs: dict[str, Any] | None
    ) -> td.DataLoader:
        """
        Reset the data loader for a specific dataset.

        Parameters
        ----------
        data_loader_name : str
            The name of the data loader to reset.
        dataset_config : Any
            The configuration for the dataset.
        data_loader_kwargs : dict[str, Any], optional
            Additional keyword arguments for resetting the data loader.

        Returns
        -------
        td.DataLoader
            The reset data loader.

        Notes
        -----
        This method should be overridden in subclasses to provide the actual reset logic.
        """
        pass

    def get_data_loader(self, data_loader_name: str, reset: bool = False, **kwargs: dict[str, Any]) -> td.DataLoader:
        """
        Retrieve a data loader, optionally resetting it.

        Parameters
        ----------
        data_loader_name : str
            The name of the data loader to retrieve.
        reset : bool
            Whether to reset the data loader before retrieving it.
        **kwargs : dict[str, Any]
            Additional keyword arguments for resetting the data loader.

        Returns
        -------
        td.DataLoader
            The requested data loader.
        """
        if (not reset) and (data_loader_name in self.data_loaders):
            return self.data_loaders[data_loader_name]
        return self.reset_data_loader(data_loader_name, **kwargs)

    def effective_batch_size(self, data_loader_name: str) -> int:
        """
        Calculate the effective batch size of a data loader.

        Parameters
        ----------
        data_loader_name : str
            The name of the data loader.

        Returns
        -------
        int
            The effective batch size of the data loader.
        """
        if data_loader_name not in self.data_loaders:
            raise ValueError(f"Data loader '{data_loader_name}' not found (perhaps it was not initialized).")
        built_in_batch_size = 1
        if hasattr(self.data_loaders[data_loader_name].dataset, "built_in_batch_size"):
            built_in_batch_size = self.data_loaders[data_loader_name].dataset.built_in_batch_size
        return int(built_in_batch_size * self.data_loaders[data_loader_name].batch_size)

    def loop_through(
        self,
        data_loader_names: list[str],
        restart_frequency: int | None = None,
        restart_memory_limit: float | None = None,
        memory_check_user_name: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Loop through specified data loaders, with optional restarts to manage memory usage.

        Parameters
        ----------
        data_loader_names : list[str]
            A list of data loader names to loop through.
        restart_frequency : int, optional
            The frequency (in number of batches) to restart the data loader to avoid memory leaks.
        restart_memory_limit : float, optional
            The memory limit (in GB) to trigger a restart of the data loader.
        memory_check_user_name : str, optional
            The user name to use for memory checking.
        **kwargs : dict[str, Any]
            Additional keyword arguments for resetting the data loaders.
        """
        # restart frequency was introduced to avoid memory issues when saving large dataset
        # restart memory limit is in GB
        for j, data_loader_name in enumerate(data_loader_names):
            # ToDo: if shuffle=True, the restart will not work as expected, the data loader will be shuffled again
            data_loader = self.get_data_loader(data_loader_name, reset=True, **kwargs)
            restart_count = -1
            last_i_processed = -1
            while True:
                restart_count += 1
                # ToDo: remove those logs, here to check how long it takes to create the iterator
                logger.debug("Creating data_loader_iterator")
                data_loader_iterator = iter(data_loader)
                logger.debug("Done creating data_loader_iterator")
                num_new_i = 0
                for i in range(len(data_loader)):
                    if i > last_i_processed:
                        logger.debug(
                            "Processing %s(%d/%d) %d/%d",
                            data_loader_name,
                            j + 1,
                            len(data_loader_names),
                            i + 1,
                            len(data_loader),
                            block_short_repetition_delay=10,
                            identifier=data_loader_name,
                            expected_nb_of_calls=len(data_loader),
                            add_eta=True,
                        )
                    try:
                        _ = get_data_loader_item(data_loader_iterator)
                    except RuntimeError:
                        del data_loader_iterator
                        break
                    if i > last_i_processed:
                        num_new_i += 1
                        last_i_processed = i
                    restart_by_frequency = (
                        (restart_frequency is not None) and (num_new_i > 0) and (num_new_i % restart_frequency == 0)
                    )
                    restart_by_memory = (restart_memory_limit is not None) and check_over_memory(
                        restart_memory_limit, user_name=memory_check_user_name
                    )
                    if restart_by_frequency or restart_by_memory:
                        del data_loader_iterator
                        # del self.data_loaders[data_loader_name]
                        logger.debug("Restarting data loader to avoid memory leak.")
                        break
                else:
                    del data_loader_iterator
                    del self.data_loaders[data_loader_name]
                    break
