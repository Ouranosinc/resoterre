"""Utility functions for data loading and preprocessing in machine learning tasks."""

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import xarray
from sklearn.model_selection import train_test_split
from torch.utils import data as td

from resoterre.logging_utils import CustomLogging
from resoterre.utils import unique_hex_digest


logger = CustomLogging(caller=logging.getLogger(__name__))

Nested: TypeAlias = torch.Tensor | dict[Any, "Nested"] | list["Nested"] | tuple["Nested", ...]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration specific to a dataset. Used as parent class for type hinting."""

    pass


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """
    Configuration for a PyTorch DataLoader.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch to load. Default is 1.
    shuffle : bool
        Whether to shuffle the data at every epoch. Default is False.
    num_workers : int
        Number of subprocesses to use for data loading. Default is 0 (data will be loaded in the main process).
    multiprocessing_context : str, optional
        The multiprocessing context to use. If None, the default context is used. Default is None.
    """

    batch_size: int = field(default=1, metadata={"is_hyperparameter": True})
    shuffle: bool = False
    num_workers: int = 0
    multiprocessing_context: str | None = None


def index_train_validation_test_split(
    n: int,
    train_fraction: float = 0.8,
    test_fraction_from_validation_set: float = 0.5,
    random_seed: int | None = None,
    shuffle: bool = True,
    shuffle_within_sets: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split indices into training, validation, and test sets.

    Parameters
    ----------
    n : int
        Total number of samples.
    train_fraction : float
        Fraction of samples to use for training.
    test_fraction_from_validation_set : float
        Fraction of the remaining samples (after training) to use for testing.
        The remaining samples will be used for validation.
    random_seed : int, optional
        Random seed for reproducibility, by default None.
    shuffle : bool
        Whether to shuffle the indices before splitting.
    shuffle_within_sets : bool
        Whether to shuffle the indices within each set after splitting.

    Returns
    -------
    tuple
        Three lists of indices: (train_indices, validation_indices, test_indices).
    """
    idx = list(range(n))
    idx_train, idx_validation_test = train_test_split(
        idx, train_size=train_fraction, random_state=random_seed, shuffle=shuffle
    )
    idx_validation, idx_test = train_test_split(
        idx_validation_test, test_size=test_fraction_from_validation_set, random_state=random_seed, shuffle=shuffle
    )
    if shuffle_within_sets:
        random.seed(random_seed)
        random.shuffle(idx_train)
        random.shuffle(idx_validation)
        random.shuffle(idx_test)
    return idx_train, idx_validation, idx_test


def recursive_collate(list_of_structures: Sequence[Nested]) -> Nested:
    """
    Recursively collates a list of structures (tensors, dicts, lists, tuples) into a single structure.

    Parameters
    ----------
    list_of_structures : list[torch.Tensor | dict[str, Any] | list[Any] | tuple[Any]]
        List of structures to collate. Each structure can be a tensor, dict, list, or tuple.

    Returns
    -------
    torch.Tensor | dict | list | tuple
        A single structure that is the result of collating the input list.

    Notes
    -----
    Used when the Dataset already return batched samples, in which case the DataLoader batch_size must be 1
    """
    if all(isinstance(x, torch.Tensor) for x in list_of_structures):
        if len(list_of_structures) == 1:
            return list_of_structures[0]
        return torch.cat([item for item in list_of_structures])
    elif all(isinstance(x, dict) for x in list_of_structures):
        reference_dict = list_of_structures[0]
        if isinstance(reference_dict, dict):  # Check to silence mypy warning
            return {k: recursive_collate([item[k] for item in list_of_structures]) for k in reference_dict.keys()}
        else:
            raise RuntimeError()
    elif all(isinstance(x, list) for x in list_of_structures):
        return [recursive_collate([item[i] for item in list_of_structures]) for i in range(len(list_of_structures[0]))]
    elif all(isinstance(x, tuple) for x in list_of_structures):
        return tuple(
            [recursive_collate([item[i] for item in list_of_structures]) for i in range(len(list_of_structures[0]))]
        )
    else:
        raise NotImplementedError(f"Unrecognized structure for torch collate (type: {type(list_of_structures[0])}).")


def normalize(
    data: np.array,
    mode: tuple[int, int] = (-1, 1),
    valid_min: float | None = None,
    valid_max: float | None = None,
    log_normalize: bool = False,
    log_offset: float = 1.0,
) -> np.array:
    """
    Normalize data to a specified range, optionally applying logarithmic normalization.

    Parameters
    ----------
    data : np.array
        Input data to normalize.
    mode : tuple[int]
        The range to normalize the data to.
    valid_min : float, optional
        Minimum value for normalization. If None, the minimum of the data is used.
    valid_max : float, optional
        Maximum value for normalization. If None, the maximum of the data is used.
    log_normalize : bool
        Whether to apply logarithmic normalization.
    log_offset : float
        Offset for logarithmic normalization to avoid log(0).

    Returns
    -------
    np.array
        Normalized data.
    """
    # If data is a numpy memoryview because of previous slicing, convert it back to numpy array
    if isinstance(data, memoryview):
        data = data.obj

    if valid_min is None:
        valid_min = data.min()
    if valid_max is None:
        valid_max = data.max()
    if log_normalize:
        data = np.log(data - valid_min + log_offset)
        valid_range = valid_max - valid_min
        valid_min = np.log(log_offset)
        valid_max = np.log(valid_range + log_offset)
    if isinstance(mode, tuple):
        return mode[0] + (data - valid_min) * (mode[1] - mode[0]) / (valid_max - valid_min)
    else:
        raise NotImplementedError()


def inverse_normalize(
    data: np.array,
    known_min: float,
    known_max: float,
    mode: tuple[int, int] = (-1, 1),
    log_normalize: bool = False,
    log_offset: float | None = 1.0,
) -> np.array:
    """
    Inverse normalize data from a specified range, optionally applying logarithmic normalization.

    Parameters
    ----------
    data : np.array
        Input data to inverse normalize.
    known_min : float
        Minimum value previously used to normalize the data.
    known_max : float
        Maximum value previously used to normalize the data.
    mode : tuple[int]
        The range to inverse normalize the data from.
    log_normalize : bool
        Whether the data was logarithmically normalized.
    log_offset : float, optional
        Offset for logarithmic normalization to avoid log(0).

    Returns
    -------
    np.array
        Inverse normalized data.
    """
    if isinstance(mode, tuple):
        if log_normalize:
            if log_offset is None:
                raise ValueError("log_offset must be provided when log_normalize is True.")
            known_min_log = np.log(log_offset)
            known_max_log = np.log(known_max - known_min + log_offset)
            data = (data - mode[0]) * (known_max_log - known_min_log) / (mode[1] - mode[0]) + known_min_log
        else:
            data = (data - mode[0]) * (known_max - known_min) / (mode[1] - mode[0]) + known_min
    else:
        raise NotImplementedError()
    if log_normalize:
        data = np.exp(data) - log_offset + known_min
    return data


class DatasetWithSplits(td.Dataset):  # type: ignore[misc]
    """
    Dataset class with support for multiple splits and dynamic data keys.

    Parameters
    ----------
    dynamic_dataset_keys : list[str]
        List of keys for dynamic dataset components.
    active_split_name : str
        Name of the active split (e.g., 'train', 'validation', 'test').
    built_in_batch_size : int
        Built-in batch size of the dataset. If the dataset returns pre-batched samples.
    """

    def __init__(
        self, dynamic_dataset_keys: list[str], active_split_name: str = "train", built_in_batch_size: int = 1
    ) -> None:
        super().__init__()
        self.dynamic_dataset_keys = dynamic_dataset_keys
        self.active_split_name = active_split_name
        # If the dataset (its retrieve_key_data method) returns pre-batched samples, set built_in_batch_size > 1
        self.built_in_batch_size = built_in_batch_size
        self.effective_built_in_batch_size = self.built_in_batch_size if self.built_in_batch_size > 1 else 1
        self.fixed_data_cache: dict[str, np.array] = {}
        # Need self.train_idx_to_key, self.validation_idx_to_key, self.test_idx_to_key in subclass
        #     (or whatever the defined splits are)

    # Need methods retrieve_{dynamic_dataset_key}_data(self, key) in subclass

    def set_active_split(self, active_split_name: str) -> None:
        """
        Set the active split name.

        Parameters
        ----------
        active_split_name : str
            Name of the active split (e.g., 'train', 'validation', 'test').
        """
        self.active_split_name = active_split_name

    def __len__(self) -> int:
        """
        Get the length of the active split.

        Returns
        -------
        int
            Length of the active split.
        """
        return len(getattr(self, f"{self.active_split_name}_idx_to_key"))

    def compute_fixed_data_cache(self) -> None:
        """Compute and cache fixed data that does not change across samples."""
        pass  # To be overwritten in subclass if there is fixed data, using the cache to avoid recomputing

    def fixed_data(self, to_torch: bool = False) -> dict[str, np.array | torch.Tensor]:
        """
        Retrieve fixed data, computing it if not already cached.

        Parameters
        ----------
        to_torch : bool
            Whether to convert the fixed data to torch tensors.

        Returns
        -------
        dict[str, np.array | torch.Tensor]
            Fixed data dictionary.
        """
        self.compute_fixed_data_cache()
        if to_torch:
            return {k: torch.from_numpy(v) for k, v in self.fixed_data_cache.items()}
        return self.fixed_data_cache

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve data for a given index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor]
            Data for the given index.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for {self.active_split_name} split")
        key = getattr(self, f"{self.active_split_name}_idx_to_key")[idx]
        dynamic_data = {}
        for dynamic_dataset_key in self.dynamic_dataset_keys:
            dynamic_data[dynamic_dataset_key] = getattr(self, f"retrieve_{dynamic_dataset_key}_data")(key)
        return {key: torch.from_numpy(value) for key, value in dynamic_data.items()}

    def get_with_fixed_data(self, idx: int, to_torch: bool = False) -> dict[str, np.array | torch.Tensor]:
        """
        Retrieve data for a given index, including fixed data.

        Parameters
        ----------
        idx : int
            Sample index.
        to_torch : bool
            Whether to convert the fixed data to torch tensors.

        Returns
        -------
        dict[str, np.array | torch.Tensor]
            Data for the given index, including fixed data.
        """
        return {**self.fixed_data(to_torch=to_torch), **self.__getitem__(idx)}


class DatasetWithSave(DatasetWithSplits):
    """
    Dataset class with support for saving and loading preprocessed data batches.

    Parameters
    ----------
    dynamic_dataset_keys : list[str]
        List of keys for dynamic dataset components.
    path_ml_data : Path | str
        Path to the directory where preprocessed data batches are saved.
    only_from_ml_data : bool
        Whether to only load data from saved files.
    active_split_name : str
        Name of the active split (e.g., 'train', 'validation', 'test').
    built_in_batch_size : int
        Built-in batch size of the dataset. If the dataset returns pre-batched samples.
    save_batch_size : int
        Number of samples to save in each preprocessed data batch.
    skip_load : bool
        Used to skip loading the data during testing.
    """

    def __init__(
        self,
        dynamic_dataset_keys: list[str],
        path_ml_data: Path | str,
        only_from_ml_data: bool = False,
        active_split_name: str = "train",
        built_in_batch_size: int = 1,
        save_batch_size: int = 1,
        skip_load: bool = False,
    ):
        super().__init__(
            dynamic_dataset_keys, active_split_name=active_split_name, built_in_batch_size=built_in_batch_size
        )
        self._get_item_count = 0
        self.path_ml_data = path_ml_data
        if only_from_ml_data and self.path_ml_data is None:
            raise ValueError("path_ml_data must be provided when only_from_ml_data is True")
        self.only_from_ml_data = only_from_ml_data
        if (self.built_in_batch_size > 1) and (save_batch_size % self.built_in_batch_size != 0):
            raise ValueError("save_batch_size must be a multiple of built_in_batch_size")
        self.save_batch_size = save_batch_size  # To save multiple samples together
        self.cached_file: Path | str | None = None
        self.cached_data: Any = {}
        self.skip_load = skip_load  # Used when explicitly saving a data loader to disk to completely skip loading
        # Need self.train_idx_to_key, self.validation_idx_to_key, self.test_idx_to_key in subclass
        #     (or whatever the defined splits are)
        # self.hex_digest = self.compute_hex_digest()  # This should appear at the end of the __init__ method
        # WARNING! This actually assumes that the dataset will be accessed sequentially from 0 to len(self) - 1

    # Need methods retrieve_{dynamic_dataset_key}_data(self, key) in subclass

    def unique_elements_for_hash(
        self,
    ) -> tuple[str | int | float, ...]:  # To be overwritten in subclass by the elements that uniquely define the state
        """
        Generate a tuple of elements that uniquely define the dataset state for hashing.

        Returns
        -------
        tuple
            A tuple of elements that uniquely define the dataset state.
        """
        return self.active_split_name, *self.dynamic_dataset_keys, self.save_batch_size

    def compute_hex_digest(self) -> str:
        """
        Generate a unique hexadecimal digest based on the dataset's unique elements.

        Returns
        -------
        str
            A unique hexadecimal digest.
        """
        return unique_hex_digest(self.unique_elements_for_hash(), 8)

    def set_active_split(self, active_split_name: str) -> None:
        """
        Set the active split name.

        Parameters
        ----------
        active_split_name : str
            Name of the active split (e.g., 'train', 'validation', 'test').
        """
        self.active_split_name = active_split_name
        self.hex_digest = self.compute_hex_digest()

    def save_idx(self, idx: int) -> int:
        """
        Compute the save index for a given sample index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        int
            Save index corresponding to the sample index.
        """
        if self.built_in_batch_size == 0:
            return idx // self.save_batch_size
        else:
            return idx // (self.save_batch_size // self.built_in_batch_size)

    def saved_batch_path(self, idx: int) -> Path:
        """
        Generate the file path for a saved batch based on the sample index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Path
            File path for the saved batch.
        """
        return Path(
            self.path_ml_data, f"{self.active_split_name}_{self.hex_digest}_{str(self.save_idx(idx)).zfill(8)}.npz"
        )

    def saved_fixed_path(self) -> Path:
        """
        Generate the file path for the saved fixed data.

        Returns
        -------
        Path
            File path for the saved fixed data.
        """
        return Path(self.path_ml_data, f"fixed_{self.hex_digest}.npz")

    def set_fixed_data_cache_from_save(self) -> None:
        """Load fixed data from saved file into cache if not already cached."""
        fixed_npz_file = self.saved_fixed_path()
        if (not self.fixed_data_cache) and fixed_npz_file.is_file():
            self.fixed_data_cache = np.load(fixed_npz_file)
            self.fixed_data_cache = {key: value for key, value in self.fixed_data_cache.items()}

    def save_idx_data(self, idx: int, data_dict: dict[str, Any]) -> None:
        """
        Save preprocessed data for a given sample index.

        Parameters
        ----------
        idx : int
            Sample index.
        data_dict : dict[str, Any]
            Dictionary containing the data to save.
        """
        self.cached_data = data_dict
        self.cached_file = self.saved_batch_path(idx)
        logger.debug("Saving batch: %s", idx, identifier="save", block_short_repetition_delay=10)
        Path(self.path_ml_data).mkdir(parents=True, exist_ok=True)
        np.savez(self.saved_batch_path(idx), **data_dict)
        fixed_npz_file = self.saved_fixed_path()
        if not fixed_npz_file.is_file():
            fixed_data = self.fixed_data()
            if fixed_data:
                logger.debug("Saving fixed data", identifier="save_fixed", block_short_repetition_delay=10)
                np.savez(fixed_npz_file, **fixed_data)

    def load_idx_data(self, idx: int, saved_batch_npz_file: Path | str) -> dict[str, torch.Tensor]:
        """
        Load preprocessed data for a given sample index from a saved numpy file.

        Parameters
        ----------
        idx : int
            Sample index.
        saved_batch_npz_file : Path | str
            Path to the saved numpy file.

        Returns
        -------
        dict[str, torch.Tensor]
            Loaded data for the given index.
        """
        logger.debug("Loading: %s", idx, identifier="load", block_short_repetition_delay=10)
        if saved_batch_npz_file == self.cached_file:
            loaded_data = self.cached_data
        else:
            loaded_data = np.load(saved_batch_npz_file)
            self.cached_data = loaded_data
            self.cached_file = saved_batch_npz_file
        built_in_ratio = self.save_batch_size // self.effective_built_in_batch_size
        d = {}
        for key in loaded_data.keys():
            if (self.save_batch_size == 1) or (key not in self.dynamic_dataset_keys):
                x = loaded_data[key]
            elif self.built_in_batch_size == 0:
                x = loaded_data[key][idx % self.save_batch_size, ...]
            elif self.built_in_batch_size == 1:
                j = idx % built_in_ratio
                x = loaded_data[key][j : j + 1, ...]
            else:
                start = (idx % built_in_ratio) * self.built_in_batch_size
                end = start + self.built_in_batch_size
                x = loaded_data[key][start:end, ...]
            d[key] = torch.from_numpy(x)
        self.set_fixed_data_cache_from_save()
        return d

    def load_idx_data_netcdf(self, idx: int, saved_batch_netcdf_file: Path | str) -> dict[str, torch.Tensor]:
        """
        Load preprocessed data for a given sample index from a saved netCDF file.

        Parameters
        ----------
        idx : int
            Sample index.
        saved_batch_netcdf_file : Path | str
            Path to the saved netCDF file.

        Returns
        -------
        dict[str, torch.Tensor]
            Loaded data for the given index.
        """
        logger.debug("Loading: %s", idx, identifier="load", block_short_repetition_delay=10)
        # Reusing cached_file and cached_data for netcdf as well
        if saved_batch_netcdf_file == self.cached_file:
            loaded_dataset = self.cached_data
        else:
            if self.cached_data is not None:
                self.cached_data.close()
            loaded_dataset = xarray.open_dataset(saved_batch_netcdf_file)
            self.cached_data = loaded_dataset
            self.cached_file = saved_batch_netcdf_file
        built_in_ratio = self.save_batch_size // self.effective_built_in_batch_size
        d = {}
        data_variables = list(loaded_dataset.data_vars.keys())
        coordinate_variables = list(loaded_dataset._coord_names)
        for key in data_variables + coordinate_variables:
            if (self.save_batch_size == 1) or (key not in self.dynamic_dataset_keys):
                x = loaded_dataset[key].values
            elif self.built_in_batch_size == 0:
                x = loaded_dataset[key].values[idx % self.save_batch_size, ...]
            elif self.built_in_batch_size == 1:
                j = idx % built_in_ratio
                x = loaded_dataset[key].values[j : j + 1, ...]
            else:
                start = (idx % built_in_ratio) * self.built_in_batch_size
                end = start + self.built_in_batch_size
                x = loaded_dataset[key].values[start:end, ...]
            try:
                d[key] = torch.from_numpy(x)
            except TypeError:
                logger.debug(
                    "Skipping variable that cannot be converted to torch tensor: %s",
                    key,
                    identifier="load_skip",
                    block_short_repetition_delay=10,
                )
        self.set_fixed_data_cache_from_save()
        return d

    def retrieve_idx_data(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieve data for a given sample index from the subclass retrieve methods.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor]
            Data for the given index.
        """
        save_idx = self.save_idx(idx)
        built_in_ratio = self.save_batch_size // self.effective_built_in_batch_size
        save_data_dict: dict[str, np.ndarray] = {}
        start = None
        end = None
        n_samples = 0
        for save_batch_idx in range(save_idx * built_in_ratio, (save_idx + 1) * built_in_ratio):
            # Get item key from idx
            try:
                key = getattr(self, f"{self.active_split_name}_idx_to_key")[save_batch_idx]
            except IndexError:
                break
            for dataset_component in self.dynamic_dataset_keys:
                data = getattr(self, f"retrieve_{dataset_component}_data")(key)
                if (self.built_in_batch_size == 0) and (self.save_batch_size > 1):
                    data = np.expand_dims(data, axis=0)
                if dataset_component in save_data_dict:
                    save_data_dict[dataset_component] = np.concatenate(
                        (save_data_dict[dataset_component], data), axis=0
                    )
                else:
                    save_data_dict[dataset_component] = data
            n_samples += self.effective_built_in_batch_size
            if save_batch_idx == idx:
                start = n_samples - self.effective_built_in_batch_size
                end = n_samples
        if self.path_ml_data is not None:
            self.save_idx_data(idx, save_data_dict)
        if (self.built_in_batch_size <= 1) and (self.save_batch_size == 1):
            d = {key: torch.from_numpy(value) for key, value in save_data_dict.items()}
        elif self.built_in_batch_size == 0:
            d = {key: torch.from_numpy(value[start:end, ...].squeeze(0)) for key, value in save_data_dict.items()}
        else:
            d = {key: torch.from_numpy(value[start:end, ...]) for key, value in save_data_dict.items()}
        return d

    @property
    def dummy_data(self) -> dict[str, torch.Tensor]:
        """
        Get dummy data structure with correct shapes, used when skip_load is True.

        Returns
        -------
        dict[str, torch.Tensor]
            Dummy data for the dataset keys.

        Notes
        -----
        This skips all loading / retrieving of data. The subclass needs to ensure each shape is correct.
        """
        return {key: torch.zeros(1, dtype=torch.float32) for key in self.dynamic_dataset_keys}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get item by index, loading from saved data if available.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, torch.Tensor]
            Data for the given index.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for {self.active_split_name} split")
        if self.path_ml_data is not None:
            saved_batch_path = self.saved_batch_path(idx)
            if saved_batch_path.is_file():
                if self.skip_load:
                    logger.debug("Skipping: %s", idx, identifier="skip", block_short_repetition_delay=10)
                    return self.dummy_data
                if saved_batch_path.suffix == ".npz":
                    return self.load_idx_data(idx, saved_batch_path)
                elif saved_batch_path.suffix == ".nc":
                    return self.load_idx_data_netcdf(idx, saved_batch_path)
                else:
                    raise ValueError(f"Unrecognized file suffix: {saved_batch_path.suffix}")
            elif self.only_from_ml_data:
                raise FileNotFoundError(f"File not found: {saved_batch_path}")
        get_item_result = self.retrieve_idx_data(idx)
        self._get_item_count += 1
        return get_item_result


class DatasetFromNetCDFSave(DatasetWithSave):
    """
    Dataset class that loads data from saved netCDF files.

    Parameters
    ----------
    dynamic_dataset_keys : list[str]
        List of keys for dynamic dataset components.
    path_ml_data : Path | str
        Path to the directory where preprocessed data batches are saved.
    active_split_name : str
        Name of the active split (e.g., 'train', 'validation', 'test').
    built_in_batch_size : int
        Built-in batch size of the dataset. If the dataset returns pre-batched samples.
    save_batch_size : int
        Number of samples to save in each preprocessed data batch. Set to 0 to infer from data.
    skip_load : bool
        Used to skip loading the data during testing.
    """

    def __init__(
        self,
        dynamic_dataset_keys: list[str],
        path_ml_data: Path | str,
        active_split_name: str = "train",
        built_in_batch_size: int = 1,
        save_batch_size: int = 1,
        skip_load: bool = False,
    ) -> None:
        if save_batch_size < 1:
            sample_ds = xarray.open_dataset(Path(path_ml_data, f"{active_split_name}_00000000.nc"))
            save_batch_size = sample_ds[dynamic_dataset_keys[0]].shape[0]
        super().__init__(
            dynamic_dataset_keys,
            path_ml_data=path_ml_data,
            only_from_ml_data=True,
            active_split_name=active_split_name,
            built_in_batch_size=built_in_batch_size,
            save_batch_size=save_batch_size,
            skip_load=skip_load,
        )
        self.train_idx_to_key = self.infer_idx_from_nc("train")
        self.validation_idx_to_key = self.infer_idx_from_nc("validation")
        self.test_idx_to_key = self.infer_idx_from_nc("test")
        self.hex_digest = self.compute_hex_digest()

    def infer_idx_from_nc(self, split_name: str) -> list[str]:
        """
        Infer indices from saved netCDF files for a given split.

        Parameters
        ----------
        split_name : str
            Name of the split (e.g., 'train', 'validation', 'test').

        Returns
        -------
        list[str]
            List of inferred indices for the split.
        """
        nc_files = sorted(list(Path(self.path_ml_data).glob(f"{split_name}_*.nc")))
        last_dataset = xarray.open_dataset(nc_files[-1])
        num_samples = (len(nc_files) - 1) * self.save_batch_size + last_dataset[self.dynamic_dataset_keys[0]].shape[0]
        # ToDo: not sure this works for all variations of built_in_batch_size and save_batch_size
        return [str(x) for x in range(math.ceil(num_samples / self.effective_built_in_batch_size))]

    def saved_batch_path(self, idx: int) -> Path:
        """
        Generate the file path for a saved netCDF batch based on the sample index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Path
            File path for the saved netCDF batch.
        """
        # ToDo: reintroduce the option to use hex digest identifier?
        return Path(self.path_ml_data, f"{self.active_split_name}_{str(self.save_idx(idx)).zfill(8)}.nc")
