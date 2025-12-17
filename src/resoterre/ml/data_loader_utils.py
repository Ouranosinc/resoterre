"""Utility functions for data loading and preprocessing in machine learning tasks."""

import random
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import torch
from sklearn.model_selection import train_test_split


Nested: TypeAlias = torch.Tensor | dict[Any, "Nested"] | list["Nested"] | tuple["Nested", ...]


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
    random_seed : int
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
    valid_min : float
        Minimum value for normalization. If None, the minimum of the data is used.
    valid_max : float
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
    log_offset: float = 1.0,
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
    log_offset : float
        Offset for logarithmic normalization to avoid log(0).

    Returns
    -------
    np.array
        Inverse normalized data.
    """
    if isinstance(mode, tuple):
        if log_normalize:
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
