"""Utility functions for the RDPS to HRDPS data loaders."""

import itertools
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from resoterre.ml.data_loader_utils import index_train_validation_test_split


def rdps_to_hrdps_split(
    valid_datetime_list: list[datetime],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
    rdps_window_size: int,
    overlap_factor: int,
    hrdps_required_unmasked_fraction: float,
    path_hrdps_mask: str,
    save_batch_size: int,
    temporal_window: int | None = 0,
    input_mode: str = "rdps_only",
) -> dict[str, Any]:
    """
    Split valid datetimes into train/validation/test sets and associate them with spatial patches and input modes.

    Parameters
    ----------
    valid_datetime_list : list[datetime]
        List of valid datetimes.
    train_fraction : float
        Fraction of datetimes to be used for training.
    validation_fraction : float
        Fraction of datetimes to be used for validation.
    test_fraction : float
        Fraction of datetimes to be used for testing.
    random_seed : int
        Random seed for reproducibility.
    rdps_window_size : int
        Size of the spatial window for RDPS data.
    overlap_factor : int
        Factor by which the spatial windows overlap.
    hrdps_required_unmasked_fraction : float
        Fraction of unmasked grid points in the HRDPS window required for inclusion.
    path_hrdps_mask : str
        Path to the HRDPS mask file (numpy .npz format with a "mask" array).
    save_batch_size : int
        Number of samples per saved batch.
    temporal_window : int, optional
        Number of hours before and after each datetime that is also part of the input data.
    input_mode : str, optional
        Input mode for the model, one of "rdps_only", "rdps_and_hrdps", or "hrdps_upscale".

    Returns
    -------
    dict[str, Any]
        A dictionary containing the split datetimes, patches and input modes.
    """
    if temporal_window is None:
        temporal_window = 0
    if input_mode == "rdps_only":
        input_mode_options = [False]
    elif input_mode == "rdps_and_hrdps":
        input_mode_options = [False, True]
    elif input_mode == "hrdps_upscale":
        input_mode_options = [True]
    else:
        raise ValueError(f"Invalid input_mode: {input_mode}")
    hrdps_window_size = rdps_window_size * 4
    num_windows = (256 // rdps_window_size) * (512 // rdps_window_size) * (overlap_factor**2)
    hrdps_mask = np.load(path_hrdps_mask)["mask"]
    window_scan_shape = (256 * overlap_factor // rdps_window_size, 512 * overlap_factor // rdps_window_size)
    rpds_step = rdps_window_size // overlap_factor
    hrdps_step = hrdps_window_size // overlap_factor

    valid_datetime_with_window = []
    for valid_datetime in valid_datetime_list:
        for i in range(1, temporal_window + 1):
            check_datetime = valid_datetime - timedelta(hours=i)
            if check_datetime not in valid_datetime_list:
                break
            check_datetime = valid_datetime + timedelta(hours=i)
            if check_datetime not in valid_datetime_list:
                break
        else:
            valid_datetime_with_window.append(valid_datetime)
    valid_datetime_list = valid_datetime_with_window

    valid_ijs = []
    for n in range(num_windows):
        (i, j) = np.unravel_index(n, window_scan_shape)
        i = int(i)  # ensure native int for json serialization
        j = int(j)
        i_rdps = i * rpds_step
        j_rdps = j * rpds_step
        i_hrdps = i * hrdps_step
        j_hrdps = j * hrdps_step
        mask_mean = np.mean(hrdps_mask[i_hrdps : i_hrdps + hrdps_window_size, j_hrdps : j_hrdps + hrdps_window_size])
        if mask_mean >= (1 - hrdps_required_unmasked_fraction):
            continue
        valid_ijs.append({"i_rdps": i_rdps, "j_rdps": j_rdps, "i_hrdps": i_hrdps, "j_hrdps": j_hrdps})

    train_split, validation_split, test_split = index_train_validation_test_split(
        n=len(valid_datetime_list),
        train_fraction=train_fraction,
        test_fraction_from_validation_set=test_fraction / (validation_fraction + test_fraction),
        random_seed=random_seed,
        shuffle=False,
        shuffle_within_sets=True,
    )

    rng = random.Random(random_seed)  # noqa: S311
    split_dict: dict[str, Any] = {}
    split_dict["train"] = list(
        itertools.product([valid_datetime_list[i] for i in train_split], valid_ijs, input_mode_options)
    )
    rng.shuffle(split_dict["train"])
    split_dict["train"] = [
        {
            "datetime_str": x[0].strftime("%Y%m%d%H"),
            "i_rdps": x[1]["i_rdps"],
            "j_rdps": x[1]["j_rdps"],
            "i_hrdps": x[1]["i_hrdps"],
            "j_hrdps": x[1]["j_hrdps"],
            "use_hrdps_upscale": x[2],
        }
        for x in split_dict["train"]
    ]
    split_dict["validation"] = list(
        itertools.product([valid_datetime_list[i] for i in validation_split], valid_ijs, input_mode_options)
    )
    rng.shuffle(split_dict["validation"])
    split_dict["validation"] = [
        {
            "datetime_str": x[0].strftime("%Y%m%d%H"),
            "i_rdps": x[1]["i_rdps"],
            "j_rdps": x[1]["j_rdps"],
            "i_hrdps": x[1]["i_hrdps"],
            "j_hrdps": x[1]["j_hrdps"],
            "use_hrdps_upscale": x[2],
        }
        for x in split_dict["validation"]
    ]
    split_dict["test"] = list(itertools.product([valid_datetime_list[i] for i in test_split], valid_ijs, [False]))
    rng.shuffle(split_dict["test"])
    split_dict["test"] = [
        {
            "datetime_str": x[0].strftime("%Y%m%d%H"),
            "i_rdps": x[1]["i_rdps"],
            "j_rdps": x[1]["j_rdps"],
            "i_hrdps": x[1]["i_hrdps"],
            "j_hrdps": x[1]["j_hrdps"],
            "use_hrdps_upscale": x[2],
        }
        for x in split_dict["test"]
    ]

    batch_dict: dict[str, Any] = {}
    for split in ["train", "validation", "test"]:
        batch_dict[split] = []
        for j in range(0, len(split_dict[split]), save_batch_size):
            batch_dict[split].append([dt for dt in split_dict[split][j : j + save_batch_size]])
    return batch_dict
