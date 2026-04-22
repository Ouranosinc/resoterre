"""Utility functions for the RDPS to HRDPS data loaders."""

import itertools
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray

from resoterre.data_management.data_io import sample_chunk_size
from resoterre.data_management.netcdf_utils import add_xarray_variable
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
    path_hrdps_mask: Path | str,
    save_batch_size: int,
    temporal_window: int | None = 0,
    restrict_hrdps_i_j: list[list[int]] | None = None,
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
    path_hrdps_mask : Path | str
        Path to the HRDPS mask file (numpy .npz format with a "mask" array).
    save_batch_size : int
        Number of samples per saved batch.
    temporal_window : int, optional
        Number of hours before and after each datetime that is also part of the input data.
    restrict_hrdps_i_j : list[list[int]]
        List of [i, j] pairs to restrict the HRDPS tiles to specific locations.
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
    restrict_hrdps_i_j = restrict_hrdps_i_j or []
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
        if restrict_hrdps_i_j and ([i_hrdps, j_hrdps] not in restrict_hrdps_i_j):
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


def save_ml_input(
    output_file: Path | str,
    input_first_block: np.ndarray,
    input_last_layer: np.ndarray | None,
    target: np.ndarray,
    heights_in_idx: np.ndarray,
    widths_in_idx: np.ndarray,
    heights_idx: np.ndarray,
    widths_idx: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    year_1d: np.ndarray,
    month_1d: np.ndarray,
    day_1d: np.ndarray,
    hour_1d: np.ndarray,
    list_of_input_variables: list[str],
    list_of_output_variables: list[str],
    use_hrdps_upscale: np.ndarray,
) -> None:
    """
    Save the ML input data to a NetCDF file with CF conventions.

    Parameters
    ----------
    output_file : Path | str
        Path to the output NetCDF file.
    input_first_block : np.ndarray
        Input data for the first block.
    input_last_layer : np.ndarray | None
        Input data for the last layer.
    target : np.ndarray
        Target data.
    heights_in_idx : np.ndarray
        Height indices for the input data.
    widths_in_idx : np.ndarray
        Width indices for the input data.
    heights_idx : np.ndarray
        Height indices for the output data.
    widths_idx : np.ndarray
        Width indices for the output data.
    latitudes : np.ndarray
        Latitude values for the output grid.
    longitudes : np.ndarray
        Longitude values for the output grid.
    year_1d : np.ndarray
        Year values for each sample.
    month_1d : np.ndarray
        Month values for each sample.
    day_1d : np.ndarray
        Day values for each sample.
    hour_1d : np.ndarray
        Hour values for each sample.
    list_of_input_variables : list[str]
        List of input variable names.
    list_of_output_variables : list[str]
        List of output variable names.
    use_hrdps_upscale : np.ndarray
        Boolean array indicating whether HRDPS upscale is used for each sample.
    """
    num_input_channel = input_first_block[0].shape[0]
    if input_last_layer is None:
        num_last_layer_channel = 0
    else:
        num_last_layer_channel = input_last_layer[0].shape[0]
    num_target_channel = target[0].shape[0]
    height_in = input_first_block[0].shape[1]
    width_in = input_first_block[0].shape[2]
    height = target[0].shape[1]
    width = target[0].shape[2]

    encoding: dict[str, Any] = {}
    cf_coordinates: dict[str, xarray.Variable] = {}
    add_xarray_variable(
        cf_coordinates, "height_in_idx", heights_in_idx, encoding, dims=("sample", "height_in"), dtype=np.int16
    )
    add_xarray_variable(
        cf_coordinates, "width_in_idx", widths_in_idx, encoding, dims=("sample", "width_in"), dtype=np.int16
    )
    add_xarray_variable(
        cf_coordinates, "height_out_idx", heights_idx, encoding, dims=("sample", "height_out"), dtype=np.int16
    )
    add_xarray_variable(
        cf_coordinates, "width_out_idx", widths_idx, encoding, dims=("sample", "width_out"), dtype=np.int16
    )
    add_xarray_variable(
        cf_coordinates,
        "lat",
        latitudes,
        encoding,
        dims=("sample", "height_out"),
        attrs={"units": "degrees_north", "standard_name": "latitude"},
        dtype=np.float32,
    )
    add_xarray_variable(
        cf_coordinates,
        "lon",
        longitudes,
        encoding,
        dims=("sample", "width_out"),
        attrs={"units": "degrees_east", "standard_name": "longitude"},
        dtype=np.float32,
    )
    add_xarray_variable(cf_coordinates, "year", year_1d, encoding, dims=("sample",), dtype=np.int16)
    add_xarray_variable(cf_coordinates, "month", month_1d, encoding, dims=("sample",), dtype=np.int8)
    add_xarray_variable(cf_coordinates, "day", day_1d, encoding, dims=("sample",), dtype=np.int8)
    add_xarray_variable(cf_coordinates, "hour", hour_1d, encoding, dims=("sample",), dtype=np.int8)
    add_xarray_variable(
        cf_coordinates, "input_variables", list_of_input_variables, encoding, dims=("input_channel",), dtype="object"
    )
    add_xarray_variable(
        cf_coordinates, "output_variables", list_of_output_variables, encoding, dims=("target_channel",), dtype="object"
    )
    add_xarray_variable(
        cf_coordinates, "use_hrdps_upscale", use_hrdps_upscale, encoding, dims=("sample",), dtype=np.int8
    )

    cf_variables: dict[str, xarray.Variable] = {}
    add_xarray_variable(
        cf_variables,
        "input_first_block",
        input_first_block,
        encoding,
        dims=("sample", "input_channel", "height_in", "width_in"),
        dtype=np.float32,
        zlib=True,
        complevel=4,
    )
    if input_last_layer is not None:
        add_xarray_variable(
            cf_variables,
            "input_last_layer",
            input_last_layer,
            encoding,
            dims=("sample", "last_layer_channel", "height_out", "width_out"),
            dtype=np.float32,
            zlib=True,
            complevel=4,
        )
    add_xarray_variable(
        cf_variables,
        "target",
        target,
        encoding,
        dims=("sample", "target_channel", "height_out", "width_out"),
        dtype=np.float32,
        zlib=True,
        complevel=4,
    )

    cf_attrs = {"Conventions": "CF-1.6"}
    ds = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)

    sample_chunk = sample_chunk_size(extra_dimensions_product=num_input_channel * height * width)
    sample_chunk = max(1, min(sample_chunk, len(input_first_block)))
    encoding["input_first_block"]["chunksizes"] = (sample_chunk, num_input_channel, height_in, width_in)
    if input_last_layer is not None:
        encoding["input_last_layer"]["chunksizes"] = (sample_chunk, num_last_layer_channel, height, width)
    encoding["target"]["chunksizes"] = (sample_chunk, num_target_channel, height, width)
    ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)
