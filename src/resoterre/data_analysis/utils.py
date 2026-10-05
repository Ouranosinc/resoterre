"""Module for CRCM emulator data analysis utilities."""

import logging
import re
from pathlib import Path
from typing import Protocol

import pandas as pd
import xarray as xr


# Surface variables mapped onto the stem of their pressure-level counterpart, so that
# a surface RCM field can be paired with a pressure-level GCM field of the same family.
SURFACE_VARIABLE_STEMS = {"tas": "ta", "uas": "ua", "vas": "va", "huss": "hus", "ps": "ps"}


class EmulatorDatasetLike(Protocol):
    """Dataset surface used by the data-analysis helpers."""

    valid_idx: list[tuple[str, int, int]]
    gcm_variables: list[str]
    crcm_variables: list[str]

    def get_open_dataset(self, dataset_type: str, key: str) -> xr.Dataset:
        """
        Return an opened xarray.Dataset for the specified dataset type and key.

        Parameters
        ----------
        dataset_type : str
            The type of dataset (e.g., "gcm", "crcm").
        key : str
            The identifier (or simulation name) for which to retrieve the dataset.

        Returns
        -------
        xr.Dataset
            The opened xarray dataset.
        """
        ...


def write_dataframe_to_csv(df: pd.DataFrame, path: Path | str, name: str, logger: logging.Logger) -> None:
    """
    Write a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to write.
    path : Path | str
        Directory where the CSV file is written.
    name : str
        Name of the CSV file, without the ``.csv`` extension.
    logger : logging.Logger
        Logger for logging output.
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{name}.csv"
    df.to_csv(file_path, index=False)
    logger.info("Wrote dataframe to %s", file_path)


def group_indices_by_sim(
    dataset: EmulatorDatasetLike,
) -> dict[str, list[tuple[int, int]]]:
    """
    Convert dataset.valid_idx to a dictionary of simulation names and their corresponding GCM and CRCM time indices.

    Parameters
    ----------
    dataset : EmulatorDatasetLike
        Dataset object containing a ``valid_idx`` attribute.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Simulation names and their GCM and CRCM time index pairs.
    """
    pairs_by_sim: dict[str, list[tuple[int, int]]] = {}
    for sim, gcm_idx, crcm_idx in dataset.valid_idx:
        pairs_by_sim.setdefault(sim, []).append((gcm_idx, crcm_idx))
    return pairs_by_sim


def first_sample(
    sim: str,
    model: str,
    var: str,
    data_gcm: dict[str, xr.Dataset],
    data_crcm: dict[str, xr.Dataset],
) -> tuple[xr.DataArray, str]:
    """
    Return the first time sample of a variable for a simulation and model.

    Parameters
    ----------
    sim : str
        Simulation name.
    model : str
        Model name, ``gcm`` or ``crcm``.
    var : str
        Variable name.
    data_gcm : dict[str, xarray.Dataset]
        GCM data keyed by simulation name.
    data_crcm : dict[str, xarray.Dataset]
        CRCM data keyed by simulation name.

    Returns
    -------
    sample : xarray.DataArray
        Field at the first time step.
    date_label : str
        Date of the first sample as ``YYYY-MM-DD``.
    """
    ds = data_gcm[sim] if model == "gcm" else data_crcm[sim]
    sample = ds[var].isel(time=0)
    sample_time = ds["time"].isel(time=0).values

    # Extract just the date (YYYY-MM-DD) as a string
    if hasattr(sample_time, "strftime"):
        date_label = sample_time.strftime("%Y-%m-%d")
    else:
        # if sample_time is not a cftime object, convert to string and slice the date portion
        date_label = str(sample_time)[:10]
    return sample, date_label


def variable_pretty_label(var: str) -> tuple[str, str]:
    """
    Return a display name and variable code for colourbar labels.

    Parameters
    ----------
    var : str
        CMIP-style variable name.

    Returns
    -------
    label : str
        Human-readable name, including units when known.
    var_code : str
        Original variable code.
    """
    # Map variable codes to preferred names/levels
    pretty_names = {
        "zg850": ("Geopotential Height at 850 hPa (m)", "zg850"),
        "zg700": ("Geopotential Height at 700 hPa (m)", "zg700"),
        "zg500": ("Geopotential Height at 500 hPa (m)", "zg500"),
        "hus850": ("Specific Humidity at 850 hPa (kg/kg)", "hus850"),
        "hus700": ("Specific Humidity at 700 hPa (kg/kg)", "hus700"),
        "hus500": ("Specific Humidity at 500 hPa (kg/kg)", "hus500"),
        "ta850": ("Temperature at 850 hPa (K)", "ta850"),
        "ta700": ("Temperature at 700 hPa (K)", "ta700"),
        "ta500": ("Temperature at 500 hPa (K)", "ta500"),
        "uas": ("Eastward Surface Wind (m/s)", "uas"),
        "ua850": ("Eastward Wind at 850 hPa (m/s)", "ua850"),
        "ua700": ("Eastward Wind at 700 hPa (m/s)", "ua700"),
        "ua500": ("Eastward Wind at 500 hPa (m/s)", "ua500"),
        "vas": ("Northward Surface Wind (m/s)", "vas"),
        "va850": ("Northward Wind at 850 hPa (m/s)", "va850"),
        "va700": ("Northward Wind at 700 hPa (m/s)", "va700"),
        "va500": ("Northward Wind at 500 hPa (m/s)", "va500"),
        "psl": ("Sea Level Pressure (Pa)", "psl"),
    }
    return pretty_names.get(var, (f"{var}", var))


def variable_family(name: str) -> str:
    """
    Return the variable family used to pair GCM and RCM fields.

    Trailing digits are removed, so ``ta850`` becomes ``ta``. Surface names such as
    ``tas`` are mapped onto that same family.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str
        Family name used to pair GCM and RCM variables.
    """
    stem = re.sub(r"\d+$", "", str(name))
    return SURFACE_VARIABLE_STEMS.get(stem, stem)


def matching_rcm_variable(gcm_variable: str, rcm_variables: list[str]) -> str | None:
    """
    Return the RCM surface variable in the same family as ``gcm_variable``, if any.

    Parameters
    ----------
    gcm_variable : str
        GCM variable name.
    rcm_variables : list of str
        Candidate RCM variable names.

    Returns
    -------
    str or None
        Matching RCM variable, preferring an exact name then a surface field.
        ``None`` if no RCM variable shares the family.
    """
    # Get the family of the GCM variable
    family = variable_family(gcm_variable)
    # Find all RCM variables in the same family
    matches = [name for name in rcm_variables if variable_family(name) == family]
    # If no matches, return None
    if not matches:
        return None
    # If GCM variable is in the matches, return it
    if gcm_variable in matches:
        return gcm_variable
    # If no exact match, return the first surface variable in the family
    surface = [name for name in matches if name in SURFACE_VARIABLE_STEMS]
    return surface[0] if surface else matches[0]
