"""Utilities for working with NetCDF files in compliance with CF conventions."""

from typing import Any

import numpy as np
import xarray


_notset = object()


class CFVariables(dict[str, xarray.Variable]):
    """Shortcut for creating and adding variables to an xarray Dataset."""

    def add(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...] | None = None,
        attributes: dict[str, str] | None = None,
        dtype: object = _notset,
        zlib: bool | object = _notset,
        complevel: int | object = _notset,
        fill_value: object = _notset,
    ) -> None:
        """
        Add a variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        data : np.ndarray
            Data for the variable.
        dims : tuple[str, ...], optional
            Dimensions of the variable. If None, use the variable name as dimension.
        attributes : dict[str, str], optional
            Attributes for the variable. If None, use an empty dictionary.
        dtype : object
            Data type for the variable. If not set, use the default data type.
        zlib : bool | object
            Whether to use zlib compression. If not set, do not specify.
        complevel : int | object
            Compression level (1-9). If not set, do not specify.
        fill_value : object
            Fill value for missing data. If not set, do not specify.
        """
        dims = dims or (name,)
        attributes = attributes or {}

        encoding = {}
        if dtype is not _notset:
            encoding["dtype"] = dtype
        if zlib is not _notset:
            encoding["zlib"] = zlib
        if complevel is not _notset:
            encoding["complevel"] = complevel
        if fill_value is not _notset:
            encoding["_FillValue"] = fill_value

        self[name] = xarray.Variable(dims=dims, data=data, attrs=attributes, encoding=encoding)


def add_xarray_variable(
    variable_dict: dict[str, xarray.Variable],
    variable_name: str,
    data: Any,
    encoding_dict: dict[str, Any] | None,
    dims: tuple[str, ...] | None = None,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Add a variable to the given dictionaries for xarray Dataset creation.

    Parameters
    ----------
    variable_dict : dict[str, xarray.Variable]
        Dictionary to store the xarray Variable objects.
    variable_name : str
        Name of the variable to add.
    data : Any
        Data for the variable.
    encoding_dict : dict[str, Any] | None
        Dictionary to store the encoding information for each variable.
    dims : tuple[str, ...], optional
        Dimensions of the variable. If None, use the variable name as dimension.
    attrs : dict[str, Any], optional
        Attributes for the variable. If None, use an empty dictionary.
    **kwargs : Any
        Additional keyword arguments for encoding (e.g., dtype, zlib, complevel, fill_value).

    Notes
    -----
    This function modifies the variable_dict and encoding_dict in place.
    """
    if dims is None:
        dims = (variable_name,)
    if encoding_dict is not None:
        encoding_dict[variable_name] = {k: v for k, v in kwargs.items()}
        variable_dict[variable_name] = xarray.Variable(dims=dims, data=data, attrs=attrs)
    else:
        variable_dict[variable_name] = xarray.Variable(
            dims=dims, data=data, attrs=attrs, encoding={k: v for k, v in kwargs.items()}
        )
