"""Utilities for working with NetCDF files in compliance with CF conventions."""

import numpy as np
import xarray


_notset = object()


class CFVariables(dict):
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
    ) -> None:
        """
        Add a variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        data : np.ndarray
            Data for the variable.
        dims : tuple[str, ...] | None
            Dimensions of the variable. If None, use the variable name as dimension.
        attributes : dict | None
            Attributes for the variable. If None, use an empty dictionary.
        dtype : object
            Data type for the variable. If not set, use the default data type.
        zlib : bool | object
            Whether to use zlib compression. If not set, do not specify.
        complevel : int | object
            Compression level (1-9). If not set, do not specify.
        """
        dims = dims or name
        attributes = attributes or {}

        encoding = {}
        if dtype is not _notset:
            encoding["dtype"] = dtype
        if zlib is not _notset:
            encoding["zlib"] = zlib
        if complevel is not _notset:
            encoding["complevel"] = complevel

        self[name] = xarray.Variable(dims=dims, data=data, attrs=attributes, encoding=encoding)
