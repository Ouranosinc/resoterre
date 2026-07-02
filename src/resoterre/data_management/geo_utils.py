"""Module for geospatial utilities."""

import numpy as np


def haversine(
    lon1: float | np.ndarray, lat1: float | np.ndarray, lon2: float | np.ndarray, lat2: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate the great circle distance in kilometers between two points on the earth (specified in decimal degrees).

    Parameters
    ----------
    lon1 : float | np.ndarray
        Longitude of the first point in decimal degrees.
    lat1 : float | np.ndarray
        Latitude of the first point in decimal degrees.
    lon2 : float | np.ndarray
        Longitude of the second point in decimal degrees.
    lat2 : float | np.ndarray
        Latitude of the second point in decimal degrees.

    Returns
    -------
    float | np.ndarray
        Distances between the two points in kilometers.

    Notes
    -----
    Only one pair of coordinates can be multi-dimensional.
    """
    # convert decimal degrees to radians
    lon1_radians = np.radians(lon1)
    lat1_radians = np.radians(lat1)
    lon2_radians = np.radians(lon2)
    lat2_radians = np.radians(lat2)

    # haversine formula
    lon_difference = lon2_radians - lon1_radians
    lat_difference = lat2_radians - lat1_radians
    a = np.sin(lat_difference / 2) ** 2 + np.cos(lat1_radians) * np.cos(lat2_radians) * np.sin(lon_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


class GridSpecification:
    """
    Class to specify a grid and its sub-tiles.

    Parameters
    ----------
    lon : np.ndarray
        2D array of longitudes.
    lat : np.ndarray
        2D array of latitudes.
    """

    def __init__(self, lon: np.array, lat: np.array) -> None:
        self.lon = lon
        self.lat = lat
        self.active_tile: str | None = None
        self.tiles_i_center: dict[str, int] = {}
        self.tiles_j_center: dict[str, int] = {}
        self.tiles_i_start: dict[str, int] = {}
        self.tiles_j_start: dict[str, int] = {}
        self.tiles_i_end: dict[str, int] = {}
        self.tiles_j_end: dict[str, int] = {}
        self.tiles_i_slice: dict[str, slice] = {}
        self.tiles_j_slice: dict[str, slice] = {}
        self.tiles_lon: dict[str, np.ndarray] = {}
        self.tiles_lat: dict[str, np.ndarray] = {}
        self.tiles_lon_corners: dict[str, np.ndarray] = {}
        self.tiles_lat_corners: dict[str, np.ndarray] = {}

    def _compute_slices(self, key: str) -> None:
        """
        Compute the slices for the tile specified by key.

        Parameters
        ----------
        key : str
            Key of the tile to compute slices for.
        """
        self.tiles_i_slice[key] = slice(self.tiles_i_start[key], self.tiles_i_end[key])
        self.tiles_j_slice[key] = slice(self.tiles_j_start[key], self.tiles_j_end[key])

    def _compute_corners(self, key: str) -> None:
        """
        Compute the corners for the tile specified by key.

        Parameters
        ----------
        key : str
            Key of the tile to compute corners for.
        """
        i_start = self.tiles_i_start[key] - 1
        if i_start < 0:
            raise ValueError("Cannot compute corners for tile with i_start < 1")
        i_end = self.tiles_i_end[key] + 1
        if i_end > self.lon.shape[0]:
            raise ValueError("Cannot compute corners for tile with i_end > lon.shape[0] - 1")
        j_start = self.tiles_j_start[key] - 1
        if j_start < 0:
            raise ValueError("Cannot compute corners for tile with j_start < 1")
        j_end = self.tiles_j_end[key] + 1
        if j_end > self.lon.shape[1]:
            raise ValueError("Cannot compute corners for tile with j_end > lon.shape[1] - 1")
        lon_tile = self.lon[i_start:i_end, j_start:j_end]
        lat_tile = self.lat[i_start:i_end, j_start:j_end]
        lon_corners_mean = (lon_tile[:-1, :-1] + lon_tile[1:, :-1] + lon_tile[:-1, 1:] + lon_tile[1:, 1:]) / 4
        lat_corners_mean = (lat_tile[:-1, :-1] + lat_tile[1:, :-1] + lat_tile[:-1, 1:] + lat_tile[1:, 1:]) / 4
        self.tiles_lon_corners[key] = lon_corners_mean
        self.tiles_lat_corners[key] = lat_corners_mean

    def _set_inner_tile(self) -> None:
        """Set the inner tile, which excludes the boundary cells."""
        key = "inner"
        self.tiles_i_start[key] = 1
        self.tiles_j_start[key] = 1
        self.tiles_i_end[key] = self.lon.shape[0] - 1
        self.tiles_j_end[key] = self.lon.shape[1] - 1
        self._compute_slices(key)
        self._compute_corners(key)

    def sub_tile(
        self,
        key: str,
        tile_center_lon: float,
        tile_center_lat: float,
        tile_size: int,
        compute_corners: bool = True,
        set_to_active: bool = False,
    ) -> None:
        """
        Create a sub-tile of the grid.

        Parameters
        ----------
        key : str
            Key to identify the sub-tile.
        tile_center_lon : float
            Longitude of the center of the sub-tile.
        tile_center_lat : float
            Latitude of the center of the sub-tile.
        tile_size : int
            Size of the sub-tile (number of grid points in each dimension).
        compute_corners : bool, optional
            Whether to compute the corners of the sub-tile.
        set_to_active : bool, optional
            Whether to set the sub-tile as the active tile.
        """
        if key in self.tiles_i_center:
            raise ValueError(f"Tile with key {key} already exists")
        if len(self.lon.shape) != 2:
            raise ValueError("Only 2D lon and lat are supported for now")
        if len(self.lat.shape) != 2:
            raise ValueError("Only 2D lon and lat are supported for now")
        tile_half_size = tile_size // 2
        distances = haversine(self.lon, self.lat, tile_center_lon, tile_center_lat)
        if isinstance(distances, float):
            raise ValueError("Distances should be a numpy array, not a float")
        indices = np.where(distances == distances.min())
        self.tiles_i_center[key] = indices[0][0]
        self.tiles_j_center[key] = indices[1][0]
        self.tiles_i_start[key] = max(self.tiles_i_center[key] - tile_half_size, 0)
        self.tiles_j_start[key] = max(self.tiles_j_center[key] - tile_half_size, 0)
        self.tiles_i_end[key] = min(self.tiles_i_center[key] + tile_half_size, self.lon.shape[0])
        self.tiles_j_end[key] = min(self.tiles_j_center[key] + tile_half_size, self.lon.shape[1])
        self._compute_slices(key)
        self.tiles_lon[key] = self.lon[self.tiles_i_slice[key], self.tiles_j_slice[key]]
        self.tiles_lat[key] = self.lat[self.tiles_i_slice[key], self.tiles_j_slice[key]]
        if compute_corners:
            self._compute_corners(key)

        if set_to_active:
            self.active_tile = key

    def coarsen_tile(self, key: str, key_coarse: str, factor: int) -> None:
        """
        Coarsen a tile by a given factor.

        Parameters
        ----------
        key : str
            Key of the tile to be coarsened.
        key_coarse : str
            Key of the resulting coarsened tile.
        factor : int
            Coarsening factor.
        """
        self.tiles_lon_corners[key_coarse] = self.tiles_lon_corners[key][::factor, ::factor]
        self.tiles_lat_corners[key_coarse] = self.tiles_lat_corners[key][::factor, ::factor]
        self.tiles_lon[key_coarse] = (
            self.tiles_lon_corners[key_coarse][:-1, :-1]
            + self.tiles_lon_corners[key_coarse][1:, :-1]
            + self.tiles_lon_corners[key_coarse][:-1, 1:]
            + self.tiles_lon_corners[key_coarse][1:, 1:]
        ) / 4
        self.tiles_lat[key_coarse] = (
            self.tiles_lat_corners[key_coarse][:-1, :-1]
            + self.tiles_lat_corners[key_coarse][1:, :-1]
            + self.tiles_lat_corners[key_coarse][:-1, 1:]
            + self.tiles_lat_corners[key_coarse][1:, 1:]
        ) / 4

    @property
    def i_slice(self) -> slice:
        """
        The i slice of the active tile.

        Returns
        -------
        slice
            The i slice of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_i_slice[self.active_tile]

    @property
    def j_slice(self) -> slice:
        """
        The j slice of the active tile.

        Returns
        -------
        slice
            The j slice of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_j_slice[self.active_tile]

    @property
    def tile_lon(self) -> np.ndarray:
        """
        Longitude array of the active tile.

        Returns
        -------
        np.ndarray
            The longitude array of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_lon[self.active_tile]

    @property
    def tile_lat(self) -> np.ndarray:
        """
        Latitude array of the active tile.

        Returns
        -------
        np.ndarray
            The latitude array of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_lat[self.active_tile]

    @property
    def tile_lon_corners(self) -> np.ndarray:
        """
        Longitude corners of the active tile.

        Returns
        -------
        np.ndarray
            The longitude corners of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_lon_corners[self.active_tile]

    @property
    def tile_lat_corners(self) -> np.ndarray:
        """
        Latitude corners of the active tile.

        Returns
        -------
        np.ndarray
            The latitude corners of the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        return self.tiles_lat_corners[self.active_tile]
