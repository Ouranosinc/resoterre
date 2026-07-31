"""Module for geospatial utilities."""

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry import Polygon

from resoterre.type_checker_utils import type_check_slice


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


class Tile:
    """
    Class representing a tile within a grid, defined by its center and corner indices.

    Parameters
    ----------
    i_center : int, optional
        Row index of the tile's center.
    j_center : int, optional
        Column index of the tile's center.
    i_start : int, optional
        Row index of the tile's starting corner.
    j_start : int, optional
        Column index of the tile's starting corner.
    i_end : int, optional
        Row index of the tile's ending corner.
    j_end : int, optional
        Column index of the tile's ending corner.
    """

    def __init__(
        self,
        i_center: int | None = None,
        j_center: int | None = None,
        i_start: int | None = None,
        j_start: int | None = None,
        i_end: int | None = None,
        j_end: int | None = None,
    ) -> None:
        self.i_center: int | None = i_center
        self.j_center: int | None = j_center
        self.i_start: int | None = i_start
        self.j_start: int | None = j_start
        self.i_end: int | None = i_end
        self.j_end: int | None = j_end
        self.i_slice: slice | None = None
        self.j_slice: slice | None = None
        self.lon: np.ndarray | None = None
        self.lat: np.ndarray | None = None
        self.lon_corners: np.ndarray | None = None
        self.lat_corners: np.ndarray | None = None

        if self.i_start is not None and self.i_end is not None:
            self.i_slice = slice(self.i_start, self.i_end)
        if self.j_start is not None and self.j_end is not None:
            self.j_slice = slice(self.j_start, self.j_end)

    def compute_corners(self, original_grid_lon: np.ndarray, original_grid_lat: np.ndarray) -> None:
        """
        Compute the corner coordinates of the tile based on the original grid's longitude and latitude.

        Parameters
        ----------
        original_grid_lon : np.ndarray
            2D array of longitudes for the original grid.
        original_grid_lat : np.ndarray
            2D array of latitudes for the original grid.
        """
        if self.i_start is None or self.i_end is None or self.j_start is None or self.j_end is None:
            raise ValueError("Tile indices must be set before computing corners")
        i_start = self.i_start - 1
        i_end = self.i_end + 1
        j_start = self.j_start - 1
        j_end = self.j_end + 1
        if i_start < 0 or i_end > original_grid_lon.shape[0] or j_start < 0 or j_end > original_grid_lon.shape[1]:
            raise ValueError("Cannot compute corners for tile without a 1-cell padding buffer within its grid")
        lon_tile = original_grid_lon[i_start:i_end, j_start:j_end]
        lat_tile = original_grid_lat[i_start:i_end, j_start:j_end]
        self.lon_corners = (lon_tile[:-1, :-1] + lon_tile[1:, :-1] + lon_tile[:-1, 1:] + lon_tile[1:, 1:]) / 4
        self.lat_corners = (lat_tile[:-1, :-1] + lat_tile[1:, :-1] + lat_tile[:-1, 1:] + lat_tile[1:, 1:]) / 4


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

    def __init__(self, lon: np.ndarray, lat: np.ndarray) -> None:
        self.lon = lon
        self.lat = lat
        self.active_tile: str | None = None
        self.tiles: dict[str, Tile] = {}

        self._set_inner_tile()

    def _set_inner_tile(self) -> None:
        """Set the inner tile, which excludes the boundary cells."""
        self.tiles["inner"] = Tile(i_start=1, j_start=1, i_end=self.lon.shape[0] - 1, j_end=self.lon.shape[1] - 1)
        self.tiles["inner"].compute_corners(original_grid_lon=self.lon, original_grid_lat=self.lat)

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
        if key in self.tiles:
            raise ValueError(f"Tile with key {key} already exists")
        if len(self.lon.shape) != 2:
            raise ValueError("Only 2D lon and lat are supported for now")
        if len(self.lat.shape) != 2:
            raise ValueError("Only 2D lon and lat are supported for now")
        tile_half_size = tile_size // 2
        distances = haversine(self.lon, self.lat, tile_center_lon, tile_center_lat)
        if isinstance(distances, float):
            raise ValueError("Distances should be a numpy array, not a float")
        indices = np.unravel_index(np.argmin(distances), distances.shape)
        self.tiles[key] = Tile(
            i_center=indices[0],
            j_center=indices[1],
            i_start=max(indices[0] - tile_half_size, 0),
            j_start=max(indices[1] - tile_half_size, 0),
            i_end=min(indices[0] + tile_half_size, self.lon.shape[0]),
            j_end=min(indices[1] + tile_half_size, self.lon.shape[1]),
        )
        self.tiles[key].lon = self.lon[self.tiles[key].i_slice, self.tiles[key].j_slice]
        self.tiles[key].lat = self.lat[self.tiles[key].i_slice, self.tiles[key].j_slice]
        if compute_corners:
            self.tiles[key].compute_corners(original_grid_lon=self.lon, original_grid_lat=self.lat)

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
        lon_corners = self.tiles[key].lon_corners
        lat_corners = self.tiles[key].lat_corners
        if lon_corners is None or lat_corners is None:
            raise ValueError(f"Tile {key} must have computed corners to coarsen")
        coarse_tile = Tile()
        coarse_tile.lon_corners = lon_corners[::factor, ::factor]
        coarse_tile.lat_corners = lat_corners[::factor, ::factor]
        coarse_tile.lon = (
            coarse_tile.lon_corners[:-1, :-1]
            + coarse_tile.lon_corners[1:, :-1]
            + coarse_tile.lon_corners[:-1, 1:]
            + coarse_tile.lon_corners[1:, 1:]
        ) / 4
        coarse_tile.lat = (
            coarse_tile.lat_corners[:-1, :-1]
            + coarse_tile.lat_corners[1:, :-1]
            + coarse_tile.lat_corners[:-1, 1:]
            + coarse_tile.lat_corners[1:, 1:]
        ) / 4
        self.tiles[key_coarse] = coarse_tile

    def _get_active_tile_property(self, property_name: str) -> Any:
        """
        Get a property of the active tile.

        Parameters
        ----------
        property_name : str
            Name of the property to retrieve.

        Returns
        -------
        Any
            The value of the requested property for the active tile.
        """
        if self.active_tile is None:
            raise ValueError("No active tile set")
        value = getattr(self.tiles[self.active_tile], property_name)
        if value is None:
            raise ValueError(f"Property {property_name} is not set for the active tile")
        return value

    @property
    def i_slice(self) -> slice:
        """
        The i slice of the active tile.

        Returns
        -------
        slice
            The i slice of the active tile.
        """
        return type_check_slice(self._get_active_tile_property("i_slice"))

    @property
    def j_slice(self) -> slice:
        """
        The j slice of the active tile.

        Returns
        -------
        slice
            The j slice of the active tile.
        """
        return type_check_slice(self._get_active_tile_property("j_slice"))

    @property
    def tile_lon(self) -> np.ndarray:
        """
        Longitude array of the active tile.

        Returns
        -------
        np.ndarray
            The longitude array of the active tile.
        """
        return self._get_active_tile_property("lon")

    @property
    def tile_lat(self) -> np.ndarray:
        """
        Latitude array of the active tile.

        Returns
        -------
        np.ndarray
            The latitude array of the active tile.
        """
        return self._get_active_tile_property("lat")

    @property
    def tile_lon_corners(self) -> np.ndarray:
        """
        Longitude corners of the active tile.

        Returns
        -------
        np.ndarray
            The longitude corners of the active tile.
        """
        return self._get_active_tile_property("lon_corners")

    @property
    def tile_lat_corners(self) -> np.ndarray:
        """
        Latitude corners of the active tile.

        Returns
        -------
        np.ndarray
            The latitude corners of the active tile.
        """
        return self._get_active_tile_property("lat_corners")


def add_neighbors(
    candidates: list[tuple[int, int]],
    i: int,
    j: int,
    i_shape: int,
    j_shape: int,
    excludes: set[tuple[int, int]] | None = None,
) -> None:
    """
    Add the neighbors of a given point (i, j) to the candidates list (inplace), excluding those in the excludes set.

    Parameters
    ----------
    candidates : list[tuple[int, int]]
        List of candidate points to which neighbors will be added.
    i : int
        Row index of the point.
    j : int
        Column index of the point.
    i_shape : int
        Total number of rows in the grid.
    j_shape : int
        Total number of columns in the grid.
    excludes : set[tuple[int, int]], optional
        Set of points to be excluded from being added to candidates.
    """
    excludes = excludes or set()
    if i > 0 and (i - 1, j) not in excludes and (i - 1, j) not in candidates:
        candidates.append((i - 1, j))
    if i < i_shape - 1 and (i + 1, j) not in excludes and (i + 1, j) not in candidates:
        candidates.append((i + 1, j))
    if j > 0 and (i, j - 1) not in excludes and (i, j - 1) not in candidates:
        candidates.append((i, j - 1))
    if j < j_shape - 1 and (i, j + 1) not in excludes and (i, j + 1) not in candidates:
        candidates.append((i, j + 1))


def compute_grids_area_weights(
    source_grid_lon: np.ndarray, source_grid_lat: np.ndarray, target_grid_spec: GridSpecification
) -> np.ndarray:
    """
    Compute the area weights for regridding from a source grid to a target grid.

    Parameters
    ----------
    source_grid_lon : np.ndarray
        2D array of longitudes for the source grid.
    source_grid_lat : np.ndarray
        2D array of latitudes for the source grid.
    target_grid_spec : GridSpecification
        GridSpecification object for the target grid.

    Returns
    -------
    np.ndarray
        Sparse matrix of area weights for regridding.
    """
    source_lon_corners = (
        source_grid_lon[:-1, :-1] + source_grid_lon[1:, :-1] + source_grid_lon[:-1, 1:] + source_grid_lon[1:, 1:]
    ) / 4
    # Assuming longitudes range match in both grids
    source_lon_corners = source_lon_corners
    source_lat_corners = (
        source_grid_lat[:-1, :-1] + source_grid_lat[1:, :-1] + source_grid_lat[:-1, 1:] + source_grid_lat[1:, 1:]
    ) / 4
    # Assuming longitudes range match in both grids
    source_grid_lon = source_grid_lon[1:-1, 1:-1]
    source_grid_lat = source_grid_lat[1:-1, 1:-1]
    rdps_index = []
    hrdps_index = []
    fractions = []
    for i in range(target_grid_spec.tile_lon.shape[0]):
        for j in range(target_grid_spec.tile_lon.shape[1]):
            target_lon = target_grid_spec.tile_lon[i, j]
            target_lat = target_grid_spec.tile_lat[i, j]
            target_polygon = Polygon(
                [
                    (target_grid_spec.tile_lon_corners[i, j], target_grid_spec.tile_lat_corners[i, j]),
                    (target_grid_spec.tile_lon_corners[i + 1, j], target_grid_spec.tile_lat_corners[i + 1, j]),
                    (target_grid_spec.tile_lon_corners[i + 1, j + 1], target_grid_spec.tile_lat_corners[i + 1, j + 1]),
                    (target_grid_spec.tile_lon_corners[i, j + 1], target_grid_spec.tile_lat_corners[i, j + 1]),
                ]
            )

            distances = haversine(source_grid_lon, source_grid_lat, target_lon, target_lat)
            if isinstance(distances, float):
                raise RuntimeError("Distances should be a numpy array, not a float")
            source_grid_closest_indices = np.where(distances == distances.min())
            i_search = source_grid_closest_indices[0][0]
            j_search = source_grid_closest_indices[1][0]
            candidates = [(i_search, j_search)]
            processed: set[tuple[int, int]] = set()
            add_neighbors(
                candidates, i_search, j_search, source_grid_lon.shape[0], source_grid_lon.shape[1], excludes=processed
            )
            while candidates:
                candidate = candidates.pop(0)
                source_polygon = Polygon(
                    [
                        (
                            source_lon_corners[candidate[0], candidate[1]],
                            source_lat_corners[candidate[0], candidate[1]],
                        ),
                        (
                            source_lon_corners[candidate[0] + 1, candidate[1]],
                            source_lat_corners[candidate[0] + 1, candidate[1]],
                        ),
                        (
                            source_lon_corners[candidate[0] + 1, candidate[1] + 1],
                            source_lat_corners[candidate[0] + 1, candidate[1] + 1],
                        ),
                        (
                            source_lon_corners[candidate[0], candidate[1] + 1],
                            source_lat_corners[candidate[0], candidate[1] + 1],
                        ),
                    ]
                )
                intersection = target_polygon.intersection(source_polygon)
                fraction = intersection.area / target_polygon.area
                processed.add(candidate)
                if fraction > 0:
                    rdps_index.append(np.ravel_multi_index(candidate, source_grid_lon.shape))
                    hrdps_index.append(np.ravel_multi_index((i, j), target_grid_spec.tile_lon.shape))
                    fractions.append(fraction)
                    add_neighbors(
                        candidates,
                        candidate[0],
                        candidate[1],
                        source_grid_lon.shape[0],
                        source_grid_lon.shape[1],
                        excludes=processed,
                    )
    return coo_matrix(
        (fractions, (hrdps_index, rdps_index)), shape=(target_grid_spec.tile_lon.size, source_grid_lon.size)
    )
