import numpy as np
import pytest

from resoterre.data_management import geo_utils


@pytest.fixture
def sample_grid():
    lon, lat = np.meshgrid(np.arange(-80, -70, 1.0), np.arange(40, 60, 1.0))
    return lon, lat


def test_haversine_distances():
    # Test the haversine function with known values
    lon1, lat1 = 0.0, 0.0
    lon2, lat2 = 0.0, 1.0
    distance = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert round(distance, 2) == 111.19  # Approximate distance in km for 1 degree latitude

    lon1, lat1 = -73.9857, 40.7484  # New York City
    lon2, lat2 = -0.1278, 51.5074  # London
    distance = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert round(distance) == 5566  # Approximate distance in km between NYC and London


def test_haversine_multidimensional():
    # Test the haversine function with multi-dimensional inputs
    lon1 = np.random.rand(5, 5) * 360 - 180  # Random longitudes
    lat1 = np.random.rand(5, 5) * 180 - 90  # Random latitudes
    lon2 = 0.0
    lat2 = 0.0
    distances = geo_utils.haversine(lon1, lat1, lon2, lat2)
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (5, 5)  # The output should have the same shape as the input arrays
    assert np.all(distances >= 0)  # Distances should be non-negative


def test_haversine_invalid_input():
    # Test the haversine function with invalid input shapes
    lon1 = np.random.rand(5, 5) * 360 - 180
    lat1 = np.random.rand(5, 5) * 180 - 90
    lon2 = np.random.rand(3, 3) * 360 - 180
    lat2 = np.random.rand(3, 3) * 180 - 90
    with pytest.raises(ValueError):
        geo_utils.haversine(lon1, lat1, lon2, lat2)


def test_tile_compute_corners():
    lon, lat = np.meshgrid(np.arange(-80, -76, 1.0), np.arange(40, 44, 1.0))
    tile = geo_utils.Tile(i_start=1, j_start=1, i_end=3, j_end=3)
    tile.compute_corners(original_grid_lon=lon, original_grid_lat=lat)
    assert tile.lon_corners is not None
    assert tile.lat_corners is not None
    assert tile.lon_corners.shape == (3, 3)
    assert tile.lat_corners.shape == (3, 3)
    assert tile.lon_corners[0, 0] == -79.5
    assert tile.lat_corners[2, 2] == 42.5


def test_grid_specification_init(sample_grid):
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)

    assert grid.lon.shape == (20, 10)
    assert "inner" in grid.tiles
    inner = grid.tiles["inner"]
    assert inner.i_start == 1
    assert inner.j_start == 1
    assert inner.i_end == 19
    assert inner.j_end == 9


def test_grid_specification_sub_tile(sample_grid):
    """Create sub-tile around known center values."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)

    # Center is at lon=-75, lat=50 (indices 10, 5)
    grid.sub_tile("test_tile", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=4, set_to_active=True)

    assert grid.active_tile == "test_tile"
    assert grid.tiles["test_tile"].i_center == 10
    assert grid.tiles["test_tile"].j_center == 5

    # With half size 2, slices should be 10-2:10+2 -> 8:12
    assert grid.i_slice == slice(8, 12)
    assert grid.j_slice == slice(3, 7)
    assert grid.tile_lon.shape == (4, 4)
    assert grid.tile_lon_corners.shape == (5, 5)


def test_grid_specification_sub_tile_boundary_clamping(sample_grid):
    """Ensure coordinate clipping on borders functions without indexing errors."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)

    # Edge tile near corner (0,0) - corners cannot be computed (raises ValueError)
    with pytest.raises(ValueError, match="padding buffer"):
        grid.sub_tile("boundary_tile", tile_center_lon=-80.0, tile_center_lat=40.0, tile_size=4, compute_corners=True)

    # Computing without corners should succeed and clamp correctly
    grid.sub_tile(
        "boundary_tile_no_corners", tile_center_lon=-80.0, tile_center_lat=40.0, tile_size=4, compute_corners=False
    )
    tile = grid.tiles["boundary_tile_no_corners"]
    assert tile.i_start == 0
    assert tile.j_start == 0


def test_grid_specification_tile_duplication_prevention(sample_grid):
    """Ensure custom keys cannot be overwritten."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)
    grid.sub_tile("tile1", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=2, compute_corners=False)
    with pytest.raises(ValueError, match="already exists"):
        grid.sub_tile("tile1", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=2, compute_corners=False)


def test_grid_specification_coarsen_tile(sample_grid):
    """Test coarsening logic on calculated corners."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)

    # Create larger sub-tile to handle a 2x coarsening
    grid.sub_tile("fine", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=6)
    grid.coarsen_tile("fine", "coarse", factor=2)

    fine_tile = grid.tiles["fine"]
    coarse_tile = grid.tiles["coarse"]

    # Original corners of size 6x6, coarsened by 2 should yield a 3x3 array
    assert fine_tile.lon_corners is not None
    assert fine_tile.lon_corners.shape == (7, 7)
    assert coarse_tile.lon_corners is not None
    assert coarse_tile.lon_corners.shape == (4, 4)
    # Computed centers of coarsened tile should be of size 2x2
    assert coarse_tile.lon is not None
    assert coarse_tile.lon.shape == (3, 3)


def test_grid_specification_coarsen_tile_missing_corners(sample_grid):
    """Verify coarsen failure states on missing properties."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)
    grid.sub_tile("no_corners", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=4, compute_corners=False)

    with pytest.raises(ValueError, match="must have computed corners"):
        grid.coarsen_tile("no_corners", "coarse", factor=2)


def test_grid_specification_active_tile_errors(sample_grid):
    """Verify error raises if trying to access properties before set active."""
    lon, lat = sample_grid
    grid = geo_utils.GridSpecification(lon, lat)

    with pytest.raises(ValueError, match="No active tile set"):
        _ = grid.tile_lon

    grid.sub_tile("active_sub", tile_center_lon=-75.0, tile_center_lat=50.0, tile_size=4, set_to_active=True)
    assert isinstance(grid.tile_lon, np.ndarray)
