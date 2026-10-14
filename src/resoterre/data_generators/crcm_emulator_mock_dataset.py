"""Module for generating mock CRCM emulator data."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray
from torch.utils import data as td

from resoterre.datasets.crcm.crcm_utils import crcm_north_america_grid_coordinates


logger = logging.getLogger(__name__)


def mock_gcm_data(path_output: Path | str, allow_overwrite: bool = False) -> None:
    """
    Generate mock GCM data for the CRCM emulator.

    Parameters
    ----------
    path_output : Path | str
        Path to the output directory where mock data will be stored.
    allow_overwrite : bool, optional
        Whether to allow overwriting existing files, by default False.
    """
    Path(path_output, "CNRM-ESM2-1").mkdir(parents=True, exist_ok=True)
    # Currently only generating CNRM-ESM2-1 mock data
    nc_file = Path(path_output, "CNRM-ESM2-1", "tas_day_CNRM-ESM2-1_historical_r1i1p1f2_gr_18500101-20141231.nc")
    if allow_overwrite or not nc_file.is_file():
        # If GCMs with other calendars are required, move time logic to cftime_utils.py
        start_datetime = datetime(1850, 1, 1, 12)
        delta_t = timedelta(days=1)
        times = [start_datetime + i * delta_t for i in range(60625)]
        ds = xarray.Dataset(
            {
                "tas": (("time", "lat", "lon"), np.zeros((60625, 128, 256), dtype="float32")),
            },
            coords={
                "time": times,
                "lat": np.linspace(-88.927735, 88.927735, 128),
                "lon": np.linspace(0, 358.59375, 256),
            },
        )
        ds.to_netcdf(
            nc_file,
            engine="netcdf4",
            encoding={
                "tas": {
                    "zlib": True,
                    "complevel": 1,
                    "shuffle": False,
                    "dtype": "float32",
                    "chunksizes": (16, 128, 256),
                }
            },
        )
    nc_file = Path(path_output, "CNRM-ESM2-1", "tas_day_CNRM-ESM2-1_ssp245_r1i1p1f2_gr_20150101-21001231.nc")
    if allow_overwrite or not nc_file.is_file():
        start_datetime = datetime(2015, 1, 1, 12)
        delta_t = timedelta(days=1)
        times = [start_datetime + i * delta_t for i in range(31411)]
        ds = xarray.Dataset(
            {
                "tas": (("time", "lat", "lon"), np.zeros((31411, 128, 256), dtype="float32")),
            },
            coords={
                "time": times,
                "lat": np.linspace(-88.927735, 88.927735, 128),
                "lon": np.linspace(0, 358.59375, 256),
            },
        )
        ds.to_netcdf(
            nc_file,
            engine="h5netcdf",
            encoding={
                "tas": {
                    "zlib": True,
                    "complevel": 1,
                    "shuffle": False,
                    "dtype": "float32",
                    "chunksizes": (16, 128, 256),
                }
            },
        )


def mock_emissions(path_output: Path | str, allow_overwrite: bool = False) -> None:
    """
    Generate mock emissions data for the CRCM emulator.

    Parameters
    ----------
    path_output : Path | str
        Path to the output directory where mock emissions data will be stored.
    allow_overwrite : bool, optional
        Whether to allow overwriting existing files, by default False.
    """
    Path(path_output).mkdir(parents=True, exist_ok=True)
    path_ghg = Path(path_output, "greenhouse_gases_hist3.dat")
    if allow_overwrite or not path_ghg.is_file():
        historical_ghg = "A\nB\nC\nD\nE\nF\n"
        for year in range(1750, 2026):
            historical_ghg += f"{year} 0.0 0.0 0.0 0.0 0.0 0.0\n"
        path_ghg.write_text(historical_ghg)
    for ssp in ["126", "245", "370", "585"]:
        path_ssp = Path(path_output, f"GHG_SSP{ssp}.dat")
        if not allow_overwrite and path_ssp.is_file():
            continue
        ssp_ghg = "A\nB\nC\nD\nE\nF\n"
        for year in range(2015, 2501):
            ssp_ghg += f"{year} 0.0 0.0 0.0 0.0 0.0 0.0\n"
        path_ssp.write_text(ssp_ghg)


def mock_crcm_data(path_output: Path | str, allow_overwrite: bool = False) -> None:
    """
    Generate mock CRCM data for the CRCM emulator.

    Parameters
    ----------
    path_output : Path | str
        Path to the output directory where mock data will be stored.
    allow_overwrite : bool, optional
        Whether to allow overwriting existing files, by default False.
    """
    rlon, rlat, lon, lat = crcm_north_america_grid_coordinates()
    for scenario in ["historical", "ssp245"]:
        # Currently only generating subset of years around historical ssp dates
        if scenario == "historical":
            years = list(range(2001, 2012, 5))
            version = "v2-r1"
        else:
            years = [2015] + list(range(2016, 2022, 5))
            version = "v1-r1"
        path_data = Path(
            path_output, "CNRM-ESM2-1", scenario, "r1i1p1f2", "CRCM5-SN", version, "day", "tas", "v20260508"
        )
        path_data.mkdir(parents=True, exist_ok=True)
        # Currently only generating CNRM-ESM2-1 mock data
        for year in years:
            final_year = year + 4
            if year == 2011:
                final_year = 2014
            elif year == 2015:
                final_year = 2015
            nc_file = Path(
                path_data,
                f"tas_NAM-12_CNRM-ESM2-1_{scenario}_r1i1p1f2_OURANOS_CRCM5-SN_{version}_day_{year}0101-{final_year}1231.nc",
            )
            if allow_overwrite or not nc_file.is_file():
                # If GCMs with other calendars are required, move time logic to cftime_utils.py
                current_datetime = datetime(year, 1, 1, 12)
                times = []
                while current_datetime.year <= final_year:
                    times.append(current_datetime)
                    current_datetime += timedelta(days=1)
                ds = xarray.Dataset(
                    {
                        "tas": (("time", "rlat", "rlon"), np.zeros((len(times), 628, 655), dtype="float32")),
                    },
                    coords={
                        "time": times,
                        "rlat": rlat,
                        "rlon": rlon,
                        "lat": (("rlat", "rlon"), lat),
                        "lon": (("rlat", "rlon"), lon),
                    },
                )
                ds.to_netcdf(
                    nc_file,
                    engine="h5netcdf",
                    encoding={
                        "time": {"units": "days since 1950-01-01", "dtype": "float64"},
                        "tas": {
                            "zlib": True,
                            "complevel": 1,
                            "shuffle": False,
                            "dtype": "float32",
                            "chunksizes": (8, 628, 655),
                        },
                    },
                )


class CRCMEmulatorMockDataset(td.Dataset):  # type: ignore[misc]
    """
    Dataset for the CRCM emulator, mock data.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset.
    num_input_channels : int
        Number of input channels.
    num_output_channels : int
        Number of output channels.
    tile_size : int
        Size of the output tile.
    coarsen_factor : int
        Factor by which the input data is coarsened.
    """

    def __init__(
        self,
        num_samples: int,
        num_input_channels: int = 2,
        num_output_channels: int = 2,
        tile_size: int = 608,
        coarsen_factor: int = 8,
    ) -> None:
        self.num_samples = num_samples
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.tile_size = tile_size
        self.coarsen_factor = coarsen_factor
        self.coarse_size = tile_size // coarsen_factor

    def __len__(self) -> int:
        """
        Get the number of sample in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Get the input and target data for a given index.

        Parameters
        ----------
        idx : int
            Index of the data to retrieve.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the input data, target data, and associated metadata.
        """
        emission_data = {
            "CO2": np.array(1.0, dtype=np.float32),
            "CH4": np.array(1.0, dtype=np.float32),
            "N2O": np.array(1.0, dtype=np.float32),
            "CFC12": np.array(1.0, dtype=np.float32),
            "CFC11_eq": np.array(1.0, dtype=np.float32),
        }

        return {
            "input_first_block": np.random.rand(self.num_input_channels, self.coarse_size, self.coarse_size).astype(
                np.float32
            ),
            "target": np.random.rand(self.num_output_channels, self.tile_size, self.tile_size).astype(np.float32),
            "year": np.array(2024, dtype=np.float32),
            "month": np.array(6, dtype=np.float32),
            "day": np.array(6, dtype=np.float32),
            **emission_data,
        }
