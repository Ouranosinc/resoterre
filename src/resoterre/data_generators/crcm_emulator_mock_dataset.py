"""Module for generating mock CRCM emulator data."""

import logging

import numpy as np
from torch.utils import data as td


logger = logging.getLogger(__name__)


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
            "CO2": 1.0,
            "CH4": 1.0,
            "N2O": 1.0,
            "CFC12": 1.0,
            "CFC11_eq": 1.0,
        }

        return {
            "input_first_block": np.random.rand(self.num_input_channels, self.coarse_size, self.coarse_size).astype(
                np.float32
            ),
            "target": np.random.rand(self.num_output_channels, self.tile_size, self.tile_size).astype(np.float32),
            "year": np.array(2024),
            "month": np.array(6),
            "day": np.array(6),
            **emission_data,
        }
