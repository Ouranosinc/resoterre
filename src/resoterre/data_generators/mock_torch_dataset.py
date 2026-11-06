"""Mock Torch Dataset and DataLoader for testing."""

import numpy as np
from torch.utils import data as td


class MockTorchDataset(td.Dataset):  # type: ignore[misc]
    """
    Mock Torch Dataset for testing.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset.
    input_shape : tuple
        Shape of the input data.
    target_shape : tuple
        Shape of the target data.
    """

    def __init__(self, num_samples: int, input_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> None:
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.target_shape = target_shape

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        Return a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the input and target arrays.
        """
        return {
            "input": np.random.rand(*self.input_shape).astype(np.float32),
            "target": np.random.rand(*self.target_shape).astype(np.float32),
        }


def mock_torch_dataloader(
    num_samples: int, input_shape: tuple[int, ...], target_shape: tuple[int, ...], batch_size: int
) -> td.DataLoader:
    """
    Mock Torch DataLoader for testing.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset.
    input_shape : tuple
        Shape of the input data.
    target_shape : tuple
        Shape of the target data.
    batch_size : int
        Batch size for the DataLoader.

    Returns
    -------
    td.DataLoader
        DataLoader for the mock dataset.
    """
    dataset = MockTorchDataset(num_samples, input_shape, target_shape)
    return td.DataLoader(dataset, batch_size=batch_size, shuffle=True)
