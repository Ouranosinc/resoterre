"""Module for data input/output operations."""

import math


def sample_chunk_size(
    extra_dimensions_product: int,
    bytes_per_value: int = 4,
    target_chunk_mib: int = 8,
    min_chunk: int = 1,
    max_chunk: int = 1024,
) -> int:
    """
    Calculate a suitable chunk size for sampling data based on the target chunk size in MiB.

    Parameters
    ----------
    extra_dimensions_product : int
        The product of the sizes of the extra dimensions.
    bytes_per_value : int
        The number of bytes per data value (e.g. 4 for float32).
    target_chunk_mib : int
        The target chunk size in MiB (choose 4 to 16 MiB typically).
    min_chunk : int
        The minimum chunk size.
    max_chunk : int
        The maximum chunk size.

    Returns
    -------
    int
        The calculated chunk size for sampling.
    """
    chunk_size_target = target_chunk_mib * 1024 * 1024  # bytes
    s = chunk_size_target / (extra_dimensions_product * bytes_per_value)
    s_rounded = max(min_chunk, min(max_chunk, int(math.floor(s + 0.5))))
    return max(1, s_rounded)
