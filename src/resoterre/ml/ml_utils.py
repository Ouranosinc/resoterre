"""Utilities for machine learning."""

import logging

import torch


logger = logging.getLogger(__name__)


def log_device_info() -> None:
    """Log information about available CUDA devices."""
    device_str = ""
    for i in range(torch.cuda.device_count()):
        device_str += f"\n    cuda: {i} - {torch.cuda.get_device_name(i)}"
    if not device_str:
        device_str = "No cuda device found."
    else:
        device_str = "cuda devices available:" + device_str
    logger.info(device_str)
