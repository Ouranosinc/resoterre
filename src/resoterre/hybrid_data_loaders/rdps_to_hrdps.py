"""Utilities for the RDPS to HRDPS tasks."""

import numpy as np
import torch

from resoterre.datasets.hrdps.hrdps_variables import hrdps_variables
from resoterre.ml.data_loader_utils import inverse_normalize


def post_process_model_output(output: torch.Tensor, variable_names: list[str]) -> dict[str, np.ndarray]:
    """
    Post-process the model output by inverse normalizing each variable.

    Parameters
    ----------
    output : torch.Tensor
        The model output tensor of shape (batch_size, num_variables, height, width).
    variable_names : list[str]
        List of variable names corresponding to the output channels.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary mapping variable names to their inverse normalized numpy arrays.
    """
    output_variables = {}
    for i in range(output.shape[1]):
        variable_name = str(variable_names[i])
        normalize_min_local = hrdps_variables[variable_name].normalize_min
        if normalize_min_local is None:
            raise ValueError(f"Variable {variable_name} does not have normalization parameters defined.")
        normalize_min: float = normalize_min_local
        normalize_max_local = hrdps_variables[variable_name].normalize_max
        if normalize_max_local is None:
            raise ValueError(f"Variable {variable_name} does not have normalization parameters defined.")
        normalize_max: float = normalize_max_local
        # ToDo: guarantee positive precipitation after inverse normalization?
        output_variables[variable_name] = inverse_normalize(
            output[:, i : i + 1, :, :].cpu().detach().numpy(),
            known_min=normalize_min,
            known_max=normalize_max,
            mode=(-1, 1),
            log_normalize=hrdps_variables[variable_name].log_normalize,
            log_offset=hrdps_variables[variable_name].normalize_log_offset,
        )
    # ToDo: climatology addition, latest run does not use climatologies for outputs...
    return output_variables
