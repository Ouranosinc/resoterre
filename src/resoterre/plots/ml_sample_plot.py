"""Module for machine learning data sample plotting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from resoterre.utils import ActionScheduler


@dataclass(frozen=True, slots=True)
class MachineLearningDataPlotConfig:
    """
    Configuration for machine learning data plotting.

    Attributes
    ----------
    dimensions_selected_index : dict[str, int | str]
        Dictionary specifying the selected index for each dimension.
    zoom_collection : dict[str, Any]
        Dictionary representation of a zoom collection.
    local_variance_max_sample_kwargs : dict[str, Any]
        Keyword arguments for local variance maximum sample calculation.
    figure_size_scale_factor : int
        Scale factor for the figure size.
    """

    dimensions_selected_index: dict[str, int | str] = field(default_factory=dict)
    zoom_collection: dict[str, Any] = field(default_factory=dict)
    local_variance_max_sample_kwargs: dict[str, Any] = field(default_factory=dict)
    figure_size_scale_factor: int = 1


def ml_sample_figures(
    path_output: Path,
    count: int,
    input_data: np.ndarray | dict[str, np.ndarray],
    target_data: np.ndarray | dict[str, np.ndarray],
    output_data: np.ndarray | dict[str, np.ndarray],
    scheduler: ActionScheduler | None = None,
) -> None:
    """
    Generate and save sample figures for machine learning data.

    Parameters
    ----------
    path_output : Path
        Path to the output directory where the figures will be saved.
    count : int
        Current iteration count for naming the output files.
    input_data : np.ndarray | dict[str, np.ndarray]
        Input data for plotting. Can be a NumPy array or a dictionary of arrays.
    target_data : np.ndarray | dict[str, np.ndarray]
        Target data for plotting. Can be a NumPy array or a dictionary of arrays.
    output_data : np.ndarray | dict[str, np.ndarray]
        Output data for plotting. Can be a NumPy array or a dictionary of arrays.
    scheduler : ActionScheduler, optional
        An optional scheduler to control when to generate the figures.
    """
    # ToDo: Output multiple figures when the inputs, targets and outputs contain multiple fields
    if scheduler is None or scheduler(count):
        if not isinstance(input_data, np.ndarray):
            raise NotImplementedError("Currently only supports input_data as np.ndarray.")
        if not isinstance(target_data, np.ndarray):
            raise NotImplementedError("Currently only supports target_data as np.ndarray.")
        if not isinstance(output_data, np.ndarray):
            raise NotImplementedError("Currently only supports output_data as np.ndarray.")
        n_rows = input_data.shape[0]
        n_cols = 3
        fig = plt.figure(figsize=(12, 4 * n_rows))
        for i in range(n_rows):
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1)
            ax1.set_title("Input")
            ax1.pcolormesh(input_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2)
            ax2.set_title("Target")
            ax2.pcolormesh(target_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3)
            ax3.set_title("Output")
            ax3.pcolormesh(output_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(Path(path_output, f"training_visualization_{count:06d}.png"))
        plt.close(fig)
