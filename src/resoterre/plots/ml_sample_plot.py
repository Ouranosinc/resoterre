"""Module for machine learning data sample plotting."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from resoterre.plots.nd_plots import nd_ax_plot
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


def num_free_dimensions(
    dict_arrays: dict[str, np.ndarray],
    first_dimension_is_batch_size: bool = True,
    num_trailing_dimensions: int = 2,
) -> int:
    """
    Calculate the number of free dimensions in the provided arrays.

    Parameters
    ----------
    dict_arrays : dict[str, np.ndarray]
        Dictionary of arrays to analyze.
    first_dimension_is_batch_size : bool
        Indicates whether the first dimension of the arrays represents the batch size.
    num_trailing_dimensions : int
        Number of trailing dimensions to exclude from the count.

    Returns
    -------
    int
        Number of free dimensions in the provided arrays.
    """
    n = 0
    for data_array in dict_arrays.values():
        if first_dimension_is_batch_size:
            n += int(np.prod(data_array.shape[1:-num_trailing_dimensions]))
        else:
            n += int(np.prod(data_array.shape[:-num_trailing_dimensions]))
    return n


def ml_sample_figure(
    path_output: Path,
    count: int,
    input_data: np.ndarray | dict[str, np.ndarray],
    target_plot_data: np.ndarray,
    output_plot_data: np.ndarray,
    output_label: str,
    first_dimension_is_batch_size: bool = True,
) -> None:
    """
    Generate and save a sample figure for machine learning data.

    Parameters
    ----------
    path_output : Path
        Path to the output directory where the figure will be saved.
    count : int
        Current iteration count for naming the output file.
    input_data : np.ndarray | dict[str, np.ndarray]
        Input data for plotting. Can be a NumPy array or a dictionary of arrays.
    target_plot_data : np.ndarray
        Target data for plotting.
    output_plot_data : np.ndarray
        Output data for plotting.
    output_label : str
        Label for the output data.
    first_dimension_is_batch_size : bool
        Indicates whether the first dimension of the arrays represents the batch size.
    """
    if isinstance(input_data, np.ndarray):
        input_data = {"input": input_data}
    num_inputs = num_free_dimensions(
        input_data, first_dimension_is_batch_size=first_dimension_is_batch_size, num_trailing_dimensions=2
    )
    inputs_side = int(np.ceil(np.sqrt(num_inputs)))

    fig = plt.figure(figsize=(36, 12), layout="constrained")
    gs = fig.add_gridspec(nrows=1, ncols=3)
    gs_left = gs[0, 0].subgridspec(inputs_side, inputs_side)
    k = 0
    for input_array in input_data.values():
        if first_dimension_is_batch_size:
            input_array = input_array[0, ...]
        for input_idx in np.ndindex(input_array.shape[:-2]):
            i, j = np.unravel_index(k, (inputs_side, inputs_side))
            ax_left = fig.add_subplot(gs_left[i, j])
            nd_ax_plot(ax_left, fig, input_array[input_idx], title="", vmin=-1, vmax=1, reverse_i=True)
            k += 1
    ax_mid = fig.add_subplot(gs[0, 1])
    nd_ax_plot(ax_mid, fig, output_plot_data, title="Output", vmin=-1, vmax=1, reverse_i=True)
    ax_right = fig.add_subplot(gs[0, 2])
    nd_ax_plot(ax_right, fig, target_plot_data, title="Target", vmin=-1, vmax=1, reverse_i=True)
    plt.savefig(Path(path_output, f"training_{count:07d}_{output_label}.png"), bbox_inches="tight")
    plt.close(fig)


def ml_sample_figures(
    path_output: Path,
    count: int,
    input_data: np.ndarray | dict[str, np.ndarray],
    target_data: np.ndarray | dict[str, np.ndarray],
    output_data: np.ndarray | dict[str, np.ndarray],
    output_labels: list[str] | None = None,
    scheduler: ActionScheduler | None = None,
    first_dimension_is_batch_size: bool = True,
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
    output_labels : list[str], optional
        List of labels for the output data.
    scheduler : ActionScheduler, optional
        An optional scheduler to control when to generate the figures.
    first_dimension_is_batch_size : bool
        Indicates whether the first dimension of the arrays represents the batch size.
    """
    if scheduler is not None and not scheduler(count):
        return
    if isinstance(target_data, np.ndarray):
        target_data = {"default": target_data}
    if isinstance(output_data, np.ndarray):
        output_data = {"default": output_data}
    if sorted(list(output_data.keys())) != sorted(list(target_data.keys())):
        raise ValueError(
            f"Output data keys {list(output_data.keys())} do not match target data keys {list(target_data.keys())}."
        )
    num_targets = num_free_dimensions(
        target_data, first_dimension_is_batch_size=first_dimension_is_batch_size, num_trailing_dimensions=2
    )
    num_outputs = num_free_dimensions(
        output_data, first_dimension_is_batch_size=first_dimension_is_batch_size, num_trailing_dimensions=2
    )
    if num_targets != num_outputs:
        raise ValueError(f"Number of targets ({num_targets}) does not match number of outputs ({num_outputs}).")
    if output_labels is None:
        output_labels = list(map(str, range(num_outputs)))

    k = 0
    for output_key, output_array in output_data.items():
        if first_dimension_is_batch_size:
            output_array = output_array[0, ...]
        for output_idx in np.ndindex(output_array.shape[:-2]):
            output_plot_data = output_array[output_idx]
            target_array = target_data[output_key]
            if first_dimension_is_batch_size:
                target_array = target_array[0, ...]
            ml_sample_figure(
                path_output=path_output,
                count=count,
                input_data=input_data,
                target_plot_data=target_array[output_idx],
                output_plot_data=output_plot_data,
                output_label=output_labels[k],
                first_dimension_is_batch_size=first_dimension_is_batch_size,
            )
            k += 1


def balanced_ml_sample_figures(
    path_figure: Path,
    input_data: np.ndarray,
    target_data: np.ndarray,
    output_data: np.ndarray,
) -> None:
    """
    Generate and save balanced sample figures for machine learning data.

    Parameters
    ----------
    path_figure : Path
        Path to the output figure file where the figure will be saved.
    input_data : np.ndarray
        The input data for the model.
    target_data : np.ndarray
        The ground truth target data.
    output_data : np.ndarray
        The model output data.
    """
    n_inputs = input_data.shape[0]
    n_targets = target_data.shape[0]
    n_outputs = output_data.shape[0]

    # Find ideal placement of subplots
    category_counts = [n_inputs, n_targets, n_outputs]
    total_subplots = sum(category_counts)
    if total_subplots <= 0:
        best_rows = 1
        best_cols_per_category = [1, 1, 1]
    else:
        best_score = None
        best_rows = 1
        best_cols_per_category = [max(1, count) for count in category_counts]

        for n_rows in range(1, total_subplots + 1):
            cols_per_category = [max(1, int(np.ceil(count / n_rows))) for count in category_counts]
            n_cols = sum(cols_per_category)
            area = n_rows * n_cols
            empty_cells = area - total_subplots
            aspect_penalty = total_subplots * abs(math.log(n_rows / n_cols))
            score = empty_cells + aspect_penalty

            candidate = (score, empty_cells, area, abs(n_rows - n_cols))
            if best_score is None or candidate < best_score:
                best_score = candidate
                best_rows = n_rows
                best_cols_per_category = cols_per_category

    n_cols = sum(best_cols_per_category)
    fig, axes = plt.subplots(best_rows, n_cols, figsize=(4 * n_cols, 4 * best_rows), squeeze=False)

    # Start with all axes hidden, then activate only those mapped to actual channels.
    for row_axes in axes:
        for ax in row_axes:
            ax.axis("off")

    categories = [
        ("Input", input_data, n_inputs),
        ("Target", target_data, n_targets),
        ("Output", output_data, n_outputs),
    ]
    col_offset = 0
    for category_idx, (title, numpy_data, n_channels) in enumerate(categories):
        block_cols = best_cols_per_category[category_idx]
        for channel_idx in range(n_channels):
            row = channel_idx % best_rows
            col_in_block = channel_idx // best_rows
            col = col_offset + col_in_block
            ax = axes[row, col]
            ax.axis("on")
            ax.set_title(title)
            ax.pcolormesh(
                numpy_data[channel_idx, :, :],
                vmin=-1,
                vmax=1,
            )
        col_offset += block_cols
    plt.tight_layout()
    Path(path_figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_figure)
    plt.close(fig)
