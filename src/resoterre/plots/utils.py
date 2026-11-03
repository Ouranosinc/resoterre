"""General plotting utilities for Resoterre."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from resoterre.plots.nd_plots import nd_plot


FIGURE_FUNCTIONS = {
    "nd_plot": nd_plot,
}


def figure_dispatcher(
    figure_type: str,
    data: np.ndarray,
    title: str,
    figure_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """
    Dispatch the appropriate figure creation function based on the figure type.

    Parameters
    ----------
    figure_type : str
        The type of figure to create.
    data : np.ndarray
        The data to be plotted.
    title : str
        The title of the figure.
    figure_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to the figure creation function.

    Returns
    -------
    plt.Figure
        The created figure.
    """
    try:
        plot_fn = FIGURE_FUNCTIONS[figure_type]
    except KeyError:
        supported = ", ".join(FIGURE_FUNCTIONS.keys())
        raise ValueError(f"Unsupported figure type: {figure_type}. Supported types are: {supported}") from None
    return plot_fn(plot_data=data, title=title, **(figure_kwargs or {}))
