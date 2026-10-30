"""General plotting utilities for Resoterre."""

import matplotlib.pyplot as plt
import numpy as np

from resoterre.plots.nd_plots import nd_plot


# ToDo: add additional arguments or handle arbitrary figure kwargs?
def figure_dispatcher(figure_type: str, data: np.ndarray, title: str) -> plt.Figure:
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

    Returns
    -------
    plt.Figure
        The created figure.
    """
    if figure_type == "nd_plot":
        return nd_plot(plot_data=data, title=title)
    elif figure_type == "nd_plot_reverse_i":
        return nd_plot(plot_data=data, title=title, reverse_i=True)
    raise ValueError(f"Unsupported figure type: {figure_type}")
