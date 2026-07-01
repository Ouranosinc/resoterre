"""Module for plotting n-dimensional data (1D, 2D, etc.) using Matplotlib."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from resoterre.logging_utils import readable_value


def nd_ax_plot(
    ax: plt.Axes,
    fig: plt.Figure,
    plot_data: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    reverse_i: bool = False,
) -> None:
    """
    Plot n-dimensional data on the given Axes.

    Parameters
    ----------
    ax : plt.Axes
        The Matplotlib Axes to plot on.
    fig : plt.Figure
        The Matplotlib Figure to which the Axes belongs.
    plot_data : np.ndarray
        The n-dimensional data to plot. Can be 0D (scalar), 1D (vector), or 2D (matrix).
    title : str
        The title for the plot.
    vmin : float, optional
        The minimum value for the color scale (only for 2D data).
    vmax : float, optional
        The maximum value for the color scale (only for 2D data).
    reverse_i : bool, optional
        Whether to reverse the order of the first dimension (only for 2D data).
    """
    plot_data = np.squeeze(plot_data)
    if len(plot_data.shape) == 0:
        ax.plot(plot_data, "ro")
        ax.set_yticks(plot_data[np.newaxis])
        ax.set_xticks([])
    elif len(plot_data.shape) == 1:
        ax.plot(plot_data, "ro")
        for x in plot_data:
            ax.axhline(y=x, color="black", linestyle=":", alpha=0.10)
    elif len(plot_data.shape) == 2:
        if reverse_i:
            plot_data = plot_data[::-1, :]
        im = ax.imshow(plot_data, cmap="viridis", vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5)
        _ = fig.colorbar(im, cax=cax, orientation="horizontal")
    else:
        raise NotImplementedError()
    if len(plot_data.shape) == 0:
        ax.set_title(title)
    else:
        ax.set_title(
            f"{title} {plot_data.shape} [{readable_value(plot_data.min())}, {readable_value(plot_data.max())}]"
        )


def nd_fig_plot(
    fig: plt.Figure,
    plot_data: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    reverse_i: bool = False,
) -> None:
    """
    Plot n-dimensional data on a new Axes in the given Figure.

    Parameters
    ----------
    fig : plt.Figure
        The Matplotlib Figure to plot on.
    plot_data : np.ndarray
        The n-dimensional data to plot. Can be 0D (scalar), 1D (vector), or 2D (matrix).
    title : str
        The title for the plot.
    vmin : float, optional
        The minimum value for the color scale (only for 2D data).
    vmax : float, optional
        The maximum value for the color scale (only for 2D data).
    reverse_i : bool, optional
        Whether to reverse the order of the first dimension (only for 2D data).
    """
    fig.set_size_inches(8, 8, forward=True)
    ax = fig.add_subplot(111)
    nd_ax_plot(ax, fig, plot_data, title=title, vmin=vmin, vmax=vmax, reverse_i=reverse_i)


def nd_save_plot(
    figure_file: Path | str,
    plot_data: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    reverse_i: bool = False,
) -> None:
    """
    Save a plot of n-dimensional data to a file.

    Parameters
    ----------
    figure_file : Path | str
        The file path to save the figure.
    plot_data : np.ndarray
        The n-dimensional data to plot. Can be 0D (scalar), 1D (vector), or 2D (matrix).
    title : str
        The title for the plot.
    vmin : float, optional
        The minimum value for the color scale (only for 2D data).
    vmax : float, optional
        The maximum value for the color scale (only for 2D data).
    reverse_i : bool, optional
        Whether to reverse the order of the first dimension (only for 2D data).
    """
    fig = plt.figure()
    nd_fig_plot(fig, plot_data, title, vmin=vmin, vmax=vmax, reverse_i=reverse_i)
    Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_file)
    plt.close(fig)


class NDPlot:
    """
    Shortcut for plotting n-dimensional data with predefined settings.

    Parameters
    ----------
    path_figures : Path | str, optional
        The directory where figures will be saved. If None, figures will not be saved.
    file_name : str
        The file name template for saving figures. Can include placeholders for string formatting.
    title : str
        The title template for the plot. Can include placeholders for string formatting.
    vmin : float, optional
        The minimum value for the color scale (only for 2D data).
    vmax : float, optional
        The maximum value for the color scale (only for 2D data).
    reverse_i : bool
        Whether to reverse the order of the first dimension (only for 2D data).
    force_underscores_in_file_name : bool
        Whether to replace spaces with underscores in the file name.
    """

    def __init__(
        self,
        path_figures: Path | str | None = None,
        file_name: str = "",
        title: str = "",
        vmin: float | None = None,
        vmax: float | None = None,
        reverse_i: bool = False,
        force_underscores_in_file_name: bool = True,
    ) -> None:
        self.path_figures = path_figures
        self.file_name = file_name
        self.title = title
        self.vmin = vmin
        self.vmax = vmax
        self.reverse_i = reverse_i
        self.force_underscores_in_file_name = force_underscores_in_file_name

    def __call__(
        self,
        plot_data: np.ndarray,
        file_name: str | None = None,
        title: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        reverse_i: bool | None = None,
        enabled: bool = True,
        strings_kwargs: dict[str, str] | None = None,
    ) -> None:
        """
        Plot the given n-dimensional data and saves the figure if enabled.

        Parameters
        ----------
        plot_data : np.ndarray
            The n-dimensional data to plot. Can be 0D (scalar), 1D (vector), or 2D (matrix).
        file_name : str, optional
            The file name template for saving the figure. Can include placeholders for string formatting.
        title : str, optional
            The title template for the plot. Can include placeholders for string formatting.
        vmin : float, optional
            The minimum value for the color scale (only for 2D data).
        vmax : float, optional
            The maximum value for the color scale (only for 2D data).
        reverse_i : bool, optional
            Whether to reverse the order of the first dimension (only for 2D data).
        enabled : bool, optional
            Whether to save the figure. If False, the function will return without saving.
        strings_kwargs : dict[str, str], optional
            A dictionary of string replacements for formatting the file name and title.
        """
        if not enabled:
            return
        strings_kwargs = strings_kwargs or {}
        local_file_name = file_name if file_name is not None else self.file_name
        local_file_name = local_file_name.format(**strings_kwargs)
        if (self.path_figures is None) and (local_file_name[0] != "/"):
            return
        if self.force_underscores_in_file_name:
            local_file_name = local_file_name.replace(" ", "_")
        local_title = title if title is not None else self.title
        local_title = local_title.format(**strings_kwargs)
        local_vmin = vmin if vmin is not None else self.vmin
        local_vmax = vmax if vmax is not None else self.vmax
        local_reverse_i = reverse_i if reverse_i is not None else self.reverse_i
        if self.path_figures is None:
            figure_file = Path(local_file_name)
        else:
            figure_file = Path(self.path_figures, local_file_name)
        nd_save_plot(
            figure_file=figure_file,
            plot_data=plot_data,
            title=local_title,
            vmin=local_vmin,
            vmax=local_vmax,
            reverse_i=local_reverse_i,
        )


class CustomPColorMesh:
    """
    Class for creating custom pseudocolor mesh plots with Matplotlib.

    Parameters
    ----------
    fig : plt.Figure, optional
        The Matplotlib Figure to plot on. If None, a new figure will be created.
    ax : plt.Axes, optional
        The Matplotlib Axes to plot on. If None, a new Axes will be created.
    scale_factor : float, optional
        A scaling factor for the figure size.
    show_colorbar : bool, optional
        Whether to display a colorbar.
    """

    def __init__(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        scale_factor: float = 1.0,
        show_colorbar: bool = True,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.cbar_ax: plt.Axes | None = None
        self.scale_factor = scale_factor
        self.show_colorbar = show_colorbar
        self.vmin: float | None = None
        self.vmax: float | None = None

    def init_figure(self, aspect_ratio: float | None = None) -> None:
        """
        Initialize the figure and axes for plotting.

        Parameters
        ----------
        aspect_ratio : float, optional
            The aspect ratio of the figure.
        """
        if aspect_ratio is None:
            width = 6.4 * self.scale_factor
            height = 4.8 * self.scale_factor
        else:
            width = 6.4 * aspect_ratio * self.scale_factor
            height = 6.4 * self.scale_factor
        if self.show_colorbar:
            width *= 1.05
        figure_size = (width, height)
        self.fig = plt.figure(figsize=figure_size)

    def plot(
        self,
        plot_data: np.ndarray,
        path_output: Path | str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_quantile: float | None = None,
        vmax_quantile: float | None = None,
        reset_ax: bool = True,
        reset_fig: bool = True,
        close_fig: bool = True,
    ) -> None:
        """
        Plot the given data as a pseudocolor mesh and optionally save the figure.

        Parameters
        ----------
        plot_data : np.ndarray
            The data to plot.
        path_output : Path | str, optional
            The file path to save the figure. If None, the figure will not be saved.
        cmap : str, optional
            The colormap to use for the plot.
        vmin : float, optional
            The minimum value for the color scale. If None, it will be determined from the data.
        vmax : float, optional
            The maximum value for the color scale. If None, it will be determined from the data.
        vmin_quantile : float, optional
            The quantile to use for determining the minimum value of the color scale. If None, it will not be used.
        vmax_quantile : float, optional
            The quantile to use for determining the maximum value of the color scale. If None, it will not be used.
        reset_ax : bool, optional
            Whether to reset the axes before plotting.
        reset_fig : bool, optional
            Whether to reset the figure before plotting.
        close_fig : bool, optional
            Whether to close the figure after plotting.
        """
        if self.ax is None or reset_ax:
            if self.fig is None or reset_fig:
                self.init_figure(aspect_ratio=plot_data.shape[1] / plot_data.shape[0])
            if self.fig is None:
                raise RuntimeError("Figure is not initialized.")
            if self.show_colorbar:
                gs = GridSpec(1, 2, width_ratios=[9.5, 0.5], figure=self.fig)
                self.ax = self.fig.add_subplot(gs[0])
                self.cbar_ax = self.fig.add_subplot(gs[1])
            else:
                self.ax = self.fig.add_subplot(111)
        if vmin_quantile is not None and vmin is None:
            vmin = np.nanquantile(plot_data, vmin_quantile)
        if vmax_quantile is not None and vmax is None:
            vmax = np.nanquantile(plot_data, vmax_quantile)
        self.vmin = vmin
        self.vmax = vmax
        pcolormesh_result = self.ax.pcolormesh(plot_data, cmap=cmap, vmin=self.vmin, vmax=self.vmax)
        min_str = readable_value(np.nanmin(plot_data))
        mean_str = readable_value(np.nanmean(plot_data))
        max_str = readable_value(np.nanmax(plot_data))
        self.ax.set_title(f"min: {min_str}, mean: {mean_str}, max: {max_str}")
        # ToDo: make colorbar in another ax with ScalarMappable?
        if self.fig is not None:
            if self.cbar_ax is None:
                self.fig.colorbar(pcolormesh_result, ax=self.ax)
            else:
                self.fig.colorbar(pcolormesh_result, cax=self.cbar_ax)
            if path_output is not None:
                path_output = Path(path_output)
                path_output.parent.mkdir(parents=True, exist_ok=True)
                self.fig.savefig(path_output)
            if close_fig:
                plt.close(self.fig)
