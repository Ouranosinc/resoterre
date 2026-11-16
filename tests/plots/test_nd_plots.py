import tempfile
from pathlib import Path

import numpy as np

from resoterre.plots import nd_plots


def test_nd_save_plot_single_value():
    with tempfile.TemporaryDirectory() as tmp_dir:
        figure_file = Path(tmp_dir, "test_plot.png")
        plot_data = np.random.rand(1)
        nd_plots.nd_save_plot(figure_file=figure_file, plot_data=plot_data, title="Test")
        assert figure_file.is_file()


def test_nd_save_plot_1d():
    with tempfile.TemporaryDirectory() as tmp_dir:
        figure_file = Path(tmp_dir, "test_plot.png")
        plot_data = np.random.rand(10)
        nd_plots.nd_save_plot(figure_file=figure_file, plot_data=plot_data, title="Test")
        assert figure_file.is_file()


def test_nd_save_plot_2d():
    with tempfile.TemporaryDirectory() as tmp_dir:
        figure_file = Path(tmp_dir, "test_plot.png")
        plot_data = np.random.rand(10, 10)
        nd_plots.nd_save_plot(figure_file=figure_file, plot_data=plot_data, title="Test")
        assert figure_file.is_file()


def test_nd_plot_2d_symmetric_without_range():
    plot_data = np.array([[-1.0, 0.0], [0.5, 3.0]])
    figure = nd_plots.nd_plot(plot_data, "Test", cmap="RdBu_r", symmetric=True, show_range=False)
    image = figure.axes[0].images[0]
    assert image.get_clim() == (-3.0, 3.0)
    assert image.get_cmap().name == "RdBu_r"
    assert figure.axes[0].get_title() == "Test (2, 2)"
