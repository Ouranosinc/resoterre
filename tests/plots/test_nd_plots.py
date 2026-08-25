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
