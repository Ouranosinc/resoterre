import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from resoterre.plots.ml_sample_plot import balanced_ml_sample_figure


def subplot_titles(figure: Figure) -> list[str]:
    return sorted(ax.get_title() for ax in figure.axes if ax.get_title())


def test_balanced_ml_sample_figure_channel_names():
    rng = np.random.default_rng(0)
    figure = balanced_ml_sample_figure(
        input_data=rng.random((4, 8, 8)),
        target_data=rng.random((1, 16, 16)),
        output_data=rng.random((1, 16, 16)),
        input_names=["zg850", "zg850 mask", "psl", "psl mask"],
        target_names=["tas"],
        output_names=["tas"],
    )
    plt.close(figure)
    assert subplot_titles(figure) == [
        "Input: psl",
        "Input: psl mask",
        "Input: zg850",
        "Input: zg850 mask",
        "Output: tas",
        "Target: tas",
    ]


def test_balanced_ml_sample_figure_without_names():
    rng = np.random.default_rng(0)
    figure = balanced_ml_sample_figure(
        input_data=rng.random((2, 8, 8)), target_data=rng.random((1, 8, 8)), output_data=rng.random((1, 8, 8))
    )
    plt.close(figure)
    assert subplot_titles(figure) == ["Input", "Input", "Output", "Target"]
