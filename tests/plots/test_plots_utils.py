import matplotlib.pyplot as plt
import numpy as np
import pytest

from resoterre.plots import utils


def test_figure_dispatcher_failure():
    with pytest.raises(ValueError):
        utils.figure_dispatcher(
            figure_type="invalid",
            data=np.random.rand(10, 10),
            title="Test Figure",
        )


def test_figure_dispatcher():
    fig = utils.figure_dispatcher(
        figure_type="nd_plot", data=np.random.rand(10, 10), title="Test Figure", figure_kwargs={"reverse_i": True}
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
