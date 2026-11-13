import tempfile
from pathlib import Path

import torch

from resoterre.config.crcm_emulator import CRCMEmulatorPlotConfig
from resoterre.pipelines.crcm_emulator import metrics
from resoterre.plots.utils import figure_dispatcher


def test_bias_map_plot():
    metric = metrics.BiasMap()
    # ToDo: make values deterministic
    target = torch.randn(4, 2, 3, 3)  # Example target tensor with shape (batch, variables, height, width)
    output = torch.randn(4, 2, 3, 3)  # Example output tensor with shape (batch, variables, height, width)
    metric_result = metric(target, output)
    # This metric does not require consolidation
    plot_config = CRCMEmulatorPlotConfig()
    # ToDo: need to dispatch plots per variables
    figure = figure_dispatcher(
        figure_type=plot_config.metric_mapping[metric.name]["figure_type"],
        data=metric_result[0, :, :].numpy(),
        title=f"validation-{metric.name}",
        figure_kwargs=plot_config.metric_mapping[metric.name]["kwargs"],
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        figure.savefig(Path(tmp_dir, "bias_map.png"))
        pass  # Add breakpoint here to inspect the saved figure
