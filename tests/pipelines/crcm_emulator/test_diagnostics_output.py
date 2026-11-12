import numpy as np
import torch
import xarray

from resoterre.pipelines.crcm_emulator.diagnostics_output import metric_figure_spec, save_diagnostics


MAPPING = {"bias_map": {"figure_type": "nd_plot", "figure_kwargs": {"reverse_i": True}}}


def test_metric_figure_spec():
    assert metric_figure_spec("tas/bias_map_historical", MAPPING) == MAPPING["bias_map"]
    assert metric_figure_spec("tas/bias_map", MAPPING) == MAPPING["bias_map"]
    assert metric_figure_spec("tas/rmse_historical", MAPPING) is None


def test_save_diagnostics(tmp_path):
    bias = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    results = {
        "tas/rmse_historical": 1.5,
        "tas/bias_map_historical": bias,
        "pr/other_map_historical": bias + 1,
    }
    paths = save_diagnostics(results, tmp_path, "exp_validation", MAPPING)

    assert sorted(path.name for path in paths) == [
        "exp_validation_diagnostics.nc",
        "exp_validation_tas_bias_map_historical.png",
    ]
    with xarray.open_dataset(tmp_path / "exp_validation_diagnostics.nc", engine="h5netcdf") as dataset:
        assert set(dataset.data_vars) == {"tas_bias_map_historical", "pr_other_map_historical"}
        np.testing.assert_array_equal(dataset["tas_bias_map_historical"].values, bias.numpy())


def test_save_diagnostics_without_maps(tmp_path):
    assert save_diagnostics({"tas/rmse_historical": 1.5}, tmp_path, "exp", MAPPING) == []
    assert not list(tmp_path.iterdir())
