import numpy as np
import torch
import xarray

from resoterre.config import BasePlotConfig
from resoterre.config.metrics import BiasMapConfig
from resoterre.pipelines.crcm_emulator.diagnostics_output import (
    diagnostic_title,
    save_diagnostics,
    variable_units,
)


bias_map_config = BiasMapConfig(plot=BasePlotConfig(figure_type="nd_plot", figure_kwargs={"reverse_i": True}))
MAPPING = {
    "bias_map": bias_map_config,
    "tas/bias_map": bias_map_config,
    "tas/bias_map_historical": bias_map_config,
    "tas/bias_map_future": bias_map_config,
}


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
        assert dataset["tas_bias_map_historical"].attrs["units"] == "K"
        assert dataset["pr_other_map_historical"].attrs["units"] == variable_units("pr/x")


def test_variable_units():
    assert variable_units("tas/bias_map_historical") == "K"
    assert variable_units("unknown_variable/bias_map") is None


def test_save_diagnostics_without_maps(tmp_path):
    assert save_diagnostics({"tas/rmse_historical": 1.5}, tmp_path, "exp", MAPPING) == []
    assert not list(tmp_path.iterdir())


def test_diagnostic_title():
    assert diagnostic_title("tas/bias_map_historical") == "tas/bias_map_historical [K]"
    assert diagnostic_title("unknown_variable/bias_map") == "unknown_variable/bias_map"
