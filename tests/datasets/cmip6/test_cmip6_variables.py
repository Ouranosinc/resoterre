import pytest

from resoterre.config.crcm_emulator import (
    DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
    DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE,
    DEFAULT_NORMALIZATION_STATS_SCENARIO,
)
from resoterre.datasets.cmip6 import cmip6_variables


def test_import_values_from_csv():
    assert cmip6_variables.cmip6_variables["tas"].min == pytest.approx(204.55, abs=0.1)
    assert cmip6_variables.cmip6_variables["ua500"].max == pytest.approx(81.00, abs=0.1)
    assert cmip6_variables.cmip6_variables["va1000"].max is None


def test_forcing_variables_use_min_max():
    co2 = cmip6_variables.cmip6_variables["CO2"]
    assert co2.normalize_min == pytest.approx(313.01, abs=0.01)
    assert co2.normalize_max == pytest.approx(561.43, abs=0.01)


def test_normalization_uses_gcm_rows_of_combined_scenario():
    tas = cmip6_variables.cmip6_variables["tas"]
    # The CRCM rows of the stats files have a different min (199.0) than the GCM rows (204.55).
    assert tas.min == pytest.approx(204.55, abs=0.01)
    assert tas.max == pytest.approx(316.46, abs=0.01)


def test_configure_cmip6_variables_uses_requested_scenario():
    combined_min = cmip6_variables.cmip6_variables["CO2"].normalize_min
    assert combined_min is not None
    try:
        cmip6_variables.configure_cmip6_variables(
            "ssp245", DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE, DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE
        )
        ssp245_min = cmip6_variables.cmip6_variables["CO2"].normalize_min
        assert ssp245_min is not None
        assert ssp245_min > combined_min
    finally:
        cmip6_variables.configure_cmip6_variables(
            DEFAULT_NORMALIZATION_STATS_SCENARIO,
            DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
            DEFAULT_FORCINGS_NORMALIZATION_STATS_FILE,
        )
    assert cmip6_variables.cmip6_variables["CO2"].normalize_min == combined_min
