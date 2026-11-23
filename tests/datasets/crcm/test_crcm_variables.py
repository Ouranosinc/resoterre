import pytest

from resoterre.config.crcm_emulator import (
    DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE,
    DEFAULT_NORMALIZATION_STATS_SCENARIO,
)
from resoterre.datasets.crcm import crcm_variables


def test_configure_crcm_variables_uses_requested_scenario():
    combined = crcm_variables.crcm_variables["tas"]
    combined_range = (combined.normalize_min, combined.normalize_max)
    historical_stats = [
        row
        for row in crcm_variables.read_normalization_stats(DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE, "historical")
        if row["model"] == "crcm" and row["variable"] == "tas"
    ]
    try:
        crcm_variables.configure_crcm_variables("historical", DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE)
        assert crcm_variables.crcm_variables["tas"].normalize_min == pytest.approx(float(historical_stats[0]["min"]))
        assert crcm_variables.crcm_variables["tas"].normalize_max == pytest.approx(float(historical_stats[0]["max"]))
    finally:
        crcm_variables.configure_crcm_variables(
            DEFAULT_NORMALIZATION_STATS_SCENARIO, DEFAULT_CLIMATE_VARS_NORMALIZATION_STATS_FILE
        )
    restored = crcm_variables.crcm_variables["tas"]
    assert (restored.normalize_min, restored.normalize_max) == combined_range
