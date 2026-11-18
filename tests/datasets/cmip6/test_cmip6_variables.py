import pytest

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
    assert {row["model"] for row in cmip6_variables.csv_data} == {"gcm", "crcm", "forcing"}
    assert {row["scenario"] for row in cmip6_variables.csv_data} == {"historical+ssp245"}
