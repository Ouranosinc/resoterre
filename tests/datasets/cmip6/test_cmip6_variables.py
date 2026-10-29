import pytest

from resoterre.datasets.cmip6 import cmip6_variables


def test_import_values_from_csv():
    assert cmip6_variables.cmip6_variables["tas"].min == pytest.approx(202.70, abs=0.1)
    assert cmip6_variables.cmip6_variables["ua500"].max == pytest.approx(81.00, abs=0.1)
    assert cmip6_variables.cmip6_variables["va1000"].max is None


def test_scalar_variables_span_all_simulations():
    co2 = cmip6_variables.cmip6_variables["CO2"]
    assert co2.normalize_min == pytest.approx(397.55, abs=0.01)
    assert co2.normalize_max == pytest.approx(399.95, abs=0.01)
