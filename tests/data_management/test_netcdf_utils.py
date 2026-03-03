import numpy as np

from resoterre.data_management import netcdf_utils


def test_cf_variables():
    cf_variables = netcdf_utils.CFVariables()
    cf_variables.add(
        "tas",
        np.zeros((16, 8)),
        dims=("lat", "lon"),
        attributes={"units": "K"},
        dtype=np.float32,
        zlib=True,
        complevel=5,
        fill_value=-9999.0,
    )
    assert "tas" in cf_variables
