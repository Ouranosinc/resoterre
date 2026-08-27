"""Module for CMIP6 utilities."""

gcm_calendars = {
    "CanESM5": "noleap",
    "CNRM-ESM2-1": "standard",
    "ERA5": "standard",
    "MPI-ESM1-2-LR": "standard",
    "NorESM2-MM": "noleap",
}

gcm_vertical_variables = ["hus", "ta", "ua", "va", "zg"]
gcm_vertical_levels = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0, 10000.0, 5000.0, 1000.0]
