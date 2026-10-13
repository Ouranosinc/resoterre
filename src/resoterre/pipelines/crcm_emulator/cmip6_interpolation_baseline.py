"""CMIP6 Interpolation Baseline."""

from resoterre.pipelines.crcm_emulator.crcm_emulator_workflow import CRCMEmulatorConfig, crcm_emulator_parse_config


class CMIP6InterpolationBaselineFromConfig:
    """
    Class to handle CMIP6 Interpolation Baseline from a given configuration.

    Parameters
    ----------
    config : CRCMEmulatorConfig
        CRCM emulator configuration object.
    initialize_zarr : bool, optional
        Whether to initialize the zarr dataset.
    """

    def __init__(self, config: CRCMEmulatorConfig, initialize_zarr: bool = False) -> None:
        self.config = crcm_emulator_parse_config(config)
        if initialize_zarr:
            self._initialize_zarr()

    def _initialize_zarr(self) -> None:
        """Initialize the zarr dataset for each GCM simulation."""
        # Logic to initialize the zarr dataset goes here
        pass
