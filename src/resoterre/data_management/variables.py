"""Module for handling variable metadata and collections."""

degree_celsius_units = "°C"


class VariableHandler:
    """
    Physical variable definition and metadata.

    Parameters
    ----------
    name : str
        The name of the variable.
    units : str
        The units of the variable.
    netcdf_key : str | None, optional
        The key used in NetCDF files for this variable. Defaults to the variable name.
    target_cf_units : str | None, optional
        The CF-compliant units for the variable if it were to be converted to standard CF units.
    vertical_level : int | float | None, optional
        The vertical level (e.g., depth or height) at which the variable is measured.
    vertical_level_units : str | None, optional
        The units of the vertical level.
    min_value : float | None, optional
        The minimum valid value for the variable.
    max_value : float | None, optional
        The maximum valid value for the variable.
    mean_min : float | None, optional
        The minimum mean value threshold for the variable.
    mean_max : float | None, optional
        The maximum mean value threshold for the variable.
    clip_min : float | None, optional
        The minimum clipping value for the variable.
    clip_max : float | None, optional
        The maximum clipping value for the variable.
    nan_min : float | None, optional
        The minimum value below which data is considered NaN.
    nan_max : float | None, optional
        The maximum value above which data is considered NaN.
    cumulative : bool, default=False
        Whether the variable represents cumulative data.
    log_normalize : bool, default=False
        Whether to apply logarithmic normalization to the variable.
    normalize_min : float | None, optional
        The minimum value for normalization.
    normalize_max : float | None, optional
        The maximum value for normalization.
    normalize_log_offset : float | None, optional
        The offset used in logarithmic normalization to avoid log(0).
        Should be slightly smaller than the smallest non-zero value.
    """

    def __init__(
        self,
        name: str,
        units: str,
        netcdf_key: str | None = None,
        target_cf_units: str | None = None,
        vertical_level: int | float | None = None,
        vertical_level_units: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        mean_min: float | None = None,
        mean_max: float | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        nan_min: float | None = None,
        nan_max: float | None = None,
        cumulative: bool = False,
        log_normalize: bool = False,
        normalize_min: float | None = None,
        normalize_max: float | None = None,
        normalize_log_offset: float | None = None,
    ):
        self.name = name
        self.units = units
        self.netcdf_key = netcdf_key or name
        self.target_cf_units = target_cf_units
        self.vertical_level = vertical_level
        if (vertical_level is not None) and (vertical_level_units is None):
            raise ValueError("If vertical_level is specified, vertical_level_units must also be specified.")
        self.vertical_level_units = vertical_level_units
        self.min = min_value
        self.max = max_value
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.nan_min = nan_min
        self.nan_max = nan_max
        self.cumulative = cumulative
        self.log_normalize = log_normalize
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.normalize_log_offset = normalize_log_offset

    def has_mean_thresholds(self) -> bool:
        """
        Check if mean thresholds are defined.

        Returns
        -------
        bool
            True if either mean_min or mean_max is defined, False otherwise.
        """
        if (self.mean_min is not None) or (self.mean_max is not None):
            return True
        return False

    def has_clip_thresholds(self) -> bool:
        """
        Check if clipping thresholds are defined.

        Returns
        -------
        bool
            True if either clip_min or clip_max is defined, False otherwise.
        """
        if (self.clip_min is not None) or (self.clip_max is not None):
            return True
        return False

    def has_nan_thresholds(self) -> bool:
        """
        Check if NaN thresholds are defined.

        Returns
        -------
        bool
            True if either nan_min or nan_max is defined, False otherwise.
        """
        if (self.nan_min is not None) or (self.nan_max is not None):
            return True
        return False


class VariableHandlerCollection(dict[str, VariableHandler]):
    """
    Initialize a VariableHandlerCollection instance.

    Parameters
    ----------
    d : dict[str, VariableHandler] | None, optional
        A dictionary to initialize the collection with.
    mapping : dict[str, str] | None, optional
        A mapping of alternative keys to the actual keys in the collection.
    """

    def __init__(self, d: dict[str, VariableHandler] | None = None, mapping: dict[str, str] | None = None):
        d = d or {}
        super().__init__(d)
        self.mapping = mapping or {}

    def __getitem__(self, key: str) -> VariableHandler:
        """
        Get a VariableHandler by key, using mapping if necessary.

        Parameters
        ----------
        key : str
            The key to look up.

        Returns
        -------
        VariableHandler
            The corresponding VariableHandler instance.
        """
        if key in self:
            value = super().__getitem__(key)
            if not isinstance(value, VariableHandler):
                raise RuntimeError("Value is not a VariableHandler instance.")
            return value
        elif key in self.mapping:
            value = super().__getitem__(self.mapping[key])
            if not isinstance(value, VariableHandler):
                raise RuntimeError("Mapped value is not a VariableHandler instance.")
            return value
        raise KeyError(key)
