degree_celsius_units = "°C"


class VariableHandler:
    def __init__(
        self,
        name,
        units,
        netcdf_key=None,
        target_cf_units=None,
        vertical_level=None,
        vertical_level_units=None,
        min_value=None,
        max_value=None,
        mean_min=None,
        mean_max=None,
        clip_min=None,
        clip_max=None,
        nan_min=None,
        nan_max=None,
        cumulative=False,
        log_normalize=False,
        normalize_min=None,
        normalize_max=None,
        normalize_log_offset=None,
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
        self.normalize_log_offset = normalize_log_offset  # Should be slightly smaller than the smallest non-zero value

    def has_mean_thresholds(self):
        if (self.mean_min is not None) or (self.mean_max is not None):
            return True
        return False

    def has_clip_thresholds(self):
        if (self.clip_min is not None) or (self.clip_max is not None):
            return True
        return False

    def has_nan_thresholds(self):
        if (self.nan_min is not None) or (self.nan_max is not None):
            return True
        return False


class VariableHandlerCollection(dict):
    def __init__(self, d=None, mapping=None):
        d = d or {}
        super().__init__(d)
        self.mapping = mapping or {}

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        elif key in self.mapping:
            return super().__getitem__(self.mapping[key])
        raise KeyError(key)
