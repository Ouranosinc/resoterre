"""Metrics configurations."""

import re
from typing import Any, ClassVar, Literal, Self

from pydantic import model_validator

from resoterre.config.base import BaseConfig, BasePlotConfig


class BaseMetricConfig(BaseConfig):
    """
    Base configuration for metrics.

    Attributes
    ----------
    name : str
        Name of the metric. This is used to resolve the appropriate subclass.
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : BasePlotConfig, optional
        Plot configuration for the metric.
    """

    name: str
    variables: tuple[str, ...] | None = None
    plot: BasePlotConfig | None = None

    _registry: ClassVar[dict[str, type["BaseMetricConfig"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass in the metric registry based on its name attribute.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to the subclass.
        """
        super().__init_subclass__(**kwargs)

        metric_name = cls.__dict__.get("name")
        if not isinstance(metric_name, str):
            raise ValueError(f"Metric subclass {cls.__name__} must define a default string value for 'name'.")
        BaseMetricConfig._registry[metric_name] = cls

    @model_validator(mode="wrap")  # type: ignore[untyped-decorator]
    @classmethod
    def _resolve_subclass(cls, value: Any, handler: Any) -> "BaseMetricConfig":
        """
        Resolve the appropriate subclass of BaseMetricConfig based on the "name" attribute in the input value.

        Parameters
        ----------
        value : Any
            The input value to validate.
        handler : Any
            The default validation handler to call if no subclass resolution is needed.

        Returns
        -------
        BaseMetricConfig
            An instance of the appropriate subclass of BaseMetricConfig.
        """
        # Only dispatch when validating against BaseMetricConfig directly with a dict
        if cls is BaseMetricConfig and isinstance(value, dict):
            metric_name: Any | None = value.get("name")
            target_cls = cls._registry.get(metric_name) if isinstance(metric_name, str) else None
            if target_cls is not None:
                return target_cls.model_validate(value)  # type: ignore[no-any-return]
        # Otherwise, proceed with normal validation for the target subclass
        return handler(value)  # type: ignore[no-any-return]

    def to_key(self) -> Self:
        """
        Return a copy of the metric configuration with the plot attribute set to None.

        Returns
        -------
        Self
            A copy of the metric configuration with the plot attribute set to None.

        Notes
        -----
        The resulting class needs to be hashable to be used as a key in dictionaries or sets.
        """
        return self.model_copy(update={"plot": None})  # type: ignore[no-any-return]

    def unique_string(self, target_variable: str | None = None) -> str:
        """
        Return a unique string representation of the metric configuration.

        Parameters
        ----------
        target_variable : str, optional
            Target variable to include in the unique string. Default is None.

        Returns
        -------
        str
            Unique string representation of the metric configuration.
        """
        if not hasattr(self, "name"):
            raise AttributeError("The metric config must have a 'name' attribute.")
        metric_str = str(self.name)
        for k, v in self.model_dump().items():
            if (k not in ["name", "plot", "variables"]) and v is not None:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if isinstance(v, float):
                        v = f"{v:.6g}"
                    v = str(v).replace("-", "neg").replace("+", "pos").replace(".", "p")
                    v = re.sub(r"[^a-zA-Z0-9_-]", "", v)
                metric_str += f"_{k}_{v}"
        if target_variable is not None:
            metric_str = f"{target_variable}/{metric_str}"
        return metric_str


class LossConfig(BaseMetricConfig):
    """
    Loss metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["loss"]
        Name of the loss metric.
    """

    name: Literal["loss"] = "loss"


class MSELossConfig(BaseMetricConfig):
    """
    Mean squared error loss metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["mse_loss"]
        Name of the mean squared error loss metric.
    """

    name: Literal["mse_loss"] = "mse_loss"


class SSIMLossConfig(BaseMetricConfig):
    """
    Structural similarity index loss metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["ssim_loss"]
        Name of the structural similarity index loss metric.
    """

    name: Literal["ssim_loss"] = "ssim_loss"


class RMSEConfig(BaseMetricConfig):
    """
    Root mean squared error metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["rmse"]
        Name of the root mean squared error metric.
    """

    name: Literal["rmse"] = "rmse"


class CCCConfig(BaseMetricConfig):
    """
    Concordance correlation coefficient metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["ccc"]
        Name of the concordance correlation coefficient metric.
    eps : float
        Small constant to avoid division by zero.
    """

    name: Literal["ccc"] = "ccc"
    eps: float = 1e-8


class WassersteinDistanceConfig(BaseMetricConfig):
    """
    Wasserstein distance metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["wasserstein_distance"]
        Name of the Wasserstein distance metric.
    """

    name: Literal["wasserstein_distance"] = "wasserstein_distance"


class BiasMapConfig(BaseMetricConfig):
    """
    Bias map metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["bias_map"]
        Name of the bias map metric.
    """

    name: Literal["bias_map"] = "bias_map"


class MeanBiasConfig(BaseMetricConfig):
    """
    Mean bias metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["mean_bias"]
        Name of the mean bias metric.
    """

    name: Literal["mean_bias"] = "mean_bias"


class TemporalQuantileBiasMapConfig(BaseMetricConfig):
    """
    Temporal quantile bias map metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["temporal_quantile_bias_map"]
        Name of the temporal quantile bias map metric.
    quantile : float
        Quantile to compute the metric for.
    """

    name: Literal["temporal_quantile_bias_map"] = "temporal_quantile_bias_map"
    quantile: float


class TemporalQuantileBiasConfig(BaseMetricConfig):
    """
    Temporal quantile bias metric configuration.

    Attributes
    ----------
    variables : tuple[str, ...], optional
        List of variables to compute the metric for. Default is all variables.
    plot : PlotConfig, optional
        Plot configuration for the metric.
    name : Literal["temporal_quantile_bias"]
        Name of the temporal quantile bias metric.
    quantile : float
        Quantile to compute the metric for.
    """

    name: Literal["temporal_quantile_bias"] = "temporal_quantile_bias"
    quantile: float
