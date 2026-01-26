"""Utilities for logging."""

import logging
import queue
import time
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any

from resoterre.utils import TemplateStore


logger = logging.getLogger(__name__)


def readable_delta_t(delta_t: int | float) -> str:
    """
    Return a human-readable string of a time delta.

    Parameters
    ----------
    delta_t : int | float
        Time delta in seconds.

    Returns
    -------
    str
        A convenient human-readable string of the time delta.
    """
    if delta_t < 0.001:
        return f"{round(delta_t * 1e6)} µs"
    elif delta_t < 1:
        return f"{round(delta_t * 1000)} ms"
    elif delta_t < 60:
        return f"{round(delta_t)} s"
    elif delta_t < 300:
        return f"{int(delta_t / 60)} min {round(delta_t % 60)} s"
    elif delta_t < 3600:
        return f"{round(delta_t / 60)} min"
    elif delta_t < 86400:
        return f"{int(delta_t / 3600)} h {round((delta_t % 3600) / 60)} min"
    else:
        return f"{int(delta_t / 86400)} d {round((delta_t % 86400) / 3600)} h"


def readable_value(value: int | float, expected_min: int | float = -1e38, expected_max: int | float = 1e38) -> str:
    """
    Return a human-readable string of a value.

    Parameters
    ----------
    value : int | float
        The value.
    expected_min : int | float
        The expected minimum value.
    expected_max : int | float
        The expected maximum value.

    Returns
    -------
    str
        A convenient human-readable string of the value.
    """
    if (value < expected_min) or (value > expected_max):
        return f"{value:.1e}"
    elif (int(value) == value) and abs(value) < 1000000:
        return f"{int(value)}"
    elif abs(value) < 0.01:
        return f"{value:.4e}"
    elif abs(value) < 100:
        return f"{value:.4f}"
    elif abs(value) < 10000:
        return f"{value:.2f}"
    elif abs(value) < 1000000:
        return f"{value:.0f}"
    else:
        return f"{value:.4e}"


class CustomLogging:
    """
    Custom logging class to handle logging with repetition overload protection.

    Parameters
    ----------
    caller : Any, optional
        The logger to use. Defaults to the root logger.
    queue : queue.Queue[tuple[str, str]], optional
        A queue to store log messages.
    quick_repetition_tolerance : int
        Number of quick repetitions allowed before blocking further messages.
    """

    def __init__(
        self,
        caller: Any = None,
        queue: queue.Queue[tuple[str, str]] | None = None,
        quick_repetition_tolerance: int = 1,
    ) -> None:
        if caller is None:
            self.caller = logging
        else:
            self.caller = caller
        self.queue = queue
        self.last_log_messages: dict[str, dict[str, Any]] = {}
        self.quick_repetition_tolerance = quick_repetition_tolerance
        self.quick_repetition_warning = False

    def log(
        self,
        caller: Callable[..., None],
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log a message with repetition overload protection.

        Parameters
        ----------
        caller : Callable[..., None]
            The logging method to use (e.g., logger.info).
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        if (block_short_repetition_delay or add_eta) and (identifier not in self.last_log_messages):
            self.last_log_messages[identifier] = {
                "count": 1,
                "first_time": time.time(),
                "last_time": 0,
                "quick_repetition_count": 0,
            }
        elif add_eta:
            self.last_log_messages[identifier]["count"] += 1
        if (not block_short_repetition_delay) and (identifier in self.last_log_messages):
            self.last_log_messages[identifier]["last_time"] = 0
            self.last_log_messages[identifier]["quick_repetition_count"] = 0
        elif identifier in self.last_log_messages:
            if time.time() - self.last_log_messages[identifier]["last_time"] < block_short_repetition_delay:
                if self.last_log_messages[identifier]["quick_repetition_count"] >= self.quick_repetition_tolerance:
                    if not self.quick_repetition_warning:
                        self.caller.warning("Some logging messages are being blocked due to short time repetition")
                        self.quick_repetition_warning = True
                    return False
        if add_eta:
            if self.last_log_messages[identifier]["count"] == 1:
                eta_time_str = "unknown"
            elif expected_nb_of_calls <= self.last_log_messages[identifier]["count"]:
                eta_time_str = "done"
            else:
                elapsed_time = time.time() - self.last_log_messages[identifier]["first_time"]
                time_per_call = elapsed_time / (self.last_log_messages[identifier]["count"] - 1)
                time_left = time_per_call * (expected_nb_of_calls - self.last_log_messages[identifier]["count"])
                eta_time_str = readable_delta_t(time_left)
            message += f" (ETA: {eta_time_str})"
        caller(message, *args, stacklevel=stacklevel + 3)
        if block_short_repetition_delay:
            self.last_log_messages[identifier]["last_time"] = time.time()
            self.last_log_messages[identifier]["quick_repetition_count"] += 1
        if self.queue is not None:
            self.queue.put((identifier, message))
        return True

    def debug(
        self,
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log a debug message.

        Parameters
        ----------
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        return self.log(
            self.caller.debug,
            message,
            *args,
            block_short_repetition_delay=block_short_repetition_delay,
            identifier=identifier,
            stacklevel=stacklevel,
            expected_nb_of_calls=expected_nb_of_calls,
            add_eta=add_eta,
        )

    def info(
        self,
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log an info message.

        Parameters
        ----------
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        return self.log(
            self.caller.info,
            message,
            *args,
            block_short_repetition_delay=block_short_repetition_delay,
            identifier=identifier,
            stacklevel=stacklevel,
            expected_nb_of_calls=expected_nb_of_calls,
            add_eta=add_eta,
        )

    def warning(
        self,
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log a warning message.

        Parameters
        ----------
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        return self.log(
            self.caller.warning,
            message,
            *args,
            block_short_repetition_delay=block_short_repetition_delay,
            identifier=identifier,
            stacklevel=stacklevel,
            expected_nb_of_calls=expected_nb_of_calls,
            add_eta=add_eta,
        )

    def error(
        self,
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log an error message.

        Parameters
        ----------
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        return self.log(
            self.caller.error,
            message,
            *args,
            block_short_repetition_delay=block_short_repetition_delay,
            identifier=identifier,
            stacklevel=stacklevel,
            expected_nb_of_calls=expected_nb_of_calls,
            add_eta=add_eta,
        )

    def critical(
        self,
        message: str,
        *args: Any,
        block_short_repetition_delay: int = 0,
        identifier: str = "",
        stacklevel: int = 0,
        expected_nb_of_calls: int = 1,
        add_eta: bool = False,
    ) -> bool:
        r"""
        Log a critical message.

        Parameters
        ----------
        message : str
            The message to log.
        \*args : Any
            Additional arguments for the logging method.
        block_short_repetition_delay : int
            Time in seconds to block repeated messages.
        identifier : str
            Unique identifier for the message.
        stacklevel : int
            Stack level for logging.
        expected_nb_of_calls : int
            Expected number of calls for ETA calculation.
        add_eta : bool
            Whether to add ETA information to the message.

        Returns
        -------
        bool
            True if the message was logged, False if it was blocked.
        """
        return self.log(
            self.caller.critical,
            message,
            *args,
            block_short_repetition_delay=block_short_repetition_delay,
            identifier=identifier,
            stacklevel=stacklevel,
            expected_nb_of_calls=expected_nb_of_calls,
            add_eta=add_eta,
        )


def default_basic_config_args(
    basic_config_args: dict[str, Any], show_logger_name: bool = False, show_date: bool = True
) -> dict[str, Any]:
    """
    Create a default set of basicConfig arguments for logging.

    Parameters
    ----------
    basic_config_args : dict[str, Any]
        Basic configuration arguments to customize.
    show_logger_name : bool
        Whether to include the logger name in the log format.
    show_date : bool
        Whether to include the date in the log format.

    Returns
    -------
    dict[str, Any]
        A dictionary of basicConfig arguments for logging.
    """
    basic_config_args_with_defaults = {k: v for k, v in basic_config_args.items()}
    if "format" not in basic_config_args_with_defaults:
        if show_logger_name:
            basic_config_args_with_defaults["format"] = (
                "%(asctime)s[%(levelname)s] (%(funcName)s, %(name)s) %(message)s"
            )
        else:
            basic_config_args_with_defaults["format"] = "%(asctime)s[%(levelname)s] (%(funcName)s) %(message)s"
    if "datefmt" not in basic_config_args_with_defaults:
        if show_date:
            basic_config_args_with_defaults["datefmt"] = "%Y-%m-%dT%H:%M:%S"
        else:
            basic_config_args_with_defaults["datefmt"] = "%H:%M:%S"
    if "level" not in basic_config_args_with_defaults:
        basic_config_args_with_defaults["level"] = logging.DEBUG
    if "force" not in basic_config_args_with_defaults:
        basic_config_args_with_defaults["force"] = True
    return basic_config_args_with_defaults


def start_root_logger(
    basic_config_args: dict[str, Any] | None = None,
    show_logger_name: bool = True,
    show_date: bool = True,
    show_loggers_on_init: bool = False,
    disable_loggers: list[str] | None = None,
    templates: TemplateStore | None = None,
) -> str:
    """
    Start a global root logger with specified configuration.

    Parameters
    ----------
    basic_config_args : dict[str, object], optional
        Basic configuration arguments for logging.
    show_logger_name : bool
        Whether to include the logger name in the log format.
    show_date : bool
        Whether to include the date in the log format.
    show_loggers_on_init : bool
        Whether to log the existing loggers on initialization.
    disable_loggers : list[str], optional
        List of logger name prefixes to disable.
    templates : TemplateStore, optional
        TemplateStore containing template paths, including 'log_file'.

    Returns
    -------
    str
        The path to the log file used by the root logger.
    """
    basic_config_args = {} if basic_config_args is None else basic_config_args
    basic_config_args = default_basic_config_args(basic_config_args, show_logger_name, show_date)
    disable_loggers = [] if disable_loggers is None else disable_loggers
    delete_filename = False
    if "filename" not in basic_config_args:
        if (templates is None) or ("log_file" not in templates):
            raise ValueError("Either 'filename' must be in basic_config_args or 'log_file' in templates.")
        basic_config_args["filename"] = templates["log_file"]
        delete_filename = True
    path_log_file = basic_config_args["filename"]
    Path(path_log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(**basic_config_args)
    if delete_filename:
        del basic_config_args["filename"]
    logger.debug("Root logger started.")
    if show_loggers_on_init:
        formatted_logger_dict = pformat(sorted(list(logging.root.manager.loggerDict.keys())), compact=True)
        logger.debug("Root logger override. loggerDict keys: \n%s", formatted_logger_dict)
    for disable_logger in disable_loggers:
        for name, known_logger in logging.root.manager.loggerDict.items():
            if (name[0 : len(disable_logger)] == disable_logger) and (hasattr(known_logger, "disabled")):
                known_logger.disabled = True
    return str(path_log_file)
