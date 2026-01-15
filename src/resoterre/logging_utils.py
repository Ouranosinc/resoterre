"""Utilities for logging."""

import logging
from pathlib import Path
from pprint import pformat
from typing import Any

from resoterre.utils import TemplateStore


logger = logging.getLogger(__name__)


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
