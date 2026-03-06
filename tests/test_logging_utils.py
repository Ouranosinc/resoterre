import logging
import tempfile
from pathlib import Path

from resoterre import logging_utils


logger = logging_utils.CustomLogging(caller=logging.getLogger(__name__))


def test_readable_delta_t():
    assert logging_utils.readable_delta_t(0) == "0 µs"
    assert logging_utils.readable_delta_t(0.1) == "100 ms"
    assert logging_utils.readable_delta_t(1) == "1 s"
    assert logging_utils.readable_delta_t(59) == "59 s"
    assert logging_utils.readable_delta_t(60) == "1 min 0 s"
    assert logging_utils.readable_delta_t(61) == "1 min 1 s"
    assert logging_utils.readable_delta_t(3600) == "1 h 0 min"
    assert logging_utils.readable_delta_t(3661) == "1 h 1 min"
    assert logging_utils.readable_delta_t(90061) == "1 d 1 h"


def test_readable_value():
    assert logging_utils.readable_value(20.0) == "20"
    assert logging_utils.readable_value(0.001) == "1.0000e-03"
    assert logging_utils.readable_value(0.1) == "0.1000"
    assert logging_utils.readable_value(100.3) == "100.30"
    assert logging_utils.readable_value(1500.456) == "1500.46"
    assert logging_utils.readable_value(100200.2) == "100200"
    assert logging_utils.readable_value(1_500_000) == "1.5000e+06"
    assert logging_utils.readable_value(20.0, expected_max=1.0) == "2.0e+01"


def test_default_basic_config_args_01():
    basic_config_args = logging_utils.default_basic_config_args(basic_config_args={})
    assert basic_config_args["format"] == "%(asctime)s[%(levelname)s] (%(funcName)s) %(message)s"
    assert basic_config_args["datefmt"] == "%Y-%m-%dT%H:%M:%S"
    assert basic_config_args["level"] == 10
    assert basic_config_args["force"] is True


def test_default_basic_config_args_02():
    basic_config_args = logging_utils.default_basic_config_args(
        basic_config_args={}, show_logger_name=True, show_date=False
    )
    assert basic_config_args["format"] == "%(asctime)s[%(levelname)s] (%(funcName)s, %(name)s) %(message)s"
    assert basic_config_args["datefmt"] == "%H:%M:%S"
    assert basic_config_args["level"] == 10
    assert basic_config_args["force"] is True


def test_start_root_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file_path = Path(tmp_dir, "test.log")
        logging_utils.start_root_logger(basic_config_args={"filename": str(log_file_path)})
        logger.debug("Test message.")
        logger.debug("Valid message", block_short_repetition_delay=10, identifier="test")
        logger.debug("Silenced message", block_short_repetition_delay=10, identifier="test")
        assert log_file_path.is_file()
