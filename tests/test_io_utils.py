import tempfile
import time
from pathlib import Path

import pytest

from resoterre import io_utils


def test_file_purge_safe_mode():
    with pytest.raises(ValueError):
        io_utils.purge_files(Path("tmp/to/somewhere"), pattern="*.txt")
    with pytest.raises(ValueError):
        io_utils.purge_files(Path("/tmp/to/somewhere"), pattern="*")  # noqa: S108


def test_file_purge_older_than():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(10):
            with Path(f"{tmp_dir}/file_{i}.txt").open("w") as f:
                f.write(f"Content of file {i}")
            time.sleep(0.01)
        purge_files = io_utils.purge_files(tmp_dir, pattern="*.txt", older_than=0.03)
        expected_purge_files = [f"file_{i}.txt" for i in range(8)]
        assert len(set(purge_files)) == 8
        for purge_file in purge_files:
            assert purge_file[-10:] in expected_purge_files


def test_file_purge_more_than():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(10):
            with Path(f"{tmp_dir}/file_{i}.txt").open("w") as f:
                f.write(f"Content of file {i}")
            time.sleep(0.001)
        purge_files = io_utils.purge_files(tmp_dir, pattern="*.txt", more_than=5)
        assert len(purge_files) == 5
        assert purge_files[0][-10:] == "file_4.txt"
        assert purge_files[1][-10:] == "file_3.txt"
        assert purge_files[2][-10:] == "file_2.txt"
        assert purge_files[3][-10:] == "file_1.txt"
        assert purge_files[4][-10:] == "file_0.txt"


def test_file_purge_older_than_and_more_than():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(10):
            with Path(f"{tmp_dir}/file_{i}.txt").open("w") as f:
                f.write(f"Content of file {i}")
            time.sleep(0.01)
        purge_files = io_utils.purge_files(
            tmp_dir, pattern="*.txt", older_than=0.07, more_than=5, must_both_be_true=True
        )
        expected_purge_files = [f"file_{i}.txt" for i in range(4)]
        assert len(set(purge_files)) == 4
        for purge_file in purge_files:
            assert purge_file[-10:] in expected_purge_files


def test_file_purge_older_than_or_more_than():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(10):
            with Path(f"{tmp_dir}/file_{i}.txt").open("w") as f:
                f.write(f"Content of file {i}")
            time.sleep(0.01)
        purge_files = io_utils.purge_files(
            tmp_dir, pattern="*.txt", older_than=0.07, more_than=5, must_both_be_true=False
        )
        assert len(purge_files) == 5


def test_override_config_paths_with_dict():
    config = {
        "path_preprocessed_batch": "/original/batch.nc",
        "path_models": "/original/models/",
        "path_output": "/original/output/",
        "other_key": 123,
    }
    overrides = {
        "path_preprocessed_batch": "/new/batch.nc",
        "path_models": None,  # Should not override
        "path_output": "/new/output/",
    }
    result = io_utils.override_config_paths(config, overrides)
    assert result["path_preprocessed_batch"] == "/new/batch.nc"
    assert result["path_models"] == "/original/models/"
    assert result["path_output"] == "/new/output/"
    assert result["other_key"] == 123


def test_override_config_paths_with_yaml(tmp_path):
    config_data = {
        "path_preprocessed_batch": "/original/batch.nc",
        "path_models": "/original/models/",
        "path_output": "/original/output/",
    }
    yaml_file = tmp_path / "config.yaml"
    import yaml

    with yaml_file.open("w") as f:
        yaml.dump(config_data, f)
    overrides = {"path_preprocessed_batch": "/override/batch.nc"}
    result = io_utils.override_config_paths(str(yaml_file), overrides)
    assert result["path_preprocessed_batch"] == "/override/batch.nc"
    assert result["path_models"] == "/original/models/"
    assert result["path_output"] == "/original/output/"
