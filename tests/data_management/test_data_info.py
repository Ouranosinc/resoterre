import numpy as np
import pytest

from resoterre.data_management import data_info


def test_data_info():
    di = data_info.DataInfo(categories={"All"})
    assert di.categories == {"All"}
    di.set_bool("exists", True)
    di.set_min("min", 0, num_values=6)
    di.set_max("max", 1, num_values=6)
    di.set_mean("mean", 0.5, num_values=6)
    di.set_shape("shape", (2, 3))
    di.init_none({"processed": "bool"})
    assert di.bool_info == {"exists": True, "processed": None}


def test_dataset_info_create_entry():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    idx = di.create_entry(file_path="/path/to/file", variable_name="dummy", level=700, dummy_metadata="test")
    assert idx == 1
    assert di.num_entries == 2
    assert di._metadata["dummy_metadata"] == [None, "test"]


def test_dataset_info_set_properties_empty_error():
    di = data_info.DatasetInfo()
    with pytest.raises(IndexError):
        di.set_properties(is_valid=False)


def test_dataset_info_set_properties():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.set_properties(shape=(2, 2))
    assert di._properties["shape"] == [(2, 2)]
    assert "shape" not in di._bool_properties


def test_dataset_info_set_properties_specific_idx():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=700)
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=500)
    di.set_properties(idx=1, shape=(2, 2))
    assert di._properties["shape"] == [None, (2, 2), None]


def test_dataset_info_set_properties_multiple_idx():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=700)
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=500)
    di.set_properties(idx=[0, 1], shape=(2, 2))
    assert di._properties["shape"] == [(2, 2), (2, 2), None]


def test_dataset_info_set_properties_boolean():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.set_properties(is_valid=False, is_bool=True)
    assert di._properties["is_valid"] == [False]
    assert "is_valid" in di._bool_properties


def test_dataset_info_set_statistics_kwargs():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=700)
    di.set_statistics(custom_statistic=0.3252)
    assert di._statistics["custom_statistic"] == [None, 0.3252]


def test_dataset_info_set_statistics_data_array():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.set_statistics(data_array=np.array([0.0, 1.0, 0.5, 0.75, 0.25]))
    assert np.array_equal(di._statistics["count"], [5])
    assert np.array_equal(di._statistics["min"], [0.0])
    assert np.array_equal(di._statistics["max"], [1.0])
    assert np.array_equal(di._statistics["mean"], [0.5])
    assert np.array_equal(di._statistics["nan_fraction"], [0.0])


def test_dataset_info_set_statistics_data_array_with_nan():
    di = data_info.DatasetInfo()
    di.create_entry(file_path="/path/to/file", variable_name="dummy", level=850)
    di.set_statistics(data_array=np.array([0.0, 1.0, 0.5, np.nan, 0.75, 0.25]))
    assert np.array_equal(di._statistics["count"], [5])
    assert np.array_equal(di._statistics["min"], [0.0])
    assert np.array_equal(di._statistics["max"], [1.0])
    assert np.array_equal(di._statistics["mean"], [0.5])
    assert np.array_equal(di._statistics["nan_fraction"], [1 / 6])
