import pytest

from resoterre.data_management import timeseries


def test_timeseries():
    ts = timeseries.Timeseries()
    ts.add(0, 0.5)
    assert ts.times == [0]
    assert ts.bounds == [None]
    assert ts.count == [1]
    assert ts.values == [0.5]


def test_timeseries_wrong_order():
    ts = timeseries.Timeseries()
    ts.add(0, 0.5)
    with pytest.raises(ValueError):
        ts.add(0, 1)


def test_timeseries_compress():
    ts = timeseries.Timeseries(maximum_length=100)
    for i in range(100):
        ts.add(i, i * 1000)
    assert len(ts.times) == 100
    ts.add(100, -1)
    assert len(ts.times) == 51
    assert ts.times[0] == 0.5
    assert ts.bounds[0] == (0, 1)
    assert ts.count[0] == 2
    assert ts.values[0] == 500.0
    assert ts.times[-1] == 100
    assert ts.bounds[-1] is None
    assert ts.count[-1] == 1
    assert ts.values[-1] == -1


def test_timeseries_compress_with_keep():
    ts = timeseries.Timeseries(maximum_length=100, nb_of_kept_starting_values=5, nb_of_kept_final_values=10)
    for i in range(100):
        ts.add(i, i * 1000)
    assert len(ts.times) == 100
    ts.add(100, -1)
    assert len(ts.times) == 58
    assert ts.times[4] == 4
    assert ts.bounds[4] is None
    assert ts.count[4] == 1
    assert ts.values[4] == 4000
    assert ts.times[-1] == 100
    assert ts.bounds[-1] is None
    assert ts.count[-1] == 1
    assert ts.values[-1] == -1


def test_timeseries_compress_twice():
    ts = timeseries.Timeseries(maximum_length=100)
    for i in range(201):
        ts.add(i, i * 1000)
    assert len(ts.times) == 51
    assert ts.times[0] == 3.5
    assert ts.bounds[0] == (0, 7)
    assert ts.count[0] == 8
    assert ts.values[0] == 3500.0
    assert ts.times[-1] == 200
    assert ts.bounds[-1] is None
    assert ts.count[-1] == 1
    assert ts.values[-1] == 200000


def test_multi_timeseries():
    mt = timeseries.MultiTimeseries()
    mt.add_concurrent_values(0, {"a": 0.5, "b": 1.5})
    assert mt["a"].times == [0]
    assert mt["b"].values == [1.5]


def test_multi_timeseries_version_change_1_0_to_1_1():
    mt: timeseries.MultiTimeseries = timeseries.MultiTimeseries.__new__(timeseries.MultiTimeseries)
    state = {
        "version": 1.0,
        "maximum_length": 100,
        "compress_method": "statistics",
        "value_method": "mean",
        "nb_of_kept_starting_values": 0,
        "nb_of_kept_final_values": 10,
    }
    mt.__setstate__(state)
    assert mt.version == 1.1
    assert mt.default_maximum_length == 100
    assert mt.default_compress_method == "statistics"
    assert mt.default_value_method == "mean"
    assert mt.default_nb_of_kept_starting_values == 0
    assert mt.default_nb_of_kept_final_values == 10
