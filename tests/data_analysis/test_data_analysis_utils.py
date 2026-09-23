import datetime
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from resoterre.data_analysis.data_analysis_utils import (
    analyze_nan_clusters,
    filter_data,
    analyze_gcm_vs_coarsened_crcm,
    compute_stats,
    matching_rcm_variable,
    max_largest_cluster_pct,
    nan_cluster_sizes,
    stats_to_dataframe,
    write_dataframe_to_csv,
    get_time_period,
    variable_family,
    mean_pool_coarsen,
    map_rcm_to_gcm_grid,
    stats_gcm_vs_crcm,
    summarize_data,
    select_valid_times,
)
from resoterre.data_analysis.data_analysis_plots import (
    visualize_temporal_mean_and_sample,
    visualize_gcm_vs_coarsened_rcm,
    scenario_from_sim,
    sim_labels,
)


class _FakeEmulatorDataset:
    """Minimal stand-in for CRCMEmulatorDataset used by select_valid_data_in_range."""

    def __init__(
        self,
        valid_idx: list[tuple[str, int, int]],
        stores: dict[str, dict[str, xr.Dataset]],
        gcm_variables: list[str],
        crcm_variables: list[str],
    ) -> None:
        self.valid_idx = valid_idx
        self._stores = stores
        self.gcm_variables = gcm_variables
        self.crcm_variables = crcm_variables

    def get_open_dataset(self, dataset_type: str, key: str) -> xr.Dataset:
        return self._stores[dataset_type][key]


def _daily_dataset(var_name: str, start: str, periods: int) -> xr.Dataset:
    times = pd.date_range(start, periods=periods, freq="D")
    return xr.Dataset({var_name: ("time", np.arange(periods, dtype=float))}, coords={"time": times})


def test_filter_data_keeps_in_window_days():
    gcm = _daily_dataset("ta850", "2000-01-01", 5)
    crcm = _daily_dataset("tas", "2000-01-01", 5)
    dataset = _FakeEmulatorDataset(
        valid_idx=[("sim_a", i, i) for i in range(5)],
        stores={"gcm": {"sim_a": gcm}, "crcm": {"sim_a": crcm}},
        gcm_variables=["ta850"],
        crcm_variables=["tas"],
    )

    data_gcm, data_crcm = filter_data(
        dataset=dataset,
        logger=logging.getLogger("test"),
        start_date=datetime.datetime(2000, 1, 2),
        end_date=datetime.datetime(2000, 1, 3),
    )

    assert list(data_gcm) == ["sim_a"]
    assert data_gcm["sim_a"].sizes["time"] == 2
    np.testing.assert_array_equal(data_gcm["sim_a"]["ta850"].values, [1.0, 2.0])
    np.testing.assert_array_equal(data_crcm["sim_a"]["tas"].values, [1.0, 2.0])


def test_filter_data_skips_simulations_outside_window():
    gcm = _daily_dataset("ta850", "2000-01-01", 3)
    crcm = _daily_dataset("tas", "2000-01-01", 3)
    dataset = _FakeEmulatorDataset(
        valid_idx=[("in_range", 1, 1), ("out_of_range", 0, 0)],
        stores={
            "gcm": {"in_range": gcm, "out_of_range": gcm},
            "crcm": {"in_range": crcm, "out_of_range": crcm},
        },
        gcm_variables=["ta850"],
        crcm_variables=["tas"],
    )

    data_gcm, data_crcm = filter_data(
        dataset=dataset,
        logger=logging.getLogger("test"),
        start_date=datetime.datetime(2000, 1, 2),
        end_date=datetime.datetime(2000, 1, 3),
    )

    assert list(data_gcm) == ["in_range"]
    assert list(data_crcm) == ["in_range"]
    assert data_gcm["in_range"].sizes["time"] == 1


def test_filter_data_without_dates_keeps_all_valid_indices_pairs():
    gcm = _daily_dataset("tas", "2000-01-01", 3)
    crcm = _daily_dataset("pr", "2000-01-01", 3)
    dataset = _FakeEmulatorDataset(
        valid_idx=[("sim_a", i, i) for i in range(3)],
        stores={"gcm": {"sim_a": gcm}, "crcm": {"sim_a": crcm}},
        gcm_variables=["tas"],
        crcm_variables=["pr"],
    )

    data_gcm, data_crcm = filter_data(dataset, logging.getLogger("test"))

    assert data_gcm["sim_a"].sizes["time"] == 3
    assert data_crcm["sim_a"].sizes["time"] == 3


def test_select_valid_times_uses_slice_for_contiguous_indices():
    ds = _daily_dataset("ta850", "2000-01-01", 5)

    selected = select_valid_times(ds, [1, 2, 3])

    assert selected.sizes["time"] == 3
    np.testing.assert_array_equal(selected["ta850"].values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        selected["time"].values,
        pd.date_range("2000-01-02", periods=3, freq="D"),
    )

    single = select_valid_times(ds["ta850"], [4])
    np.testing.assert_array_equal(single.values, [4.0])


def test_select_valid_times_selects_scattered_indices():
    ds = _daily_dataset("ta850", "2000-01-01", 5)

    selected = select_valid_times(ds, [0, 2, 4])

    assert selected.sizes["time"] == 3
    np.testing.assert_array_equal(selected["ta850"].values, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        selected["time"].values,
        pd.to_datetime(["2000-01-01", "2000-01-03", "2000-01-05"]),
    )


def test_summarize_data_writes_gcm_and_crcm_rows(tmp_path):
    gcm = _daily_dataset("ta850", "2000-01-01", 5)
    crcm = _daily_dataset("tas", "2000-01-01", 5)
    dataset = _FakeEmulatorDataset(
        valid_idx=[("sim_a", i, i) for i in range(5)],
        stores={"gcm": {"sim_a": gcm}, "crcm": {"sim_a": crcm}},
        gcm_variables=["ta850"],
        crcm_variables=["tas"],
    )

    stats_df = summarize_data(
        dataset=dataset,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
        start_date=datetime.datetime(2000, 1, 2),
        end_date=datetime.datetime(2000, 1, 3),
    )

    assert list(stats_df.columns) == [
        "sim",
        "model",
        "variable",
        "n_samples",
        "pct_non_nan_values",
        "mean",
        "std",
        "min",
        "max",
        "range",
    ]
    assert list(zip(stats_df["model"], stats_df["variable"])) == [("gcm", "ta850"), ("crcm", "tas")]
    assert set(stats_df["sim"]) == {"sim_a"}
    assert set(stats_df["n_samples"]) == {2}
    # Filtered days are 2000-01-02 and 2000-01-03, whose values are 1.0 and 2.0.
    assert stats_df[stats_df["model"] == "crcm"]["mean"].values[0] == pytest.approx(1.5)
    assert stats_df[stats_df["model"] == "crcm"]["std"].values[0] == pytest.approx(0.5)
    assert stats_df[stats_df["model"] == "crcm"]["min"].values[0] == pytest.approx(1.0)
    assert stats_df[stats_df["model"] == "crcm"]["max"].values[0] == pytest.approx(2.0)
    assert stats_df[stats_df["model"] == "crcm"]["range"].values[0] == pytest.approx(1.0)
    assert stats_df[stats_df["model"] == "gcm"]["mean"].values[0] == pytest.approx(1.5)

    written = pd.read_csv(tmp_path / "simulation_stats.csv")
    assert len(written) == 2
    assert list(written["variable"]) == ["ta850", "tas"]


def test_summarize_data_raises_when_no_days_in_range(tmp_path):
    gcm = _daily_dataset("ta850", "2000-01-01", 3)
    crcm = _daily_dataset("tas", "2000-01-01", 3)
    dataset = _FakeEmulatorDataset(
        valid_idx=[("sim_a", i, i) for i in range(3)],
        stores={"gcm": {"sim_a": gcm}, "crcm": {"sim_a": crcm}},
        gcm_variables=["ta850"],
        crcm_variables=["tas"],
    )

    with pytest.raises(ValueError, match="No simulations have valid days"):
        summarize_data(
            dataset=dataset,
            output_dir=tmp_path,
            logger=logging.getLogger("test"),
            start_date=datetime.datetime(1999, 1, 1),
            end_date=datetime.datetime(1999, 1, 31),
        )

    assert not (tmp_path / "simulation_stats.csv").exists()


def test_get_time_period_accepts_python_datetime_against_datetime64():
    times = xr.DataArray(pd.date_range("2000-01-01", periods=5, freq="D"), dims="time")
    times.encoding.update({"units": "days since 2000-01-01", "calendar": "standard", "dtype": np.dtype("int64")})
    mask = get_time_period(times, datetime.datetime(2000, 1, 2), datetime.datetime(2000, 1, 3))
    np.testing.assert_array_equal(mask, [False, True, True, False, False])


def _spatial_dataset(
    var_name: str,
    start: str,
    periods: int,
    offset: float = 0.0,
    ny: int = 4,
    nx: int = 4,
) -> xr.Dataset:
    times = pd.date_range(start, periods=periods, freq="D")
    data = np.arange(periods * ny * nx, dtype=float).reshape(periods, ny, nx) + offset
    return xr.Dataset(
        {var_name: (("time", "y", "x"), data)},
        coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)},
    )


def test_visualize_temporal_mean_and_sample_writes_pngs(tmp_path):
    visualize_temporal_mean_and_sample(
        data_gcm={
            "historical": _spatial_dataset("tas", "2000-01-01", 2, offset=0.0),
            "ssp245": _spatial_dataset("tas", "2000-01-01", 2, offset=10.0),
        },
        data_crcm={
            "historical": _spatial_dataset("tas", "2000-01-01", 2, offset=1.0),
            "ssp245": _spatial_dataset("tas", "2000-01-01", 2, offset=11.0),
        },
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 4
    assert {p.name for p in pngs} == {
        "gcm_tas_historical.png",
        "gcm_tas_ssp245.png",
        "crcm_tas_historical.png",
        "crcm_tas_ssp245.png",
    }


def test_scenario_from_sim_uses_underscore_split():
    assert scenario_from_sim("CNRM-ESM2-1_historical_r1i1p1f2") == "historical"
    assert scenario_from_sim("CNRM-ESM2-1_ssp245_r1i1p1f2") == "ssp245"
    assert scenario_from_sim("MPI-ESM1-2-HR_ssp585_r1i1p1f1") == "ssp585"
    assert scenario_from_sim("historical") == "historical"


def test_sim_labels_disambiguates_shared_scenario():
    # One simulation per scenario keeps the short scenario label.
    assert sim_labels(["CNRM-ESM2-1_historical_r1i1p1f2", "CNRM-ESM2-1_ssp245_r1i1p1f2"]) == {
        "CNRM-ESM2-1_historical_r1i1p1f2": "historical",
        "CNRM-ESM2-1_ssp245_r1i1p1f2": "ssp245",
    }

    # Two ensemble members of the same scenario fall back to the full name.
    assert sim_labels(["CNRM-ESM2-1_historical_r1i1p1f2", "CNRM-ESM2-1_historical_r2i1p1f2"]) == {
        "CNRM-ESM2-1_historical_r1i1p1f2": "CNRM-ESM2-1_historical_r1i1p1f2",
        "CNRM-ESM2-1_historical_r2i1p1f2": "CNRM-ESM2-1_historical_r2i1p1f2",
    }

    # A repeated simulation name must not be mistaken for two simulations.
    assert sim_labels(["CNRM-ESM2-1_ssp245_r1i1p1f2", "CNRM-ESM2-1_ssp245_r1i1p1f2"]) == {
        "CNRM-ESM2-1_ssp245_r1i1p1f2": "ssp245",
    }


def test_variable_family_maps_surface_onto_pressure_level_stem():
    assert variable_family("ta850") == variable_family("tas") == "ta"
    assert variable_family("ua500") == variable_family("uas") == "ua"
    assert variable_family("va500") == variable_family("vas") == "va"
    # hus is already a stem, so hus850 must pair with huss rather than with a "hu" family.
    assert variable_family("hus850") == variable_family("huss") == "hus"
    # Variables with no surface counterpart are returned unchanged.
    assert variable_family("pr") == "pr"
    assert variable_family("psl") == "psl"
    assert variable_family("zg500") == "zg"


def test_matching_rcm_variable_pairs_by_family():
    rcm_variables = ["tas", "pr", "uas", "vas", "huss"]
    assert matching_rcm_variable("ta1000", rcm_variables) == "tas"
    assert matching_rcm_variable("ta850", rcm_variables) == "tas"
    assert matching_rcm_variable("tas", rcm_variables) == "tas"
    assert matching_rcm_variable("ua850", rcm_variables) == "uas"
    assert matching_rcm_variable("uas", rcm_variables) == "uas"
    assert matching_rcm_variable("va500", rcm_variables) == "vas"
    assert matching_rcm_variable("zg850", rcm_variables) is None
    assert matching_rcm_variable("psl", rcm_variables) is None
    assert matching_rcm_variable("pr", rcm_variables) == "pr"
    assert matching_rcm_variable("huss", rcm_variables) == "huss"
    assert matching_rcm_variable("hus850", rcm_variables) == "huss"


def test_mean_pool_coarsen_pools_by_factor():
    data = xr.DataArray(np.arange(start=2, stop=10, step=2).reshape(2, 2), dims=("y", "x"))
    expected = xr.DataArray(np.array([[5.0]]), dims=("y", "x"))
    xr.testing.assert_equal(mean_pool_coarsen(data, 2), expected)


def test_map_rcm_to_gcm_grid_copies_gcm_coordinates():
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    values = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)
    rcm_coarse = xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="tas",
    )
    gcm = xr.DataArray(
        values + 10.0,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [10.0, 20.0], "x": [100.0, 200.0]},
        name="ta850",
    )

    mapped = map_rcm_to_gcm_grid(rcm_coarse, gcm)

    np.testing.assert_array_equal(mapped.values, values)
    np.testing.assert_array_equal(mapped["y"].values, [10.0, 20.0])
    np.testing.assert_array_equal(mapped["x"].values, [100.0, 200.0])
    assert mapped.name == "tas"
    assert mapped.dims == ("time", "y", "x")


def test_map_rcm_to_gcm_grid_raises_on_shape_mismatch():
    times = pd.date_range("2000-01-01", periods=2, freq="D")
    rcm_coarse = xr.DataArray(
        np.zeros((2, 4, 4)),
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
        name="tas",
    )
    gcm = xr.DataArray(
        np.zeros((2, 2, 2)),
        dims=("time", "y", "x"),
        coords={"time": times, "y": np.arange(2), "x": np.arange(2)},
        name="ta850",
    )

    with pytest.raises(ValueError, match="shape mismatch after coarsen"):
        map_rcm_to_gcm_grid(rcm_coarse, gcm)


def test_analyze_gcm_vs_coarsened_crcm(tmp_path):
    stats_per_var = analyze_gcm_vs_coarsened_crcm(
        data_gcm={"historical": _spatial_dataset("ta850", "2000-01-01", 2, ny=2, nx=2)},
        data_crcm={
            "historical": xr.Dataset(
                {
                    "tas": (
                        ("time", "y", "x"),
                        np.array([
                            [
                                [0, 0, 1, 1],
                                [0, 0, 1, 1],
                                [2, 2, 3, 3],
                                [2, 2, 3, 3],
                            ],
                            [
                                [4, 4, 5, 5],
                                [4, 4, 5, 5],
                                [6, 6, 7, 7],
                                [6, 6, 7, 7],
                            ],
                        ])
                    )
                },
                coords={
                    "time": pd.date_range("2000-01-01", periods=2, freq="D"),
                    "y": np.arange(4),
                    "x": np.arange(4),
                },
            )
        },
   
        gcm_variables=["ta850"],
        rcm_variables=["tas"],
        coarsen_factor=2,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )
    

    csv_path = tmp_path / "gcm_vs_coarsened_rcm.csv"
    assert csv_path.is_file()
    stats_df = pd.read_csv(csv_path)
    assert list(stats_df.columns) == [
        "sim",
        "gcm",
        "rcm",
        "n_valid",
        "pct_valid",
        "bias",
        "mae",
        "rmse",
        "pearson_r_clim",
    ]
    assert len(stats_df) == 1
    assert stats_df.loc[0, "sim"] == "historical"
    assert stats_df.loc[0, "gcm"] == "ta850"
    assert stats_df.loc[0, "rcm"] == "tas"
    assert list(stats_per_var) == [("historical", "ta850")]

    assert stats_per_var[("historical", "ta850")]["n_valid"] == 8
    assert stats_per_var[("historical", "ta850")]["pct_valid_pixels"] == 100.0
    assert stats_per_var[("historical", "ta850")]["bias"] == pytest.approx(0.0)
    assert stats_per_var[("historical", "ta850")]["mae"] == pytest.approx(0.0)
    assert stats_per_var[("historical", "ta850")]["rmse"] == pytest.approx(0.0)
    assert stats_per_var[("historical", "ta850")]["pearson_r_clim"] == pytest.approx(1.0)

    visualize_gcm_vs_coarsened_rcm(
        stats_per_var=stats_per_var,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )
    pngs = list(tmp_path.glob("gcm_vs_rcm_*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "gcm_vs_rcm_ta850_historical.png"


def _multi_var_spatial(var_names: list[str], ny: int, nx: int, offset: float = 0.0, periods: int = 2) -> xr.Dataset:
    times = pd.date_range("2000-01-01", periods=periods, freq="D")
    data_vars = {
        name: (("time", "y", "x"), np.arange(periods * ny * nx, dtype=float).reshape(periods, ny, nx) + offset + i)
        for i, name in enumerate(var_names)
    }
    return xr.Dataset(data_vars, coords={"time": times, "y": np.arange(ny), "x": np.arange(nx)})


def test_compare_gcm_vs_coarsened_rcm_skips_unpaired_variables(tmp_path):
    stats_per_var = analyze_gcm_vs_coarsened_crcm(
        data_gcm={"historical": _multi_var_spatial(["ta850", "ua850", "zg850"], ny=2, nx=2)},
        data_crcm={"historical": _multi_var_spatial(["tas", "uas", "pr"], ny=4, nx=4, offset=1.0)},
        gcm_variables=["ta850", "ua850", "zg850"],
        rcm_variables=["tas", "uas", "pr"],
        coarsen_factor=2,
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )

    assert set(stats_per_var) == {("historical", "ta850"), ("historical", "ua850")}
    stats_df = pd.read_csv(tmp_path / "gcm_vs_coarsened_rcm.csv")
    assert set(zip(stats_df["gcm"], stats_df["rcm"])) == {("ta850", "tas"), ("ua850", "uas")}


def test_filter_data_keeps_gcm_and_crcm_indices_paired():
    gcm = _daily_dataset("tas", "2000-01-01", 5)
    crcm = _daily_dataset("pr", "2000-01-01", 5)
    # Out of order, with one pair repeated: GCM step 2 pairs with CRCM step 0,
    # and GCM step 0 pairs with CRCM step 1.
    dataset = _FakeEmulatorDataset(
        valid_idx=[("sim_a", 2, 0), ("sim_a", 0, 1), ("sim_a", 2, 0)],
        stores={"gcm": {"sim_a": gcm}, "crcm": {"sim_a": crcm}},
        gcm_variables=["tas"],
        crcm_variables=["pr"],
    )

    data_gcm, data_crcm = filter_data(dataset, logging.getLogger("test"))

    # Deduplicating each list independently would yield CRCM [0.0, 1.0] and break the pairing.
    np.testing.assert_array_equal(data_gcm["sim_a"]["tas"].values, [0.0, 2.0])
    np.testing.assert_array_equal(data_crcm["sim_a"]["pr"].values, [1.0, 0.0])

    # test only one start_date value raises error
    with pytest.raises(ValueError):
        filter_data(dataset, logging.getLogger("test"), start_date=datetime.datetime(2000, 1, 1))


def test_compare_gcm_rcm_recovers_a_known_offset():
    gcm = _spatial_dataset("tas", "2000-01-01", 4)["tas"]
    rcm = gcm + 2.5

    stats = stats_gcm_vs_crcm(gcm, rcm)

    assert stats["n_valid"] == gcm.size
    assert stats["pct_valid_pixels"] == 100.0
    assert stats["bias"] == pytest.approx(2.5)
    assert stats["mae"] == pytest.approx(2.5)
    assert stats["rmse"] == pytest.approx(2.5)
    assert stats["pearson_r_clim"] == pytest.approx(1.0)


def test_compare_gcm_rcm_ignores_pixels_missing_in_either_model():
    gcm = _spatial_dataset("tas", "2000-01-01", 2)["tas"]
    rcm = gcm + 1.0
    rcm = rcm.where(rcm["x"] > 0)  # drop one column of the RCM field

    stats = stats_gcm_vs_crcm(gcm, rcm)

    assert stats["n_valid"] < gcm.size
    assert stats["bias"] == pytest.approx(1.0)


def test_nan_cluster_sizes_orders_clusters_largest_first():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0:3] = True  # cluster of 3
    mask[3, 3] = True  # cluster of 1

    np.testing.assert_array_equal(nan_cluster_sizes(mask), [3.0, 1.0])
    assert nan_cluster_sizes(np.zeros((4, 4), dtype=bool)).size == 0


def test_nan_cluster_sizes_honours_connectivity():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True  # diagonally adjacent

    np.testing.assert_array_equal(nan_cluster_sizes(mask, connectivity=8), [2.0])
    np.testing.assert_array_equal(nan_cluster_sizes(mask, connectivity=4), [1.0, 1.0])


def test_max_largest_cluster_pct_matches_across_block_sizes():
    mask = np.zeros((3, 4, 4), dtype=bool)
    mask[0, 0, 0] = True  # 1 of 16 pixels -> 6.25%
    mask[2, 0:2, 0:2] = True  # 4 of 16 pixels -> 25%
    mask_da = xr.DataArray(mask, dims=("time", "y", "x"))

    assert max_largest_cluster_pct(mask) == pytest.approx(25.0)
    # Streaming a DataArray in blocks must give the same answer as one array.
    assert max_largest_cluster_pct(mask_da, block_size=1) == pytest.approx(25.0)
    assert max_largest_cluster_pct(mask_da, block_size=2) == pytest.approx(25.0)
    assert max_largest_cluster_pct(mask_da, block_size=100) == pytest.approx(25.0)


def test_write_dataframe_to_csv_preserves_small_magnitudes(tmp_path):
    df = pd.DataFrame({"variable": ["huss", "pr", "tas"], "mean": [0.00832, 1.2e-5, 288.153]})

    write_dataframe_to_csv(df, tmp_path, "small_values", logging.getLogger("test"))

    written = pd.read_csv(tmp_path / "small_values.csv")
    assert written.loc[0, "mean"] == pytest.approx(0.00832)
    assert written.loc[1, "mean"] == pytest.approx(1.2e-5)
    assert written.loc[2, "mean"] == pytest.approx(288.2, abs=0.05)


def test_compute_stats_summary_matches_numpy():
    ds = _spatial_dataset("tas", "2000-01-01", 4)
    values = ds["tas"].values

    stats = compute_stats(ds)
    stats_df = stats_to_dataframe("sim_a", "gcm", stats, logging.getLogger("test"))

    assert list(stats_df["variable"]) == ["tas"]
    assert int(stats_df.loc[0, "n_samples"]) == 4
    assert int(stats_df.loc[0, "pct_non_nan_values"]) == 100
    assert stats_df.loc[0, "mean"] == pytest.approx(float(values.mean()))
    assert stats_df.loc[0, "std"] == pytest.approx(float(values.std()))  # ddof=0
    assert stats_df.loc[0, "min"] == pytest.approx(float(values.min()))
    assert stats_df.loc[0, "max"] == pytest.approx(float(values.max()))


def test_analyze_nan_clusters(tmp_path):
    ds = _spatial_dataset("tas", "2000-01-01", 3, ny=4, nx=4)
    # Two persistently NaN clusters: a 2x2 block (4 of 16 pixels -> 25%) and an
    # isolated pixel (1 of 16 -> 6.25%), far enough apart not to be connected.
    block = (ds["y"] > 1) | (ds["x"] > 1)
    isolated = ~((ds["y"] == 3) & (ds["x"] == 3))
    ds = ds.assign(tas=ds["tas"].where(block & isolated))

    nan_df = analyze_nan_clusters(
        model_data={"historical": ds},
        variables=["tas"],
        output_dir=tmp_path,
        logger=logging.getLogger("test"),
    )

    assert nan_df.loc[0, "n_clusters"] == 2
    assert nan_df.loc[0, "pct_always_nan"] == pytest.approx(31.25)
    assert nan_df.loc[0, "max_cluster_pct"] == pytest.approx(25.0)
    assert nan_df.loc[0, "min_cluster_pct"] == pytest.approx(6.25)
    assert nan_df.loc[0, "max_cluster_pct_over_time"] == pytest.approx(25.0)

    written = pd.read_csv(tmp_path / "nan_clusters.csv")
    assert written.loc[0, "cluster_pcts"] == "25.00;6.25"
