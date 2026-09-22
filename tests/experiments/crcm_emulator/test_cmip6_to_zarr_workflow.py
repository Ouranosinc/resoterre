from resoterre.experiments.crcm_emulator import cmip6_to_zarr_workflow


def test_get_chunk_indices():
    gcm = "CNRM-ESM2-1"
    start_datetime = "2000-01-01"
    end_datetime = "2000-03-31"
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 90)]
    end_datetime = cmip6_to_zarr_workflow.chunk_index_to_datetime(
        gcm, start_datetime, end_datetime, chunk_idx=chunk_indices[-1][1]
    )
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31


def test_get_chunk_indices_no_leap():
    gcm = "CanESM5"
    start_datetime = "2000-01-01"
    end_datetime = "2000-03-31"
    chunk_indices = cmip6_to_zarr_workflow.get_chunk_indices(
        gcm, start_datetime, end_datetime, chunk_size=8, chunks_per_task=2
    )
    assert chunk_indices == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 89)]
    end_datetime = cmip6_to_zarr_workflow.chunk_index_to_datetime(
        gcm, start_datetime, end_datetime, chunk_idx=chunk_indices[-1][1]
    )
    assert end_datetime.year == 2000
    assert end_datetime.month == 3
    assert end_datetime.day == 31
