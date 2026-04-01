from resoterre.data_management import data_io


def test_sample_chunk_size_01():
    # Suppose you have a (100, 1000, 1000) array to chunk in the first dimension
    chunk_size = data_io.sample_chunk_size(
        extra_dimensions_product=1000 * 1000,
        bytes_per_value=4,  # float32, each block has 1000 * 1000 * 4 bytes = 4,000,000 bytes
        target_chunk_mib=8,  # 8 MiB = 8 * 1024 * 1024 bytes = 8,388,608 bytes
        min_chunk=1,
        max_chunk=1024,
    )
    assert chunk_size == 2  # optimal strategy if to chunk the first dimension in blocks of 2


def test_sample_chunk_size_02():
    chunk_size = data_io.sample_chunk_size(
        extra_dimensions_product=10000 * 10000, bytes_per_value=4, target_chunk_mib=8, min_chunk=1, max_chunk=1024
    )
    assert chunk_size == 1


def test_sample_chunk_size_03():
    chunk_size = data_io.sample_chunk_size(
        extra_dimensions_product=100 * 100, bytes_per_value=4, target_chunk_mib=8, min_chunk=1, max_chunk=4
    )
    assert chunk_size == 4
