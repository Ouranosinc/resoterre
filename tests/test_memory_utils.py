import secrets

from resoterre import memory_utils


def test_readable_memory_usage():
    assert memory_utils.readable_memory_usage(500) == "500 B"
    assert memory_utils.readable_memory_usage(2048) == "2 KB"
    assert memory_utils.readable_memory_usage(5 * 1024**2) == "5 MB"
    assert memory_utils.readable_memory_usage(3 * 1024**3) == "3 GB"
    assert memory_utils.readable_memory_usage(2 * 1024**4) == "2 TB"


def test_get_memory_usage_by_user():
    user_memory = memory_utils.get_memory_usage_by_user()
    user_memory_string = memory_utils.readable_memory_usage(user_memory)
    assert user_memory_string.endswith("B")
    assert user_memory > 1


def test_check_over_memory():
    x = [secrets.randbelow(10**8) / 10**8 for _ in range(10000000)]  # ~ 380 MB
    assert len(x) == 10000000
    assert memory_utils.check_over_memory(memory_threshold_in_gb=0.25)
