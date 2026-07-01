import datetime

from resoterre import calendar_utils


def test_iter_year_month():
    start = datetime.datetime(2023, 1, 1)
    end = datetime.datetime(2023, 3, 1)
    expected = [(2023, 1), (2023, 2), (2023, 3)]
    result = list(calendar_utils.iter_year_month(start, end))
    assert result == expected
