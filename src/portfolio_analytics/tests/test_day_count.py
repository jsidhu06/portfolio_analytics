"""Tests for day-count conventions."""

import datetime as dt
import numpy as np

from portfolio_analytics.enums import DayCountConvention
from portfolio_analytics.utils import calculate_year_fraction


def test_act_365f_enum():
    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2026, 1, 1)
    assert np.isclose(calculate_year_fraction(start, end, DayCountConvention.ACT_365F), 1.0)


def test_30_360_us():
    start = dt.datetime(2025, 1, 31)
    end = dt.datetime(2025, 2, 28)
    expected = 28.0 / 360.0
    assert np.isclose(
        calculate_year_fraction(start, end, DayCountConvention.THIRTY_360_US), expected
    )
