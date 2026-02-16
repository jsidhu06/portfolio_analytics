import datetime as dt

from portfolio_analytics.rates import DiscountCurve
from portfolio_analytics.utils import calculate_year_fraction


def flat_curve(
    pricing_date: dt.datetime,
    maturity: dt.datetime,
    rate: float,
    name: str = "r",
) -> DiscountCurve:
    ttm = calculate_year_fraction(pricing_date, maturity)
    return DiscountCurve.flat(name, rate, end_time=ttm)
