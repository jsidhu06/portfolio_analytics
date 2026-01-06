"Constant short rate discounting"

import logging
import numpy as np
from .utils import get_year_deltas


class ConstantShortRate:
    """Class for constant short rate discounting.

    Attributes
    ==========
    name: string
        name of the object
    short_rate: float (positive)
        constant rate for discounting

    Methods
    =======
    get_discount_factors:
        get discount factors given a list/array of datetime objects
        or year fractions
    """

    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            logging.warning(
                "Negative short rate supplied to ConstantShortRate class. "
                "Discount factors will exceed 1 for future dates."
            )

    def get_discount_factors(self, date_list, dtobjects=True) -> np.ndarray:
        """Get discount factors for given date list.

        Applies the formula: DF(t) = exp(-r * t)

        Parameters
        ==========
        date_list: list or tuple
            collection of datetime objects or year fractions
        dtobjects: bool, default True
            if True, interpret date_list as datetime objects
            if False, interpret as year fractions

        Returns
        =======
        discount_factors: np.ndarray
            array of shape (n, 2) with [date_or_time, discount_factor]
            for each input date
        """
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        discount_factors = np.exp(-self.short_rate * dlist)
        return np.array((date_list, discount_factors)).T
