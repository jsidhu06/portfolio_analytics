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

    def get_discount_factors(self, date_list, dtobjects=True):
        """Get discount factors for given date list."""
        if dtobjects is True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        discount_factors = np.exp(-self.short_rate * dlist)
        return np.array((date_list, discount_factors)).T
