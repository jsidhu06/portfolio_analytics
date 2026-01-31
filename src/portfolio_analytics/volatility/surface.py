"""Volatility surface container and interpolation."""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.interpolate import griddata


@dataclass(frozen=True)
class VolatilityQuote:
    """Single volatility quote.

    Parameters
    ----------
    strike : float
        Strike price
    expiry : float
        Time to expiry in years
    implied_volatility : float
        Implied volatility (annualized)
    """

    strike: float
    expiry: float
    implied_volatility: float

    def __post_init__(self):
        """Validate parameters."""
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive")
        if self.implied_volatility < 0:
            raise ValueError("implied_volatility must be non-negative")


class VolatilitySurface:
    """Container for volatility quotes with interpolation capabilities.

    This class stores volatility quotes and provides methods to extract
    volatility smiles, term structures, and interpolated values.

    Parameters
    ----------
    quotes : list[VolatilityQuote]
        List of volatility quotes
    interpolation_method : str, optional
        Interpolation method: 'linear', 'cubic', 'nearest' (default: 'linear')
    """

    def __init__(
        self,
        quotes: list[VolatilityQuote],
        interpolation_method: str = "linear",
    ):
        if not quotes:
            raise ValueError("quotes list cannot be empty")

        self.quotes = quotes
        self.interpolation_method = interpolation_method

        # Extract unique strikes and expiries
        self._strikes = sorted(set(q.strike for q in quotes))
        self._expiries = sorted(set(q.expiry for q in quotes))

    def get_vol(self, strike: float, expiry: float) -> float:
        """Get interpolated volatility for given strike and expiry.

        Parameters
        ----------
        strike : float
            Strike price
        expiry : float
            Time to expiry in years

        Returns
        -------
        float
            Interpolated implied volatility

        Raises
        ------
        ValueError
            If strike/expiry is outside the surface range
        """
        # Prepare data for interpolation
        strikes = np.array([q.strike for q in self.quotes])
        expiries = np.array([q.expiry for q in self.quotes])
        vols = np.array([q.implied_volatility for q in self.quotes])

        # Interpolate
        vol = griddata(
            (strikes, expiries),
            vols,
            (strike, expiry),
            method=self.interpolation_method,
            fill_value=np.nan,
        )

        if np.isnan(vol):
            raise ValueError(f"No volatility available for strike={strike}, expiry={expiry}")

        return float(vol)

    def get_smile(self, expiry: float, tolerance: float = 1e-3) -> dict[float, float]:
        """Get volatility smile for a given expiry.

        Parameters
        ----------
        expiry : float
            Time to expiry in years
        tolerance : float, optional
            Tolerance for matching expiry (default: 1e-3)

        Returns
        -------
        dict[float, float]
            Dictionary mapping strike -> implied volatility
        """
        smile_quotes = [
            q for q in self.quotes if abs(q.expiry - expiry) < tolerance
        ]

        if not smile_quotes:
            # Interpolate for each available strike
            smile = {}
            for strike in self._strikes:
                try:
                    vol = self.get_vol(strike, expiry)
                    smile[strike] = vol
                except ValueError:
                    continue
            return smile

        return {q.strike: q.implied_volatility for q in smile_quotes}

    def get_term_structure(self, strike: float, tolerance: float = 1e-3) -> dict[float, float]:
        """Get volatility term structure for a given strike.

        Parameters
        ----------
        strike : float
            Strike price
        tolerance : float, optional
            Tolerance for matching strike (default: 1e-3)

        Returns
        -------
        dict[float, float]
            Dictionary mapping expiry -> implied volatility
        """
        term_quotes = [
            q for q in self.quotes if abs(q.strike - strike) < tolerance
        ]

        if not term_quotes:
            # Interpolate for each available expiry
            term_structure = {}
            for expiry in self._expiries:
                try:
                    vol = self.get_vol(strike, expiry)
                    term_structure[expiry] = vol
                except ValueError:
                    continue
            return term_structure

        return {q.expiry: q.implied_volatility for q in term_quotes}

    def get_available_strikes(self) -> list[float]:
        """Get list of available strikes."""
        return self._strikes.copy()

    def get_available_expiries(self) -> list[float]:
        """Get list of available expiries."""
        return self._expiries.copy()


class VolatilityInterpolator:
    """Base class for volatility interpolation models.

    Subclasses should implement get_vol() method.
    """

    def get_vol(self, strike: float, expiry: float) -> float:
        """Get interpolated volatility.

        Parameters
        ----------
        strike : float
            Strike price
        expiry : float
            Time to expiry in years

        Returns
        -------
        float
            Implied volatility
        """
        raise NotImplementedError("Subclasses must implement get_vol()")
