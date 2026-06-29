"""
Identity verification by price overlap.

Mapping a CoinMarketCap id to an exchange symbol is not always the ticker: the
exchange may rename it, reuse it for a different coin, or price-scale it (a
1000SHIB contract is 1000x the coin). This verifies a candidate symbol by
comparing the exchange close against the coin's CoinMarketCap price over their
overlapping days.

The deciding signal is ratio stability, not correlation. All of crypto trends
together, so two different coins correlate highly; only the same coin holds a
near-constant price ratio. A wrong mapping wanders even when it correlates.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class Verdict:
    """The outcome of matching an exchange series to a CoinMarketCap price."""

    accepted: bool
    checked: bool  # whether enough days overlapped to judge at all
    scale: float  # exchange_close / cmc_price; ~1, or ~1000 for a scaled contract
    cv: float  # coefficient of variation of the ratio; small means a stable ratio
    correlation: float
    points: int  # overlapping days compared


def verdict(
    cmc_price: dict[date, float],
    exchange_close: dict[date, float],
    *,
    min_points: int = 8,
    max_cv: float = 0.03,
    min_correlation: float = 0.9,
) -> Verdict:
    """
    Whether `exchange_close` is the same coin as `cmc_price` over their overlap.

    Accepts when the price ratio is stable (`cv` below `max_cv`) and the two
    series move together. With fewer than `min_points` overlapping days there is
    no basis to judge, so the verdict is unaccepted and unchecked, which the
    caller treats as "trust the ticker" rather than "wrong coin".
    """
    common = sorted(set(cmc_price) & set(exchange_close))
    if len(common) < min_points:
        return Verdict(False, False, float("nan"), float("nan"), float("nan"), len(common))

    c = np.array([cmc_price[d] for d in common], dtype=float)
    b = np.array([exchange_close[d] for d in common], dtype=float)
    ratio = b / c
    scale = float(ratio.mean())
    cv = float(ratio.std() / scale) if scale else float("inf")
    correlation = float(np.corrcoef(c, b)[0, 1])
    accepted = cv < max_cv and correlation > min_correlation
    return Verdict(accepted, True, scale, cv, correlation, len(common))
