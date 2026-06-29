"""
Identity-keyed historic OHLCV download, for survivorship-free backtests.

Resolves each CoinMarketCap id into the time-bounded exchange symbols it traded
under, downloads them from a `Source`, and stores the candles through
freqtrade's data handler. See `catalog.create_catalog`.
"""

from freqtrade.util.historic_data.catalog import (
    Catalog,
    Coin,
    Identity,
    PairResult,
    Provenance,
    Report,
    Span,
    Universe,
    create_catalog,
)
from freqtrade.util.historic_data.source import BinanceVisionSource, Source
from freqtrade.util.historic_data.verify import Verdict, verdict


__all__ = [
    "BinanceVisionSource",
    "Catalog",
    "Coin",
    "Identity",
    "PairResult",
    "Provenance",
    "Report",
    "Source",
    "Span",
    "Universe",
    "Verdict",
    "create_catalog",
    "verdict",
]
