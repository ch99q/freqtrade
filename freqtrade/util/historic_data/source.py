"""
Exchange OHLCV sources for the historic-data catalog.

A `Source` fetches candles for an exchange symbol over a date range, including
symbols the live exchange API no longer lists. `BinanceVisionSource` reads the
public data.binance.vision archive, which retains delisted and rebranded pairs,
which is exactly the data a survivorship-free backtest needs.
"""

import io
import logging
import zipfile
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS


logger = logging.getLogger(__name__)


class Source(ABC):
    """An exchange OHLCV archive, addressed by symbol and date range."""

    name: str = "source"

    @abstractmethod
    def klines(
        self, base: str, quote: str, timeframe: str, start: date, stop: date
    ) -> pd.DataFrame:
        """
        Candles for the `base`/`quote` market over [start, stop] inclusive, in
        freqtrade column order with UTC `date`. The source forms its own exchange
        symbol from base and quote, since that spelling is exchange-specific.
        Returns an empty frame when nothing is found, so a delisted or unknown
        market is a quiet gap, never an error.
        """


class BinanceVisionSource(Source):
    """
    Reads the public data.binance.vision spot archive.

    Whole months come from the monthly archive; the trailing days the monthly
    archive has not published yet fall back to daily files, so coverage reaches
    roughly yesterday. Parsed months are cached to disk, so a re-run is offline.
    """

    name = "binance-vision"
    _BASE = "https://data.binance.vision/data/spot"

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        session: requests.Session | None = None,
        today: date | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._session = session or _retrying_session(retries)
        self._today = today or datetime.now(UTC).date()

    def klines(
        self, base: str, quote: str, timeframe: str, start: date, stop: date
    ) -> pd.DataFrame:
        symbol = f"{base}{quote}".upper()
        frames: list[pd.DataFrame] = []
        for year, month in _months(start, stop):
            monthly = self._monthly(symbol, timeframe, year, month)
            if monthly is not None:
                frames.append(monthly)
                continue
            # A past month with no monthly file means the market did not trade
            # then; only the recent month the archive has not published yet is
            # worth stitching from daily files, so a missing pair fails fast.
            if self._is_recent(year, month):
                frames.extend(self._daily(symbol, timeframe, year, month, start, stop))

        if not frames:
            return _empty_frame()
        out = pd.concat(frames, ignore_index=True)
        mask = (out["date"].dt.date >= start) & (out["date"].dt.date <= stop)
        out = out.loc[mask]
        return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    def _monthly(self, symbol: str, tf: str, year: int, month: int) -> pd.DataFrame | None:
        stamp = f"{year:04d}-{month:02d}"
        url = f"{self._BASE}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{stamp}.zip"
        return self._cached(f"{symbol}-{tf}-{stamp}", url)

    def _daily(
        self, symbol: str, tf: str, year: int, month: int, start: date, stop: date
    ) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for day in pd.date_range(date(year, month, 1), _month_end(year, month), freq="D"):
            day = day.date()
            if day < start or day > stop:
                continue
            stamp = day.isoformat()
            url = f"{self._BASE}/daily/klines/{symbol}/{tf}/{symbol}-{tf}-{stamp}.zip"
            frame = self._cached(f"{symbol}-{tf}-{stamp}", url)
            if frame is not None:
                frames.append(frame)
        return frames

    def _is_recent(self, year: int, month: int) -> bool:
        # The monthly archive lags by up to a month, so the current and previous
        # calendar months may need the daily files; older months never do.
        months_ago = (self._today.year - year) * 12 + (self._today.month - month)
        return 0 <= months_ago <= 1

    def _cached(self, key: str, url: str) -> pd.DataFrame | None:
        cache_file = self._cache_dir / f"{key}.feather"
        if cache_file.is_file():
            try:
                return pd.read_feather(cache_file)
            except Exception:
                # A truncated cache file is treated as a miss and re-fetched.
                logger.debug("Re-fetching unreadable cache file %s", cache_file)
        frame = self._download(url)
        if frame is not None:
            frame.to_feather(cache_file)
        return frame

    def _download(self, url: str) -> pd.DataFrame | None:
        resp = self._session.get(url, timeout=self._timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
            raw = pd.read_csv(archive.open(archive.namelist()[0]), header=None)
        return _to_ohlcv(raw)


def _to_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    # Newer archive files carry a header row; drop it so column 0 is numeric.
    if not str(raw.iloc[0, 0]).lstrip("-").isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    open_time = pd.to_numeric(raw[0])
    # Binance switched open_time from milliseconds to microseconds in 2025; the
    # magnitude of the value distinguishes them with no version to track.
    unit = "us" if open_time.iloc[0] > 10**14 else "ms"
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(open_time, unit=unit, utc=True),
            "open": raw[1].astype("float64"),
            "high": raw[2].astype("float64"),
            "low": raw[3].astype("float64"),
            "close": raw[4].astype("float64"),
            "volume": raw[5].astype("float64"),
        }
    )
    return out[DEFAULT_DATAFRAME_COLUMNS]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)


def _months(start: date, stop: date):
    year, month = start.year, start.month
    while (year, month) <= (stop.year, stop.month):
        yield year, month
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year


def _month_end(year: int, month: int) -> date:
    return (date(year, month, 1) + pd.offsets.MonthEnd(1)).date()


def _retrying_session(retries: int) -> requests.Session:
    """A session that retries transient failures with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
