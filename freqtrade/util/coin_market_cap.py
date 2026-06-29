"""
Historical market-cap snapshots from CoinMarketCap.

Reads the daily ranked listing that powers the pages under
coinmarketcap.com/historical/. The endpoint is public (no key), serves any
calendar date back to 2013, and returns the full ranked universe for that day
with circulating supply, price, volume, and market cap.

Snapshots are cached to disk per day, so a backtest reads historical rankings
without making a network call inside its candle loop.
"""

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listings/historical"
_USD_ID = 2781

# A browser User-Agent and a coinmarketcap Referer are the only headers the
# endpoint checks; a bare request answers 404 while these answer 200.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://coinmarketcap.com/historical/",
}

_COLUMNS = [
    "cmc_id",
    "rank",
    "symbol",
    "name",
    "market_cap",
    "price",
    "circulating_supply",
    "volume24h",
]


class FtCoinMarketCapApi:
    """
    Cached reader for CoinMarketCap historical daily listings.

    :param cache_dir: directory where per-day snapshots are stored as feather.
    :param limit: ranks fetched per day; 1000 covers any realistic top-N universe.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        limit: int = 1000,
        timeout: float = 30.0,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._limit = limit
        self._timeout = timeout
        self._session = session or _retrying_session(retries)
        self._memory: dict[date, pd.DataFrame] = {}

    def snapshot(self, day: datetime | date) -> pd.DataFrame:
        """The ranked listing for one day, from memory, disk, or the network."""
        day = day.date() if isinstance(day, datetime) else day
        if day in self._memory:
            return self._memory[day]

        cache_file = self._cache_dir / f"{day.isoformat()}.feather"
        frame = None
        if cache_file.is_file():
            try:
                frame = pd.read_feather(cache_file)
            except Exception:
                # A truncated or corrupt cache file is treated as a miss.
                frame = None
        if frame is None:
            frame = self._download(day)
            if not frame.empty:
                frame.to_feather(cache_file)

        self._memory[day] = frame
        return frame

    def ranked_symbols(self, day: datetime | date, max_rank: int) -> list[str]:
        """Symbols ranked within `max_rank` on `day`, best rank first."""
        frame = self.snapshot(day)
        if frame.empty:
            return []
        top = frame[frame["rank"] <= max_rank].sort_values("rank")
        return top["symbol"].tolist()

    def prefetch(self, start: datetime | date, stop: datetime | date) -> int:
        """
        Download and cache every daily snapshot in [start, stop].

        Run once offline so a backtest reads only the cache and makes no network
        call inside its candle loop, which keeps it deterministic.

        :return: number of days cached.
        """
        start = start.date() if isinstance(start, datetime) else start
        stop = stop.date() if isinstance(stop, datetime) else stop
        cached = 0
        for day in pd.date_range(start, stop, freq="D"):
            if not self.snapshot(day.date()).empty:
                cached += 1
        return cached

    def union_symbols(
        self, start: datetime | date, stop: datetime | date, max_rank: int, step_days: int = 7
    ) -> list[str]:
        """
        Every symbol that held a rank within `max_rank` on any sampled day in
        [start, stop], ordered by their best rank over the window.

        A fixed single-day top-N would silently drop coins that were large then
        and are small now, so a survivorship-free backtest needs the union over
        time as its candidate universe.
        """
        start = start.date() if isinstance(start, datetime) else start
        stop = stop.date() if isinstance(stop, datetime) else stop

        best_rank: dict[str, int] = {}
        for day in pd.date_range(start, stop, freq=f"{step_days}D"):
            try:
                frame = self.snapshot(day.date())
            except requests.RequestException as exc:
                logger.warning("CMC snapshot for %s failed, skipping: %s", day.date(), exc)
                continue
            if frame.empty:
                continue
            for row in frame[frame["rank"] <= max_rank].itertuples():
                if row.symbol not in best_rank or row.rank < best_rank[row.symbol]:
                    best_rank[row.symbol] = row.rank
        return [sym for sym, _ in sorted(best_rank.items(), key=lambda kv: kv[1])]

    def _download(self, day: date) -> pd.DataFrame:
        resp = self._session.get(
            _ENDPOINT,
            params={
                "date": day.isoformat(),
                "start": 1,
                "limit": self._limit,
                "convertId": _USD_ID,
            },
            headers=_HEADERS,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        # The endpoint returns HTTP 200 with a status error for logical failures
        # (bad date, throttling); a silent empty universe would otherwise result.
        error = (payload.get("status") or {}).get("error_message")
        if error and error not in ("SUCCESS", None):
            raise OperationalException(f"CoinMarketCap returned an error for {day}: {error}")
        data = payload.get("data")
        rows = data.get("data") if isinstance(data, dict) else data
        return pd.DataFrame([_normalize(row) for row in (rows or [])], columns=_COLUMNS)


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


def _normalize(row: dict) -> dict:
    # The quote's `name` echoes the requested convertId ("2781"), not "USD".
    usd = next((q for q in row.get("quotes", []) if q.get("name") in ("USD", str(_USD_ID))), {})
    supply = row.get("circulatingSupply")
    return {
        # CMC's stable numeric id, the only safe identity. Tickers get reused
        # (LUNA was Terra, then Terra 2.0); the id never is.
        "cmc_id": row.get("id"),
        "rank": row.get("cmcRank"),
        "symbol": row.get("symbol"),
        "name": row.get("name"),
        "market_cap": usd.get("marketCap"),
        "price": usd.get("price"),
        # Some coins report supplies above int64; store as float so the column
        # stays numeric and serializes to feather.
        "circulating_supply": float(supply) if supply is not None else None,
        "volume24h": usd.get("volume24h"),
    }
