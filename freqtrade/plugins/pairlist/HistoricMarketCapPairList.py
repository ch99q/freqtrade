"""
Historic Market Cap PairList provider

Ranks pairs by CoinMarketCap's historical daily market-cap snapshots as of the
simulated time, so the traded universe changes across a backtest the way it did
in reality. Unlike the live `MarketCapPairList` it is free of survivorship and
lookahead bias and supports backtesting. With `--enable-dynamic-pairlist` the
whitelist is re-selected on each candle.

This handler is maintained outside upstream freqtrade; it is a standalone
pairlist so the fork stays a thin additive layer over upstream.
"""

import logging
from datetime import datetime, timedelta

from freqtrade.constants import PairPrefixes
from freqtrade.enums import RunMode
from freqtrade.enums.runmode import OPTIMIZE_MODES
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now
from freqtrade.util.coin_market_cap import FtCoinMarketCapApi


logger = logging.getLogger(__name__)


class HistoricMarketCapPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.YES

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets` not specified. Please check your configuration "
                'for "pairlist.config.number_assets"'
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_assets = self._pairlistconfig["number_assets"]
        self._max_rank = self._pairlistconfig.get("max_rank", 30)
        self._sample_days = self._pairlistconfig.get("sample_days", 7)
        self._symbol_map: dict[str, str] = {
            k.upper(): v.upper() for k, v in self._pairlistconfig.get("symbol_map", {}).items()
        }
        # The union is emitted once, to load OHLCV for the whole window. Afterwards
        # the slice date is set on every candle, so this guards the one in-loop
        # candle where the slice date is not set yet from selecting the union.
        self._union_loaded = False
        cache_dir = self._config["datadir"] / "marketcap_cmc"
        # The cache is keyed by date alone, so a fixed deep limit keeps the same
        # files reusable when max_rank changes between runs.
        self._cmc = FtCoinMarketCapApi(cache_dir, limit=max(1000, self._max_rank))

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._number_assets} pairs by historic market cap rank."

    @staticmethod
    def description() -> str:
        return "Provides pair list based on CoinMarketCap historical market cap rank."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist",
            },
            "max_rank": {
                "type": "number",
                "default": 30,
                "description": "Max rank of assets",
                "help": "Maximum market cap rank of assets to use from the pairlist",
            },
            "sample_days": {
                "type": "number",
                "default": 7,
                "description": "Universe sampling step (days)",
                "help": "Days between snapshots when building the historical universe.",
            },
        }

    def get_markets_exchange(self) -> list[str]:
        return list(
            self._exchange.get_markets(
                quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
            ).keys()
        )

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        # On dynamic-pairlist startup the slice date is not yet set; emit the whole
        # historical union so OHLCV is loaded for every coin the window can select.
        if self._is_startup_union():
            return self._union_pairlist()

        # Per-candle (and static backtest, and live): the ranking is time-dependent,
        # so it is not cached across candles.
        _pairlist = self.verify_blacklist(self.get_markets_exchange(), logger.info)
        return self.filter_pairlist(_pairlist, tickers)

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        marketcap_list = self._cmc.ranked_symbols(self._rank_date(), max_rank=self._max_rank)
        filtered_pairlist: list[str] = []
        if not marketcap_list:
            return filtered_pairlist

        markets = self.get_markets_exchange()
        for mc_pair in marketcap_list[: self._max_rank]:
            pair = f"{self._map_symbol(mc_pair).upper()}/{self._pair_format()}"
            resolved = self._resolve_pair(pair, pairlist, markets, filtered_pairlist)
            if resolved:
                filtered_pairlist.append(resolved)
                if len(filtered_pairlist) == self._number_assets:
                    break
        return filtered_pairlist

    def _resolve_pair(
        self, pair: str, pairlist: list[str], markets: list[str], filtered_pairlist: list[str]
    ) -> str | None:
        if pair in filtered_pairlist:
            return None
        if pair in pairlist:
            return pair
        if pair not in markets:
            for prefix in PairPrefixes:
                test_prefix = f"{prefix}{pair}"
                # Guard the resolved variant too: the plain-pair check above misses
                # a prefixed pair (e.g. 1000SHIB/USDT) already in the output list.
                if test_prefix in pairlist and test_prefix not in filtered_pairlist:
                    return test_prefix
        return None

    def _union_pairlist(self) -> list[str]:
        """Every exchange pair whose coin was within `max_rank` on any sampled day."""
        self._union_loaded = True
        start, stop = self._backtest_window()
        # Each candle ranks by the prior day's snapshot, so the first candle (at the
        # window start) needs the day before the window; prefetch and union from there.
        start = start - timedelta(days=1)
        # Cache every daily snapshot of the window up front (with retries), so the
        # per-candle ranking reads only the cache and the backtest makes no network
        # call inside its loop.
        self._cmc.prefetch(start, stop)
        symbols = self._cmc.union_symbols(start, stop, self._max_rank, self._sample_days)

        markets = self.get_markets_exchange()
        resolved: list[str] = []
        missing: list[str] = []
        for symbol in symbols:
            pair = f"{self._map_symbol(symbol).upper()}/{self._pair_format()}"
            match = self._resolve_pair(pair, markets, markets, resolved)
            if match:
                resolved.append(match)
            else:
                missing.append(symbol)

        if missing:
            self.logger.info(
                "CMC historical universe: %d of %d top-%d symbols have no tradable "
                "%s market and are excluded from the backtest: %s",
                len(missing),
                len(symbols),
                self._max_rank,
                self._stake_currency,
                ", ".join(missing),
            )
        return self.verify_blacklist(resolved, logger.info)

    def _rank_date(self) -> datetime:
        """
        The snapshot date to rank a candle by: the prior day of the simulated time.

        A CMC daily snapshot for date D is an end-of-day capture, so ranking a
        candle on D by snapshot(D) would let it see up to a day of same-day future
        ranking. Ranking by the prior day's snapshot uses only data that was final
        before the candle, which keeps the selection strictly bias-free.
        """
        return self._point_in_time() - timedelta(days=1)

    def _point_in_time(self) -> datetime:
        """The time whose ranking to read: the candle now, the window start, or today."""
        slice_date = self._current_time()
        if slice_date is not None:
            return slice_date
        if self._is_optimize():
            return self._backtest_window()[0]
        return dt_now()

    def _current_time(self) -> datetime | None:
        dataprovider = getattr(self._pairlistmanager, "_dataprovider", None)
        return dataprovider.slice_date if dataprovider is not None else None

    def _is_optimize(self) -> bool:
        return RunMode(self._config.get("runmode", RunMode.OTHER)) in OPTIMIZE_MODES

    def _is_startup_union(self) -> bool:
        return (
            not self._union_loaded
            and self._is_optimize()
            and self._config.get("enable_dynamic_pairlist", False)
            and self._current_time() is None
        )

    def _backtest_window(self) -> tuple[datetime, datetime]:
        from freqtrade.configuration import TimeRange

        timerange_str = self._config.get("timerange")
        timerange = TimeRange.parse_timerange(None if timerange_str is None else str(timerange_str))
        if timerange.startdt is None or timerange.stopdt is None:
            self.logger.warning(
                "HistoricMarketCapPairList needs an explicit `--timerange` to build the "
                "historical universe; without one it collapses to a single day."
            )
        start = timerange.startdt or dt_now()
        stop = timerange.stopdt or dt_now()
        return start, stop

    def _pair_format(self) -> str:
        stake = self._stake_currency.upper()
        is_futures = self._exchange._config["trading_mode"] == "futures"
        return f"{stake}:{stake}" if is_futures else stake

    def _map_symbol(self, symbol: str) -> str:
        return self._symbol_map.get(symbol.upper(), symbol)
