"""Tests for the CoinMarketCap historical source of MarketCapPairList."""

from datetime import datetime
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import SupportsBacktesting
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.util.coin_market_cap import FtCoinMarketCapApi
from tests.conftest import EXMS, get_patched_exchange, log_has_re


def _response(rows: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": rows, "status": {"error_message": "SUCCESS"}})
    return resp


def _row(rank: int, symbol: str, market_cap: float) -> dict:
    return {
        "id": rank,
        "name": symbol,
        "symbol": symbol,
        "cmcRank": rank,
        "circulatingSupply": 1000,
        "totalSupply": 1000,
        "quotes": [{"name": "2781", "price": 1.0, "marketCap": market_cap, "volume24h": 1.0}],
    }


def test_cmc_api_snapshot_caches_to_disk(tmp_path):
    session = MagicMock()
    session.get = MagicMock(return_value=_response([_row(1, "BTC", 9), _row(2, "ETH", 8)]))

    api = FtCoinMarketCapApi(tmp_path, session=session)
    first = api.snapshot(datetime(2021, 1, 3))
    assert list(first["symbol"]) == ["BTC", "ETH"]
    assert session.get.call_count == 1
    assert (tmp_path / "2021-01-03.feather").is_file()

    # given a fresh client over the same dir / when reading the same day
    # then the snapshot comes from disk, with no second network call
    fresh = FtCoinMarketCapApi(tmp_path, session=MagicMock())
    again = fresh.snapshot(datetime(2021, 1, 3))
    assert list(again["symbol"]) == ["BTC", "ETH"]


def test_cmc_api_carries_cmc_id(tmp_path):
    session = MagicMock()
    rows = [{**_row(1, "LUNA", 9), "id": 4172}]
    session.get = MagicMock(return_value=_response(rows))
    df = FtCoinMarketCapApi(tmp_path, session=session).snapshot(datetime(2022, 1, 1))
    assert "cmc_id" in df.columns
    assert int(df["cmc_id"].iloc[0]) == 4172


def test_cmc_api_supply_above_int64_serializes(tmp_path):
    # a supply larger than int64 must not break feather caching
    row = {**_row(1, "SHIB", 9), "circulatingSupply": 589_000_000_000_000_000_000}
    session = MagicMock()
    session.get = MagicMock(return_value=_response([row]))
    api = FtCoinMarketCapApi(tmp_path, session=session)
    df = api.snapshot(datetime(2021, 1, 3))
    assert (tmp_path / "2021-01-03.feather").is_file()
    assert df["circulating_supply"].iloc[0] == 5.89e20


def test_cmc_api_corrupt_cache_redownloads(tmp_path):
    session = MagicMock()
    session.get = MagicMock(return_value=_response([_row(1, "BTC", 9)]))
    (tmp_path / "2021-01-03.feather").write_bytes(b"not a feather file")
    df = FtCoinMarketCapApi(tmp_path, session=session).snapshot(datetime(2021, 1, 3))
    assert list(df["symbol"]) == ["BTC"]
    assert session.get.call_count == 1


def test_cmc_api_prefetch_then_offline(tmp_path):
    snapshots = {
        "2021-01-01": [_row(1, "BTC", 9)],
        "2021-01-02": [_row(1, "BTC", 9)],
        "2021-01-03": [_row(1, "BTC", 9)],
    }
    session = MagicMock()
    session.get = MagicMock(
        side_effect=lambda url, params, headers, timeout: _response(snapshots[params["date"]])
    )
    api = FtCoinMarketCapApi(tmp_path, session=session)
    assert api.prefetch(datetime(2021, 1, 1), datetime(2021, 1, 3)) == 3

    # a fresh client with a dead session reads only the cache, no network
    offline = FtCoinMarketCapApi(
        tmp_path, session=MagicMock(get=MagicMock(side_effect=AssertionError))
    )
    assert offline.ranked_symbols(datetime(2021, 1, 2), max_rank=5) == ["BTC"]


def test_cmc_api_ranked_and_union(tmp_path):
    snapshots = {
        "2021-01-03": [_row(1, "BTC", 9), _row(2, "ETH", 8), _row(3, "XRP", 7)],
        "2021-01-10": [_row(1, "BTC", 9), _row(2, "SOL", 8), _row(3, "ETH", 7)],
    }

    def fake_get(url, params, headers, timeout):
        return _response(snapshots[params["date"]])

    session = MagicMock()
    session.get = MagicMock(side_effect=fake_get)
    api = FtCoinMarketCapApi(tmp_path, session=session)

    assert api.ranked_symbols(datetime(2021, 1, 3), max_rank=2) == ["BTC", "ETH"]
    # union over both weeks within top-3, ordered by best rank
    union = api.union_symbols(datetime(2021, 1, 3), datetime(2021, 1, 10), max_rank=3, step_days=7)
    assert union == ["BTC", "ETH", "SOL", "XRP"]


@pytest.fixture
def cmc_conf(default_conf_usdt, tmp_path):
    default_conf_usdt["datadir"] = tmp_path
    default_conf_usdt["trading_mode"] = "spot"
    default_conf_usdt["exchange"]["pair_whitelist"].extend(["BTC/USDT", "ETH/USDT", "XRP/USDT"])
    return default_conf_usdt


def test_marketcap_cmc_flips_supports_backtesting(mocker, cmc_conf, markets):
    cmc_conf["pairlists"] = [{"method": "HistoricMarketCapPairList", "number_assets": 3}]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    exchange = get_patched_exchange(mocker, cmc_conf)
    pm = PairListManager(exchange, cmc_conf)

    handler = pm._pairlist_handlers[0]
    assert handler.supports_backtesting == SupportsBacktesting.YES


def test_marketcap_cmc_point_in_time(mocker, cmc_conf, markets):
    cmc_conf["pairlists"] = [{"method": "HistoricMarketCapPairList", "number_assets": 2}]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.ranked_symbols",
        return_value=["BTC", "ETH", "XRP"],
    )
    exchange = get_patched_exchange(mocker, cmc_conf)
    pm = PairListManager(exchange, cmc_conf)
    pm.refresh_pairlist()
    assert pm.whitelist == ["BTC/USDT", "ETH/USDT"]


def test_marketcap_cmc_symbol_map(mocker, cmc_conf, markets):
    cmc_conf["pairlists"] = [
        {
            "method": "HistoricMarketCapPairList",
            "number_assets": 1,
            "symbol_map": {"MIOTA": "XRP"},
        }
    ]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.ranked_symbols",
        return_value=["MIOTA"],
    )
    exchange = get_patched_exchange(mocker, cmc_conf)
    pm = PairListManager(exchange, cmc_conf)
    pm.refresh_pairlist()
    assert pm.whitelist == ["XRP/USDT"]


def test_marketcap_cmc_union_and_gap_report(mocker, cmc_conf, markets, caplog):
    cmc_conf["runmode"] = RunMode.BACKTEST
    cmc_conf["enable_dynamic_pairlist"] = True
    cmc_conf["timerange"] = "20210101-20210201"
    cmc_conf["pairlists"] = [
        {"method": "HistoricMarketCapPairList", "number_assets": 2, "max_rank": 10}
    ]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    union_mock = mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.union_symbols",
        return_value=["BTC", "ETH", "XRP", "DEADCOIN"],
    )
    prefetch_mock = mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.prefetch", return_value=0
    )
    exchange = get_patched_exchange(mocker, cmc_conf)
    pm = PairListManager(exchange, cmc_conf)
    pm.refresh_pairlist()

    # the union is loaded in full (not capped at number_assets) so data exists for
    # every coin the per-candle ranking can later choose
    assert pm.whitelist == ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    assert union_mock.called
    # the window is prefetched up front so the candle loop stays offline
    assert prefetch_mock.called
    assert log_has_re(
        r"CMC historical universe: 1 of 4 top-10 symbols have no tradable USDT "
        r"market.*DEADCOIN",
        caplog,
    )


def test_marketcap_cmc_inloop_candle_is_not_the_union(mocker, cmc_conf, markets):
    # The union is returned only once, to load data. The first in-loop candle (slice
    # date still None) must narrow to the point-in-time ranking, never the union, or
    # a coin that only enters the top-N later could be traded on the first candle.
    cmc_conf["runmode"] = RunMode.BACKTEST
    cmc_conf["enable_dynamic_pairlist"] = True
    cmc_conf["timerange"] = "20210101-20210201"
    cmc_conf["pairlists"] = [
        {"method": "HistoricMarketCapPairList", "number_assets": 3, "max_rank": 10}
    ]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    # XRP is in the window's union but not in the top-N as of the window start.
    mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.union_symbols",
        return_value=["BTC", "ETH", "XRP"],
    )
    mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.ranked_symbols",
        return_value=["BTC", "ETH"],
    )
    mocker.patch("freqtrade.util.coin_market_cap.FtCoinMarketCapApi.prefetch", return_value=0)
    exchange = get_patched_exchange(mocker, cmc_conf)
    pm = PairListManager(exchange, cmc_conf)

    pm.refresh_pairlist()  # startup: union loads data for BTC, ETH, XRP
    assert "XRP/USDT" in pm.whitelist
    pm.refresh_pairlist()  # first in-loop candle: ranks as of window start, drops XRP
    assert pm.whitelist == ["BTC/USDT", "ETH/USDT"]


def test_marketcap_cmc_resolve_pair_no_duplicate(mocker, cmc_conf, markets):
    # given the same resolved 1000-prefixed pair offered twice
    # when resolving against a list that already contains it
    # then the second resolution returns None, so the output never duplicates it
    cmc_conf["pairlists"] = [{"method": "HistoricMarketCapPairList", "number_assets": 2}]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )
    exchange = get_patched_exchange(mocker, cmc_conf)
    handler = PairListManager(exchange, cmc_conf)._pairlist_handlers[0]

    pairlist = ["1000SHIB/USDT"]
    out: list[str] = []
    first = handler._resolve_pair("SHIB/USDT", pairlist, [], out)
    assert first == "1000SHIB/USDT"
    out.append(first)
    second = handler._resolve_pair("SHIB/USDT", pairlist, [], out)
    assert second is None
    assert out == ["1000SHIB/USDT"]


def test_marketcap_cmc_per_candle_uses_prior_day(mocker, cmc_conf, markets):
    # The whitelist on a candle for date D must equal the top-N ranking as of D-1,
    # and no snapshot at or after the simulated clock may ever be read.
    from freqtrade.data.dataprovider import DataProvider

    cmc_conf["runmode"] = RunMode.BACKTEST
    cmc_conf["pairlists"] = [
        {"method": "HistoricMarketCapPairList", "number_assets": 2, "max_rank": 10}
    ]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )

    rankings = {
        datetime(2021, 1, 9).date(): ["BTC", "ETH"],
        datetime(2021, 1, 10).date(): ["ETH", "BTC", "XRP"],
    }
    requested: list = []

    def fake_ranked(day, max_rank):
        day = day.date() if isinstance(day, datetime) else day
        requested.append(day)
        return rankings[day]

    mocker.patch(
        "freqtrade.util.coin_market_cap.FtCoinMarketCapApi.ranked_symbols", side_effect=fake_ranked
    )

    exchange = get_patched_exchange(mocker, cmc_conf)
    dataprovider = DataProvider(cmc_conf, exchange)
    pm = PairListManager(exchange, cmc_conf, dataprovider)

    simulated = datetime(2021, 1, 10)
    dataprovider._set_dataframe_max_date(simulated)
    pm.refresh_pairlist()

    # ranked as of the prior day (Jan 9), so the Jan 9 ordering wins
    assert pm.whitelist == ["BTC/USDT", "ETH/USDT"]
    assert all(day < simulated.date() for day in requested)


def test_marketcap_cmc_per_candle_offline_from_cache(mocker, cmc_conf, markets, tmp_path):
    # The per-candle ranking must be served entirely from the prefetched cache, so a
    # network failure inside the candle loop must not surface.
    from freqtrade.data.dataprovider import DataProvider

    cmc_conf["runmode"] = RunMode.BACKTEST
    cmc_conf["pairlists"] = [
        {"method": "HistoricMarketCapPairList", "number_assets": 2, "max_rank": 10}
    ]
    mocker.patch.multiple(
        EXMS, markets=PropertyMock(return_value=markets), exchange_has=MagicMock(return_value=True)
    )

    snapshots = {
        "2021-01-08": [_row(1, "BTC", 9), _row(2, "ETH", 8)],
        "2021-01-09": [_row(1, "BTC", 9), _row(2, "ETH", 8)],
    }
    working = MagicMock()
    working.get = MagicMock(
        side_effect=lambda url, params, headers, timeout: _response(snapshots[params["date"]])
    )
    api = FtCoinMarketCapApi(tmp_path, session=working)
    assert api.prefetch(datetime(2021, 1, 8), datetime(2021, 1, 9)) == 2

    exchange = get_patched_exchange(mocker, cmc_conf)
    dataprovider = DataProvider(cmc_conf, exchange)
    pm = PairListManager(exchange, cmc_conf, dataprovider)
    handler = pm._pairlist_handlers[0]
    # swap the live-built client for the prefetched one, then break the network
    handler._cmc = api
    api._session = MagicMock(get=MagicMock(side_effect=AssertionError("network used in loop")))

    dataprovider._set_dataframe_max_date(datetime(2021, 1, 9))
    pm.refresh_pairlist()
    assert pm.whitelist == ["BTC/USDT", "ETH/USDT"]


def test_cmc_api_status_error_raises(tmp_path):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": None, "status": {"error_message": "Invalid date"}})
    session = MagicMock(get=MagicMock(return_value=resp))
    with pytest.raises(OperationalException, match=r"CoinMarketCap returned an error"):
        FtCoinMarketCapApi(tmp_path, session=session).snapshot(datetime(2021, 1, 3))
