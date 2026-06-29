import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType
from freqtrade.util.historic_data import (
    BinanceVisionSource,
    Source,
    create_catalog,
    verdict,
)


# --- Source: Binance-vision archive parsing ----------------------------------


def _kline_zip(rows: list[list], header: bool = False) -> bytes:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume")
    lines += [",".join(str(c) for c in r) for r in rows]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("klines.csv", "\n".join(lines))
    return payload.getvalue()


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def _resp(content: bytes, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def test_source_parses_and_filters(tmp_path):
    rows = [[_ms(date(2022, 1, d)), 1.0 + d, 2.0, 0.5, 1.5, 100.0] for d in range(1, 6)]
    session = MagicMock(get=MagicMock(return_value=_resp(_kline_zip(rows))))
    src = BinanceVisionSource(tmp_path, session=session)

    out = src.klines("FTT", "USDT", "1d", date(2022, 1, 2), date(2022, 1, 4))

    assert list(out.columns) == DEFAULT_DATAFRAME_COLUMNS
    assert out["date"].dt.date.tolist() == [date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4)]
    assert out["open"].iloc[0] == pytest.approx(3.0)


def test_source_caches_to_disk(tmp_path):
    rows = [[_ms(date(2022, 1, 1)), 1.0, 2.0, 0.5, 1.5, 100.0]]
    session = MagicMock(get=MagicMock(return_value=_resp(_kline_zip(rows))))
    src = BinanceVisionSource(tmp_path, session=session)

    src.klines("FTT", "USDT", "1d", date(2022, 1, 1), date(2022, 1, 1))
    calls_after_first = session.get.call_count
    src.klines("FTT", "USDT", "1d", date(2022, 1, 1), date(2022, 1, 1))

    # the month is served from the feather cache, so no second network call
    assert session.get.call_count == calls_after_first


def test_source_404_past_month_fails_fast(tmp_path):
    session = MagicMock(get=MagicMock(return_value=_resp(b"", status=404)))
    src = BinanceVisionSource(tmp_path, session=session, today=date(2026, 1, 1))

    out = src.klines("DEAD", "USDT", "1d", date(2022, 1, 1), date(2022, 1, 31))
    assert out.empty
    assert list(out.columns) == DEFAULT_DATAFRAME_COLUMNS
    # a past month with no monthly file does not fan out into 31 daily requests
    assert session.get.call_count == 1


def test_source_recent_month_uses_daily_fallback(tmp_path):
    # the monthly file 404s (not published yet); the day is served from daily
    day = date(2025, 6, 2)
    daily_zip = _kline_zip([[_ms(day), 1.0, 2.0, 0.5, 1.5, 100.0]])

    def get(url, timeout):
        return _resp(daily_zip) if "/daily/" in url else _resp(b"", status=404)

    session = MagicMock(get=MagicMock(side_effect=get))
    src = BinanceVisionSource(tmp_path, session=session, today=date(2025, 6, 15))

    out = src.klines("BTC", "USDT", "1d", day, day)
    assert out["date"].dt.date.tolist() == [day]


def test_source_drops_header_and_detects_microseconds(tmp_path):
    micros = _ms(date(2025, 6, 1)) * 1000
    rows = [[micros, 1.0, 2.0, 0.5, 1.5, 100.0]]
    session = MagicMock(get=MagicMock(return_value=_resp(_kline_zip(rows, header=True))))
    src = BinanceVisionSource(tmp_path, session=session)

    out = src.klines("BTC", "USDT", "1d", date(2025, 6, 1), date(2025, 6, 1))
    assert out["date"].dt.date.tolist() == [date(2025, 6, 1)]


# --- Catalog: identity resolution and storage --------------------------------


def _snapshot(rows: list[tuple]) -> pd.DataFrame:
    # rows: (cmc_id, rank, symbol, name)
    return pd.DataFrame(
        [
            {"cmc_id": c, "rank": r, "symbol": s, "name": n, "market_cap": 0, "price": 0}
            for c, r, s, n in rows
        ]
    )


class FakeCmc:
    def __init__(self, by_day: dict[date, pd.DataFrame]):
        self._by_day = by_day

    def snapshot(self, day):
        day = day.date() if isinstance(day, datetime) else day
        return self._by_day.get(day, pd.DataFrame())


class FakeSource(Source):
    name = "fake"

    def __init__(self, missing: set[str] | None = None):
        self.missing = missing or set()
        self.calls: list[tuple] = []

    def klines(self, base, quote, timeframe, start, stop):
        self.calls.append((base, quote, start, stop))
        if base in self.missing:
            return pd.DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)
        days = pd.date_range(start, stop, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "date": days,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        )


def _catalog(tmp_path, by_day, source):
    return create_catalog(
        datadir=tmp_path,
        stake_currency="USDT",
        cmc=FakeCmc(by_day),
        source=source,
        timeframe="1d",
    )


def test_universe_membership_and_rename(tmp_path):
    by_day = {
        date(2022, 1, 1): _snapshot([(3890, 5, "MATIC", "Polygon"), (1, 1, "BTC", "Bitcoin")]),
        date(2022, 8, 1): _snapshot([(3890, 5, "POL", "Polygon"), (1, 1, "BTC", "Bitcoin")]),
    }
    cat = _catalog(tmp_path, by_day, FakeSource())
    universe = cat.universe(date(2022, 1, 1), date(2022, 12, 31), max_rank=10, step_days=1)

    polygon = universe.coin(3890)
    assert polygon.identity.symbols == ("MATIC", "POL")
    spans = polygon.spans()
    # the rename splits into two adjacent spans, contiguous across the boundary
    assert [s.symbol for s in spans] == ["MATIC", "POL"]
    assert spans[0].stop + timedelta(days=1) == spans[1].start


def test_download_stores_loadable_data(tmp_path):
    by_day = {date(2022, 1, 1): _snapshot([(1, 1, "BTC", "Bitcoin")])}
    cat = _catalog(tmp_path, by_day, FakeSource())
    report = cat.universe(date(2022, 1, 1), date(2022, 1, 10), max_rank=10, step_days=1).download()

    assert report.candles > 0
    handler = get_datahandler(tmp_path, "feather")
    loaded = handler.ohlcv_load("BTC/USDT", "1d", CandleType.SPOT, warn_no_data=False)
    assert not loaded.empty
    assert loaded["date"].dt.date.min() == date(2022, 1, 1)


def test_ticker_reuse_merges_disjoint_without_conflict(tmp_path):
    # Terra (4172) is LUNA until mid-2022; Luna 2.0 (20314) is LUNA after. Disjoint
    # in time, so they merge into one LUNA/USDT file with no conflict flagged.
    by_day = {
        date(2022, 1, 1): _snapshot([(4172, 5, "LUNA", "Terra")]),
        date(2022, 11, 1): _snapshot([(20314, 5, "LUNA", "Terra 2.0")]),
    }
    cat = _catalog(tmp_path, by_day, FakeSource())
    report = cat.universe(date(2022, 1, 1), date(2022, 12, 31), max_rank=10, step_days=1).download()

    luna = [p for p in report.pairs if p.pair == "LUNA/USDT"]
    assert len(luna) == 1
    assert luna[0].conflict is False
    assert {pr.cmc_id for pr in luna[0].provenance} == {4172, 20314}


def test_missing_data_is_a_recorded_gap(tmp_path):
    by_day = {date(2022, 1, 1): _snapshot([(999, 5, "DEAD", "Defunct")])}
    cat = _catalog(tmp_path, by_day, FakeSource(missing={"DEAD"}))
    report = cat.universe(date(2022, 1, 1), date(2022, 1, 10), max_rank=10, step_days=1).download()

    assert report.candles == 0
    assert [pr.symbol for pr in report.missing] == ["DEAD"]
    handler = get_datahandler(tmp_path, "feather")
    assert handler.ohlcv_load("DEAD/USDT", "1d", CandleType.SPOT, warn_no_data=False).empty


def test_rank_cutoff_excludes_lower_ranked(tmp_path):
    by_day = {date(2022, 1, 1): _snapshot([(1, 1, "BTC", "Bitcoin"), (50, 50, "FOO", "Foo")])}
    cat = _catalog(tmp_path, by_day, FakeSource())
    universe = cat.universe(date(2022, 1, 1), date(2022, 1, 2), max_rank=10)
    assert {i.cmc_id for i in universe.identities()} == {1}


# --- Verification: identity by price overlap ---------------------------------


def _days(n: int, start: date = date(2022, 1, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def test_verdict_accepts_matching_series():
    days = _days(20)
    cmc = {d: 10.0 + i for i, d in enumerate(days)}
    exchange = {d: (10.0 + i) * 1.001 for i, d in enumerate(days)}  # USDT vs USD drift
    v = verdict(cmc, exchange)
    assert v.accepted and v.checked
    assert abs(v.scale - 1.0) < 0.01 and v.cv < 0.01


def test_verdict_accepts_scaled_series_and_reports_scale():
    days = _days(20)
    cmc = {d: 0.00001 * (1 + 0.01 * i) for i, d in enumerate(days)}
    exchange = {d: cmc[d] * 1000 for d in days}  # a 1000x contract
    v = verdict(cmc, exchange)
    assert v.accepted and abs(v.scale - 1000) < 1 and v.cv < 0.01


def test_verdict_rejects_different_coin():
    days = _days(20)
    cmc = {d: 0.0001 * (1 + 0.004 * i) for i, d in enumerate(days)}  # Terra Classic, tiny
    exchange = {d: 2.0 + (i % 4) for i, d in enumerate(days)}  # Luna 2.0, unrelated
    v = verdict(cmc, exchange)
    assert v.checked and not v.accepted


def test_verdict_unchecked_when_too_few_points():
    days = _days(3)
    series = {d: 10.0 for d in days}
    v = verdict(series, series)
    assert not v.checked and not v.accepted


def _priced(rows: list[tuple]) -> pd.DataFrame:
    # rows: (cmc_id, rank, symbol, name, price)
    return pd.DataFrame(
        [
            {"cmc_id": c, "rank": r, "symbol": s, "name": n, "market_cap": 0, "price": p}
            for c, r, s, n, p in rows
        ]
    )


class PriceSource(Source):
    name = "price-fake"

    def __init__(self, closes: dict[str, dict[date, float]]):
        self.closes = closes

    def klines(self, base, quote, timeframe, start, stop):
        series = self.closes.get(base, {})
        days = sorted(d for d in series if start <= d <= stop)
        if not days:
            return pd.DataFrame(columns=DEFAULT_DATAFRAME_COLUMNS)
        vals = [series[d] for d in days]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(days).tz_localize("UTC"),
                "open": vals,
                "high": vals,
                "low": vals,
                "close": vals,
                "volume": 1.0,
            }
        )


# Verification widens the price series weekly out from a span, so these use a
# multi-month history (not a few days) to give the verdict enough points.
def _priced_history(cmc_id, symbol, name, price_fn, start=date(2022, 1, 1), n=140):
    days = [start + timedelta(days=i) for i in range(n)]
    by_day = {d: _priced([(cmc_id, 5, symbol, name, price_fn(i))]) for i, d in enumerate(days)}
    return days, by_day


def _run(tmp_path, by_day, closes, days):
    cat = create_catalog(
        datadir=tmp_path, stake_currency="USDT", cmc=FakeCmc(by_day),
        source=PriceSource(closes), timeframe="1d",
    )
    return cat.universe(days[0], days[-1], max_rank=10).download()


def test_collision_is_rejected_not_stored(tmp_path):
    # The exchange's LUNA market tracks Luna 2.0 (~$2), but CMC's id 4172 is Terra
    # Classic (~$0.0001). The price mismatch rejects it, and nothing is stored.
    days, by_day = _priced_history(
        4172, "LUNA", "Terra Classic", lambda i: 0.0001 * (1 + 0.004 * i)
    )
    closes = {"LUNA": {d: 2.0 + (i % 4) for i, d in enumerate(days)}}
    report = _run(tmp_path, by_day, closes, days)

    luna = next(p for p in report.pairs if p.pair == "LUNA/USDT")
    assert luna.candles == 0
    assert [pr.cmc_id for pr in report.rejected] == [4172]
    handler = get_datahandler(tmp_path, "feather")
    assert handler.ohlcv_load("LUNA/USDT", "1d", CandleType.SPOT, warn_no_data=False).empty


def test_match_is_stored_with_scale_in_provenance(tmp_path):
    days, by_day = _priced_history(1, "BTC", "Bitcoin", lambda i: 10000.0 + 50 * i)
    closes = {"BTC": {d: (10000.0 + 50 * i) * 1.001 for i, d in enumerate(days)}}
    report = _run(tmp_path, by_day, closes, days)

    btc = next(p for p in report.pairs if p.pair == "BTC/USDT")
    assert btc.candles > 0
    assert btc.provenance[0].verdict.accepted
    assert abs(btc.provenance[0].verdict.scale - 1.0) < 0.01


def test_scaled_candidate_is_discovered(tmp_path):
    # The coin trades only as a 1000x contract; the direct ticker has no market.
    days, by_day = _priced_history(5994, "SHIB", "Shiba Inu", lambda i: 0.00001 * (1 + 0.01 * i))
    closes = {"1000SHIB": {d: 0.00001 * (1 + 0.01 * i) * 1000 for i, d in enumerate(days)}}
    report = _run(tmp_path, by_day, closes, days)

    stored = next(p for p in report.pairs if p.candles > 0)
    assert stored.pair == "1000SHIB/USDT"
    assert abs(stored.provenance[0].verdict.scale - 1000) < 1
