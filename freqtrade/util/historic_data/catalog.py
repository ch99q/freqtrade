"""
Identity-keyed historic OHLCV catalog.

A coin's exchange ticker is not a stable identity: `MATIC` became `POL`, and
`LUNA` was Terra and later a different coin entirely. CoinMarketCap's numeric id
is stable, so this catalog keys on it. It reads the daily snapshots, resolves
each id into the time-bounded `(symbol, range)` spans it traded under, downloads
each span from an exchange `Source`, and stores the result through freqtrade's
own data handler, so the files drop straight into a backtest.

The chain reads as instruction:

    catalog = create_catalog(datadir=..., stake_currency="USDT", cmc=..., source=...)
    report = catalog.universe(start, stop, max_rank=30).download()
"""

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pandas as pd

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, PairPrefixes
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.data.history.datahandlers.idatahandler import IDataHandler
from freqtrade.enums import CandleType
from freqtrade.util.coin_market_cap import FtCoinMarketCapApi
from freqtrade.util.historic_data.source import Source
from freqtrade.util.historic_data.verify import Verdict, verdict


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Span:
    """One contiguous range a coin traded under a single exchange symbol."""

    cmc_id: int
    symbol: str
    start: date
    stop: date


@dataclass(frozen=True)
class Identity:
    """A coin, keyed by its stable CoinMarketCap id, over a window."""

    cmc_id: int
    name: str
    spans: tuple[Span, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(span.symbol for span in self.spans))


@dataclass(frozen=True)
class Provenance:
    """Where one stored span came from, and how its identity was checked."""

    cmc_id: int
    pair: str
    symbol: str
    start: date
    stop: date
    candles: int
    source: str
    found: bool
    stored: bool = True
    verdict: Verdict | None = None


@dataclass(frozen=True)
class PairResult:
    """The outcome of storing one pair, possibly merged from several spans."""

    pair: str
    candles: int
    provenance: tuple[Provenance, ...]
    conflict: bool


@dataclass(frozen=True)
class Report:
    """The outcome of a universe download."""

    pairs: tuple[PairResult, ...]
    identities: tuple[Identity, ...]

    @property
    def candles(self) -> int:
        return sum(p.candles for p in self.pairs)

    @property
    def missing(self) -> tuple[Provenance, ...]:
        return tuple(pr for p in self.pairs for pr in p.provenance if not pr.found)

    @property
    def rejected(self) -> tuple[Provenance, ...]:
        """Spans that had data but failed price verification, so were not stored."""
        return tuple(pr for p in self.pairs for pr in p.provenance if pr.found and not pr.stored)

    @property
    def conflicts(self) -> tuple[PairResult, ...]:
        return tuple(p for p in self.pairs if p.conflict)


@dataclass(frozen=True)
class _Context:
    datadir: Path
    stake_currency: str
    timeframe: str
    candle_type: CandleType
    cmc: FtCoinMarketCapApi
    source: Source
    datahandler: IDataHandler
    symbol_map: Mapping[str, str]
    verify: bool


def create_catalog(
    *,
    datadir: Path,
    stake_currency: str,
    cmc: FtCoinMarketCapApi,
    source: Source,
    timeframe: str = "1d",
    candle_type: CandleType = CandleType.SPOT,
    data_format: str = "feather",
    symbol_map: Mapping[str, str] | None = None,
    verify: bool = True,
) -> "Catalog":
    """
    Open a catalog bound to one exchange data directory and source.

    :param datadir: freqtrade data directory the stored candles land in.
    :param stake_currency: quote currency of the pairs to build, e.g. "USDT".
    :param cmc: CoinMarketCap snapshot reader, the identity source.
    :param source: exchange OHLCV source the candles are downloaded from.
    :param verify: cross-check each candidate symbol against its CoinMarketCap
        price, so a reused or mismatched ticker is corrected or rejected, not
        stored as the wrong coin.
    """
    ctx = _Context(
        datadir=Path(_require(datadir, "datadir")),
        stake_currency=_require(stake_currency, "stake_currency").upper(),
        timeframe=_require(timeframe, "timeframe"),
        candle_type=candle_type,
        cmc=_require(cmc, "cmc"),
        source=_require(source, "source"),
        datahandler=get_datahandler(Path(datadir), data_format),
        symbol_map={k.upper(): v.upper() for k, v in (symbol_map or {}).items()},
        verify=verify,
    )
    return Catalog(ctx)


_VerdictCache = dict[tuple[int, str], Verdict]


@dataclass(frozen=True)
class Catalog:
    _ctx: _Context

    def universe(
        self,
        start: datetime | date,
        stop: datetime | date,
        *,
        max_rank: int = 30,
        step_days: int = 7,
    ) -> "Universe":
        """Narrow to the coins ranked within `max_rank` on any sampled day of the window."""
        start = _as_date(start)
        stop = _as_date(stop)
        identities = _resolve(self._ctx, start, stop, max_rank, step_days)
        return Universe(self._ctx, start, stop, tuple(identities))


@dataclass(frozen=True)
class Universe:
    _ctx: _Context
    start: date
    stop: date
    _identities: tuple[Identity, ...]

    def identities(self) -> tuple[Identity, ...]:
        """The resolved coins, each with its time-bounded symbol spans."""
        return self._identities

    def coin(self, cmc_id: int) -> "Coin":
        """Narrow to one coin by its CoinMarketCap id."""
        for identity in self._identities:
            if identity.cmc_id == cmc_id:
                return Coin(self._ctx, identity)
        raise KeyError(f"cmc_id {cmc_id} is not in this universe")

    def download(self) -> Report:
        """Resolve, verify, download, and store every coin's spans."""
        cache: _VerdictCache = {}
        markets = [
            _resolve_market(self._ctx, span, cache)
            for identity in self._identities
            for span in identity.spans
        ]
        return _report(self._ctx, markets, self._identities)


@dataclass(frozen=True)
class Coin:
    _ctx: _Context
    identity: Identity

    def spans(self) -> tuple[Span, ...]:
        """The contiguous `(symbol, range)` spans this coin traded under."""
        return self.identity.spans

    def download(self) -> tuple[PairResult, ...]:
        """Resolve, verify, download, and store this coin's spans."""
        cache: _VerdictCache = {}
        markets = [_resolve_market(self._ctx, span, cache) for span in self.identity.spans]
        return _report(self._ctx, markets, (self.identity,)).pairs


@dataclass(frozen=True)
class _Market:
    """One span resolved to an exchange market, with its verification verdict."""

    span: Span
    base: str
    pair: str
    frame: pd.DataFrame
    verdict: Verdict | None
    store: bool


# Verification samples the coin's price weekly and needs enough overlapping days
# to judge a stable ratio; a coin present across the window reaches this within a
# couple of months, so widening rarely costs more than a few extra snapshots.
_VERIFY_STEP = 7
_VERIFY_TARGET = 16
_VERIFY_MAX_DAYS = 365


def _resolve_market(ctx: _Context, span: Span, cache: "_VerdictCache") -> _Market:
    """
    Pick the exchange market for a span and verify its identity.

    The direct ticker is tried first. If it verifies, or there is too little
    price overlap to judge, it is trusted. Only a *checked* mismatch (data that
    exists but tracks a different coin) triggers a search of price-scaled
    candidates, and a mismatch with no verified alternative is left unstored.
    """
    bases = _candidates(ctx, span.symbol)
    direct = _fetch(ctx, span, bases[0], cache)
    if not direct.frame.empty and _trusted(direct.verdict):
        return replace(direct, store=True)

    for base in bases[1:]:
        candidate = _fetch(ctx, span, base, cache)
        if not candidate.frame.empty and _accepted(candidate.verdict):
            return replace(candidate, store=True)

    return replace(direct, store=False)


def _trusted(v: Verdict | None) -> bool:
    """A direct ticker is trusted when off, unverified, accepted, or unjudgeable."""
    return v is None or v.accepted or not v.checked


def _accepted(v: Verdict | None) -> bool:
    return v is not None and v.accepted


def _fetch(ctx: _Context, span: Span, base: str, cache: "_VerdictCache") -> _Market:
    frame = ctx.source.klines(base, ctx.stake_currency, ctx.timeframe, span.start, span.stop)
    decided = None
    if ctx.verify and not frame.empty:
        key = (span.cmc_id, base)
        if key not in cache:
            cache[key] = _verify(ctx, span, base)
        decided = cache[key]
    return _Market(span, base, f"{base}/{ctx.stake_currency}", frame, decided, store=False)


def _verify(ctx: _Context, span: Span, base: str) -> Verdict:
    """
    Verify the `cmc_id -> base` mapping over the coin's whole symbol span.

    Identity is a fact about the mapping, not about the backtest window, so the
    price series is widened out from the span on a cheap daily timeframe until it
    has enough points or runs into a symbol change. A short window then inherits a
    confident verdict instead of going unjudged.
    """
    prices = _widen_prices(ctx, span.cmc_id, span.symbol, _pivot(span))
    if not prices:
        return verdict({}, {})
    closes = _closes(ctx.source.klines(base, ctx.stake_currency, "1d", min(prices), max(prices)))
    return verdict(prices, closes)


def _widen_prices(ctx: _Context, cmc_id: int, symbol: str, pivot: date) -> dict[date, float]:
    """CMC price for `cmc_id` while it held `symbol`, sampled outward from `pivot`."""
    prices: dict[date, float] = {}
    seed = _price_at(ctx, cmc_id, symbol, pivot)
    if seed is not None:
        prices[pivot] = seed
    for direction in (-1, 1):
        for step in range(1, _VERIFY_MAX_DAYS // _VERIFY_STEP + 1):
            if len(prices) >= _VERIFY_TARGET:
                break
            day = pivot + timedelta(days=direction * _VERIFY_STEP * step)
            price = _price_at(ctx, cmc_id, symbol, day)
            if price is None:
                break  # the coin is absent or relabeled here: a span boundary
            prices[day] = price
    return prices


def _price_at(ctx: _Context, cmc_id: int, symbol: str, day: date) -> float | None:
    snapshot = ctx.cmc.snapshot(day)
    if snapshot.empty:
        return None
    row = snapshot[snapshot["cmc_id"] == cmc_id]
    if row.empty or row["symbol"].iloc[0] != symbol:
        return None
    price = row["price"].iloc[0]
    return float(price) if pd.notna(price) and price > 0 else None


def _pivot(span: Span) -> date:
    return span.start + (span.stop - span.start) / 2


def _report(ctx: _Context, markets: list[_Market], identities: tuple[Identity, ...]) -> Report:
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    spans: dict[str, list[Span]] = defaultdict(list)
    provenance: dict[str, list[Provenance]] = defaultdict(list)
    for market in markets:
        stored = market.store and not market.frame.empty
        spans[market.pair].append(market.span)
        provenance[market.pair].append(
            Provenance(
                cmc_id=market.span.cmc_id,
                pair=market.pair,
                symbol=market.base,
                start=market.span.start,
                stop=market.span.stop,
                candles=len(market.frame),
                source=ctx.source.name,
                found=not market.frame.empty,
                stored=stored,
                verdict=market.verdict,
            )
        )
        if stored:
            frames[market.pair].append(market.frame)

    results = tuple(
        PairResult(pair, _write(ctx, pair, frames.get(pair, [])), tuple(provenance[pair]),
                   _has_overlap(spans[pair]))
        for pair in provenance
    )
    _log_report(results)
    return Report(results, identities)


def _write(ctx: _Context, pair: str, frames: list[pd.DataFrame]) -> int:
    """Merge fetched frames with any stored data and write the pair once."""
    if not frames:
        return 0
    existing = ctx.datahandler.ohlcv_load(pair, ctx.timeframe, ctx.candle_type, warn_no_data=False)
    if not existing.empty:
        frames = [*frames, existing[DEFAULT_DATAFRAME_COLUMNS]]
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    ctx.datahandler.ohlcv_store(pair, ctx.timeframe, merged, ctx.candle_type)
    return len(merged)


def _candidates(ctx: _Context, symbol: str) -> list[str]:
    base = _base(ctx, symbol)
    return [base, *(f"{prefix}{base}" for prefix in PairPrefixes)]


def _closes(frame: pd.DataFrame) -> dict[date, float]:
    return dict(zip(frame["date"].dt.date, frame["close"], strict=False))


def _has_overlap(spans: list[Span]) -> bool:
    """True if two different coins claim the same symbol over overlapping dates."""
    ordered = sorted(spans, key=lambda s: s.start)
    for earlier, later in pairwise(ordered):
        if later.cmc_id != earlier.cmc_id and later.start <= earlier.stop:
            return True
    return False


def _resolve(
    ctx: _Context, start: date, stop: date, max_rank: int, step_days: int
) -> list[Identity]:
    member: set[int] = set()
    history: dict[int, list[tuple[date, str]]] = {}
    names: dict[int, str] = {}

    samples = [day.date() for day in pd.date_range(start, stop, freq=f"{step_days}D")]
    for day in samples:
        frame = ctx.cmc.snapshot(day)
        if frame.empty:
            continue
        for row in frame.itertuples():
            history.setdefault(row.cmc_id, []).append((day, row.symbol))
            names[row.cmc_id] = row.name
            if row.rank <= max_rank:
                member.add(row.cmc_id)

    last_sample = samples[-1] if samples else stop
    identities = []
    for cmc_id in member:
        spans = _spans(cmc_id, history[cmc_id], stop, last_sample)
        identities.append(Identity(cmc_id, names[cmc_id], tuple(spans)))
    return identities


def _spans(
    cmc_id: int, seen: list[tuple[date, str]], window_stop: date, last_sample: date
) -> list[Span]:
    """
    Compress a coin's `(day, symbol)` observations into contiguous spans.

    A span runs from where its symbol first appears to the day before the next
    symbol does, so a rename splits one coin into two adjacent pairs with no gap.
    Spans are bounded by what was actually observed, never stretched across the
    whole window, so a coin seen only early and a different coin that later reused
    its ticker do not overlap. The final span reaches `window_stop` only when the
    coin was still present at the last sampled day, so a coin present throughout
    is covered to the edge while a delisted one stops where it vanished.
    """
    ordered = sorted(seen)
    starts: list[tuple[date, str]] = []
    for day, symbol in ordered:
        if not starts or starts[-1][1] != symbol:
            starts.append((day, symbol))

    last_observed = ordered[-1][0]
    spans = []
    for index, (first_day, symbol) in enumerate(starts):
        if index < len(starts) - 1:
            span_stop = starts[index + 1][0] - timedelta(days=1)
        else:
            span_stop = window_stop if last_observed == last_sample else last_observed
        spans.append(Span(cmc_id, symbol, first_day, span_stop))
    return spans


def _base(ctx: _Context, symbol: str) -> str:
    return ctx.symbol_map.get(symbol.upper(), symbol).upper()


def _pair(ctx: _Context, symbol: str) -> str:
    return f"{_base(ctx, symbol)}/{ctx.stake_currency}"


def _log_report(results: tuple[PairResult, ...]) -> None:
    stored = [r for r in results if r.candles]
    logger.info(
        "Historic data: stored %d pairs (%d candles); %d pairs had no data.",
        len(stored),
        sum(r.candles for r in stored),
        len(results) - len(stored),
    )
    for result in results:
        if result.conflict:
            logger.warning(
                "Pair %s has overlapping spans from different coins; the merge may "
                "mix two coins. Inspect the provenance for %s.",
                result.pair,
                result.pair,
            )


def _as_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


def _require(value, name: str):
    if value is None or value == "":
        raise ValueError(f"{name} is required")
    return value
