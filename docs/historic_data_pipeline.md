# Historic data pipeline

A survivorship-free backtest needs OHLCV for coins the exchange no longer lists,
identified correctly across renames and ticker reuse. freqtrade's `download-data`
goes through the exchange's current market list, so a delisted coin (FTT, Terra)
is silently skipped. This pipeline fills that gap.

It is a fork-only addition under `freqtrade.util.historic_data`, built from new
files so it stays a thin additive layer over upstream.

## The identity problem

A coin's exchange ticker is not a stable identity, in two ways:

- A coin **renames**: Polygon was `MATIC`, then `POL`. Same coin, two tickers.
- A ticker is **reused**: `LUNA` was Terra, which collapsed, and later a
  different coin took the ticker. One ticker, two coins.

CoinMarketCap's numeric id is stable through both. The pipeline keys on it: it
reads the daily snapshots, resolves each id into the time-bounded `(symbol,
range)` spans it traded under, and downloads each span. A rename becomes two
adjacent pairs; reuse becomes two disjoint date ranges under one pair, never
mixed.

## Usage

```python
from datetime import date
from pathlib import Path

from freqtrade.util.coin_market_cap import FtCoinMarketCapApi
from freqtrade.util.historic_data import BinanceVisionSource, create_catalog

cmc = FtCoinMarketCapApi(Path("user_data/data/binance/marketcap_cmc"))
source = BinanceVisionSource(Path("user_data/data/binance/_archive"))

catalog = create_catalog(
    datadir=Path("user_data/data/binance"),
    stake_currency="USDT",
    cmc=cmc,
    source=source,
    timeframe="1h",
)

report = catalog.universe(date(2022, 1, 1), date(2023, 1, 1), max_rank=30).download()
print(report.candles, "candles across", len(report.pairs), "pairs")
for gap in report.missing:
    print("no archive data for", gap.symbol)
```

The stored feather files land in `datadir` in freqtrade's own format, so a
backtest reads them with no extra step.

## The chain

The API narrows from the whole catalog to one coin:

```python
catalog.universe(start, stop, max_rank=30)   # the coins ranked in the window
        .identities()                         # read: each coin and its spans
        .coin(cmc_id)                         # narrow to one coin
        .download()                           # fetch and store its spans
```

`universe(...).download()` is the same effect over every coin at once, merging
any that share a pair.

## Sources

A `Source` fetches candles for an exchange symbol over a date range. The pipeline
ships `BinanceVisionSource`, which reads the public data.binance.vision archive,
including delisted and rebranded pairs. The archive publishes whole months on a
lag, so the trailing days come from daily files; coverage reaches roughly
yesterday. Parsed months are cached to disk, so a re-run is offline.

To add another exchange, implement one method:

```python
class MySource(Source):
    name = "my-exchange"

    def klines(self, symbol, timeframe, start, stop):
        ...  # return a DataFrame [date, open, high, low, close, volume], or empty
```

## Verifying identity by price

A ticker can point at the wrong coin: an exchange reuses `LUNA` for a second
coin, or trades a token only as a price-scaled `1000SHIB` contract. With
`verify=True` (the default) the catalog cross-checks each candidate symbol
against the coin's CoinMarketCap price over their overlap before storing it.

The deciding signal is **ratio stability**, not correlation. All of crypto trends
together, so two different coins correlate highly; only the same coin holds a
near-constant price ratio. A stable ratio also reveals the scale, so a `1000x`
contract is recognised and stored under its real symbol with no hardcoding.

The resolution per span:

- The direct ticker verifies, or there is too little overlap to judge, so it is
  trusted.
- The direct ticker has data but tracks a different coin, so price-scaled
  candidates are searched and the verified one is used.
- Nothing verifies, so the data is left unstored and recorded in
  `report.rejected`, never written as the wrong coin.

Identity is a property of the mapping, not of your backtest window, so the price
series is **widened out from the span** until it has enough points or hits a
symbol change, on a cheap daily timeframe. A one-week backtest of a coin still
inherits a verdict from months of its price history, and the widening stops at a
rename so it never blends two coins. The verdict is cached per `(cmc_id, symbol)`,
so it is computed once and reused. Only a coin with too little history on either
side, or no exchange data to compare, stays unjudged and trusts its ticker. Pass
`verify=False` to store by ticker alone.

## Provenance and gaps

Every span carries provenance: which cmc_id, which symbol, which date range, how
many candles, from which source, and its verification `verdict` (accepted, scale,
ratio stability). `report.missing` lists spans with no archive data, and
`report.conflicts` flags any pair where two different coins claim the same symbol
over overlapping dates, so a questionable merge is visible rather than silent.
