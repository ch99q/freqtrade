# HistoricMarketCapPairList

A fork-only pairlist that ranks pairs by **CoinMarketCap historical market cap as
of the simulated time**. The built-in `MarketCapPairList` uses today's live
ranking, so in backtesting every candle sees the same present-day universe, which
is biased. `HistoricMarketCapPairList` instead reads the ranking that was true on
each backtest day, so the traded universe evolves the way it actually did, free of
survivorship and lookahead bias.

It is a standalone handler kept outside upstream freqtrade, so the fork stays a
thin additive layer.

## Config

```json
"pairlists": [
    {
        "method": "HistoricMarketCapPairList",
        "number_assets": 20,
        "max_rank": 30,
        "sample_days": 7
    }
]
```

- `number_assets`: how many pairs the whitelist holds.
- `max_rank`: the market-cap rank cutoff to select from.
- `sample_days`: days between snapshots when building the historical universe at
  backtest start. The default 7 keeps the up-front download small; the per-candle
  ranking is always daily regardless.

Run a backtest with `--enable-dynamic-pairlist` so the whitelist is re-selected on
each candle:

```bash
freqtrade backtesting -c config.json --timerange 20230601-20260531 \
  --enable-dynamic-pairlist
```

## How it stays bias-free

- **Time-correct.** Each candle ranks by the CoinMarketCap snapshot of the prior
  day, so it never sees same-day or future ranking.
- **Survivorship-free.** At startup it loads OHLCV for every coin that was within
  `max_rank` on any sampled day of the window, including coins later delisted, then
  per candle narrows to that day's top `number_assets`.
- **Offline and deterministic.** Snapshots are fetched and cached up front, so the
  candle loop makes no network call.
- Coins with no tradable market on your exchange are logged once and excluded, so
  the residual gap is explicit.

## Data note

You still download OHLCV yourself, as with any pairlist. The endpoint serves any
date back to 2013, no API key. A coin's exchange ticker is not a stable identity
(for example `LUNA` was Terra, then Terra 2.0); for pristine delisted-coin data a
CoinMarketCap-id-keyed download pipeline is the next step, tracked separately.
