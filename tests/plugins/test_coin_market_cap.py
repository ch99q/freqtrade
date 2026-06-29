"""
Locks the CoinMarketCap parser against a real captured response.

The other CMC tests (`test_marketcap_cmc.py`) build mock rows by hand, so a
change in CMC's real wire shape would slip past them. The payload below is
trimmed from one live call to the historical listings endpoint for 2021-01-03
(`convertId=2781`), keeping the real `data` list shape and the real `quotes`
structure where each quote is keyed by `"name": "2781"`, the convertId as a
string, not `"USD"`. If CMC ever keys the USD quote differently, the parser
returns all-None numeric fields and this test fails.
"""

from datetime import datetime
from unittest.mock import MagicMock

from freqtrade.util.coin_market_cap import FtCoinMarketCapApi, _normalize


# Three top coins from the live 2021-01-03 response, verbatim. The quote dict
# carries `"name": "2781"` (the convertId), which is what the parser matches on.
_REAL_ROWS = [
    {
        "id": 1,
        "name": "Bitcoin",
        "symbol": "BTC",
        "slug": "bitcoin",
        "cmcRank": 1,
        "circulatingSupply": 18589737,
        "totalSupply": 18589737,
        "maxSupply": 21000000,
        "quotes": [
            {
                "name": "2781",
                "price": 32782.02446581,
                "volume24h": 78665235201.84283,
                "marketCap": 609409213147.0334,
                "lastUpdated": "2021-01-03T00:00:00.000Z",
            }
        ],
    },
    {
        "id": 1027,
        "name": "Ethereum",
        "symbol": "ETH",
        "slug": "ethereum",
        "cmcRank": 2,
        "circulatingSupply": 114104674.624,
        "totalSupply": 114104674.624,
        "maxSupply": None,
        "quotes": [
            {
                "name": "2781",
                "price": 975.50767291,
                "volume24h": 45200463368.20847,
                "marketCap": 111309985610.90729,
                "lastUpdated": "2021-01-03T00:00:00.000Z",
            }
        ],
    },
    {
        "id": 825,
        "name": "Tether",
        "symbol": "USDT",
        "slug": "tether",
        "cmcRank": 3,
        "circulatingSupply": 21329136250.15833,
        "totalSupply": 23270442550.15833,
        "maxSupply": None,
        "quotes": [
            {
                "name": "2781",
                "price": 1.00051411,
                "volume24h": 120425679796.26683,
                "marketCap": 21340101768.041767,
                "lastUpdated": "2021-01-03T00:00:00.000Z",
            }
        ],
    },
]

_NUMERIC_COLUMNS = ["cmc_id", "market_cap", "price", "circulating_supply", "volume24h"]


def test_normalize_populates_numeric_fields_from_real_quote():
    # given a real row whose USD quote is keyed `"name": "2781"`, not "USD"
    out = _normalize(_REAL_ROWS[0])

    assert out["cmc_id"] == 1
    assert out["rank"] == 1
    assert out["symbol"] == "BTC"
    assert out["market_cap"] == 609409213147.0334
    assert out["price"] == 32782.02446581
    assert out["volume24h"] == 78665235201.84283
    assert out["circulating_supply"] == 18589737.0


def test_snapshot_parses_real_payload_without_none_fields(tmp_path):
    # The live endpoint returns `data` as a bare list, not a dict wrapping it.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={"data": _REAL_ROWS, "status": {"error_message": "SUCCESS"}}
    )
    session = MagicMock()
    session.get = MagicMock(return_value=resp)

    df = FtCoinMarketCapApi(tmp_path, session=session).snapshot(datetime(2021, 1, 3))

    assert list(df["symbol"]) == ["BTC", "ETH", "USDT"]
    for column in _NUMERIC_COLUMNS:
        assert not df[column].isnull().any(), f"{column} came back with None values"
    assert df["price"].iloc[2] == 1.00051411
    assert df["circulating_supply"].iloc[1] == 114104674.624
