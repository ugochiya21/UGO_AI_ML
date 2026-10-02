"""The tradable universe, with Vanta symbols and everything the risk engine
needs to know about each instrument.

Symbol ids, leverage caps and fees come from taoshidev/vanta-network
(vali_objects/trade_pair.py). Check `GET /trade-pairs` on your account
before going live: Vanta enables and disables pairs over time.
"""
from dataclasses import dataclass, field

FOREX = "forex"
CRYPTO = "crypto"
COMMODITIES = "commodities"
INDICES = "indices"
EQUITIES = "equities"

# Vanta transaction (spread) fee as a fraction of order value, per side.
FEE_RATE = {CRYPTO: 0.0003, EQUITIES: 0.0001, COMMODITIES: 0.00005, FOREX: 0.0, INDICES: 0.0}

# Carry fee charged per day on notional (approximation of Vanta's rates;
# Hyperliquid-sourced pairs pay live funding instead, which varies).
DAILY_CARRY_RATE = {CRYPTO: 0.0003, FOREX: 0.03 / 365, INDICES: 0.0525 / 365,
                    COMMODITIES: 0.05 / 365, EQUITIES: 0.066 / 365}


@dataclass(frozen=True)
class Instrument:
    symbol: str                 # Vanta trade_pair id
    asset_class: str
    news_currencies: tuple      # calendar currencies that move this market
    # Exposure of a LONG position. A short flips every sign. Two open trades
    # that share a (key, sign) are the same bet and are not stacked.
    exposure: dict = field(default_factory=dict)
    max_leverage: float = 1.0   # per-position cap for subaccounts
    data_source: str = ""       # "dukascopy" | "binance" | "yahoo" | "csv"
    data_symbol: str = ""
    trades_weekend: bool = False
    # Only for stocks: US regular session (UTC hours, approx, ignores DST).
    session_utc: tuple | None = None

    @property
    def fee_rate(self) -> float:
        return FEE_RATE[self.asset_class]

    @property
    def daily_carry(self) -> float:
        return DAILY_CARRY_RATE[self.asset_class]

    def signed_exposure(self, direction: int) -> set:
        return {(k, v * direction) for k, v in self.exposure.items()}


def _fx(sym, base, quote, duka=None):
    return Instrument(sym, FOREX, (base, quote), {base: 1, quote: -1}, 2.5,
                      "dukascopy", duka or sym)


UNIVERSE = {i.symbol: i for i in [
    # Forex majors and the two most liquid yen crosses
    _fx("EURUSD", "EUR", "USD"), _fx("GBPUSD", "GBP", "USD"),
    _fx("USDJPY", "USD", "JPY"), _fx("AUDUSD", "AUD", "USD"),
    _fx("USDCAD", "USD", "CAD"), _fx("USDCHF", "USD", "CHF"),
    _fx("NZDUSD", "NZD", "USD"), _fx("EURJPY", "EUR", "JPY"),
    _fx("GBPJPY", "GBP", "JPY"),
    # Commodities (Hyperliquid perps on Vanta)
    Instrument("GOLDUSDC", COMMODITIES, ("USD",), {"GOLD": 1, "USD": -1}, 1.0, "dukascopy", "XAUUSD"),
    Instrument("SILVERUSDC", COMMODITIES, ("USD",), {"GOLD": 1, "USD": -1}, 0.5, "dukascopy", "XAGUSD"),
    Instrument("WTIOILUSDC", COMMODITIES, ("USD", "CAD"), {"OIL": 1}, 0.5, "dukascopy", "LIGHTCMDUSD"),
    # Indices (Hyperliquid perps; the old SPX/NDX/DJI pairs are disabled)
    Instrument("SP500USDC", INDICES, ("USD",), {"US_EQUITY": 1}, 1.5, "dukascopy", "USA500IDXUSD"),
    Instrument("XYZ100USDC", INDICES, ("USD",), {"US_EQUITY": 1}, 1.5, "dukascopy", "USATECHIDXUSD"),
    # Crypto
    Instrument("BTCUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 0.5, "binance", "BTCUSDT", True),
    Instrument("ETHUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 0.5, "binance", "ETHUSDT", True),
    Instrument("SOLUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 0.5, "binance", "SOLUSDT", True),
    # A few of the most liquid US stocks
    *[Instrument(t, EQUITIES, ("USD",), {"US_EQUITY": 1, f"STOCK_{t}": 1}, 0.5,
                 "yahoo", t, False, (14, 21))
      for t in ("NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA")],
]}


def get(symbol: str) -> Instrument:
    return UNIVERSE[symbol]
