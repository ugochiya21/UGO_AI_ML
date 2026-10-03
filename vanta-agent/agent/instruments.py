"""The tradable universe, with Vanta symbols and everything the risk engine
needs to know about each instrument.

Symbol ids and fees come from taoshidev/vanta-network
(vali_objects/trade_pair.py). Leverage caps are Vanta's published Base
buying power for Classic accounts (vantatrading.io/rules, Oct 2026):
per pair, per asset class, and for the whole portfolio. Check `GET /trade-pairs` on your account
before going live: Vanta enables and disables pairs over time.
"""
from dataclasses import dataclass, field

FOREX = "forex"
CRYPTO = "crypto"
COMMODITIES = "commodities"
INDICES = "indices"
EQUITIES = "equities"

# Buying power (gross notional / equity), Base level. Longs and shorts add up.
CLASS_LIMIT = {CRYPTO: 1.5, FOREX: 10.0, COMMODITIES: 1.5, INDICES: 3.0, EQUITIES: 1.0}
PORTFOLIO_LIMIT = 15.0
# Forex pairs with an NZD leg are capped lower than the other 22 pairs.
NZD_CROSSES = {"AUDNZD", "EURNZD", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY"}

# Vanta transaction fee as a fraction of order value, per side.
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
    return Instrument(sym, FOREX, (base, quote), {base: 1, quote: -1},
                      5.0 if sym in NZD_CROSSES else 10.0, "dukascopy", duka or sym)


UNIVERSE = {i.symbol: i for i in [
    # Forex majors and the two most liquid yen crosses
    _fx("EURUSD", "EUR", "USD"), _fx("GBPUSD", "GBP", "USD"),
    _fx("USDJPY", "USD", "JPY"), _fx("AUDUSD", "AUD", "USD"),
    _fx("USDCAD", "USD", "CAD"), _fx("USDCHF", "USD", "CHF"),
    _fx("NZDUSD", "NZD", "USD"), _fx("EURJPY", "EUR", "JPY"),
    _fx("GBPJPY", "GBP", "JPY"),
    # Crosses - all enabled on Vanta (vali_objects/trade_pair.py)
    *[_fx(b + q, b, q) for b, q in [
        ("AUD", "CAD"), ("AUD", "CHF"), ("AUD", "JPY"), ("AUD", "NZD"),
        ("CAD", "CHF"), ("CAD", "JPY"), ("CHF", "JPY"), ("EUR", "AUD"), ("EUR", "CAD"),
        ("EUR", "CHF"), ("EUR", "GBP"), ("EUR", "NZD"), ("GBP", "AUD"), ("GBP", "CAD"),
        ("GBP", "CHF"), ("GBP", "NZD"), ("NZD", "CAD"), ("NZD", "CHF"), ("NZD", "JPY")]],
    # Commodities (Hyperliquid perps on Vanta)
    Instrument("GOLDUSDC", COMMODITIES, ("USD",), {"GOLD": 1, "USD": -1}, 1.5, "dukascopy", "XAUUSD"),
    Instrument("SILVERUSDC", COMMODITIES, ("USD",), {"GOLD": 1, "USD": -1}, 1.5, "dukascopy", "XAGUSD"),
    Instrument("WTIOILUSDC", COMMODITIES, ("USD", "CAD"), {"OIL": 1}, 1.5, "dukascopy", "LIGHTCMDUSD"),
    # Indices (Hyperliquid perps; the old SPX/NDX/DJI pairs are disabled)
    Instrument("SP500USDC", INDICES, ("USD",), {"US_EQUITY": 1}, 2.5, "dukascopy", "USA500IDXUSD"),
    Instrument("XYZ100USDC", INDICES, ("USD",), {"US_EQUITY": 1}, 2.5, "dukascopy", "USATECHIDXUSD"),
    # Crypto
    Instrument("BTCUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 1.5, "binance", "BTCUSDT", True),
    Instrument("ETHUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 1.5, "binance", "ETHUSDT", True),
    Instrument("SOLUSDC", CRYPTO, ("USD",), {"CRYPTO": 1}, 1.5, "binance", "SOLUSDT", True),
    # More of Vanta's 34 crypto perps (all on Bitstamp except XMR, PUMP).
    # Per-pair buying power: XRP, DOGE 1.5x; the other alts 0.5x.
    # PEPE/SHIB trade on Vanta as kPEPE/kSHIB (per 1,000 coins) - same moves.
    *[Instrument(f"{c}USDC", CRYPTO, ("USD",), {"CRYPTO": 1},
                 1.5 if c in ("XRP", "DOGE") else 0.5, "binance", f"{c}USDT", True)
      for c in ("XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK", "DOT", "TRX", "LTC", "BCH",
                "TAO", "SUI", "ARB", "NEAR", "ALGO", "ASTER", "UNI", "AAVE", "CRV", "HYPE",
                "ZEC", "ENA", "ZRO", "WLD", "PEPE", "HBAR", "XLM", "SHIB", "CC")],
    # A few of the most liquid US stocks
    *[Instrument(t, EQUITIES, ("USD",), {"US_EQUITY": 1, f"STOCK_{t}": 1}, 0.5,
                 "yahoo", t, False, (14, 21))
      for t in ("NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA")],
]}


def get(symbol: str) -> Instrument:
    return UNIVERSE[symbol]
