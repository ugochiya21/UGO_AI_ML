"""Technical indicators (pure pandas) and look-ahead-safe multi-timeframe
alignment.

Bars are indexed by their OPEN time (UTC). A bar opened at t closes at
t + bar length; nothing in that bar may be used before then.
"""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    minus_di = 100 * pd.Series(minus_dm, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLC bars of a higher timeframe, indexed by the time they CLOSE."""
    out = df.resample(rule, label="right", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def align(htf: pd.Series | pd.DataFrame, ltf_index: pd.DatetimeIndex, bar=pd.Timedelta("1h")):
    """Value of a higher-timeframe series (indexed by close time) as known at
    the close of each lower-timeframe bar - never peeks at an unfinished bar."""
    closes = ltf_index + bar
    out = htf.reindex(htf.index.union(closes)).ffill().reindex(closes)
    out.index = ltf_index
    return out
