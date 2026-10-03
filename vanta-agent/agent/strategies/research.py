"""Two intraday ideas from published research, fixed BEFORE any backtest
(no tuning). Both run on 5-minute bars, are 1:2, and close by 16:00 New
York time if neither the stop nor the TP is hit.

A. late_momentum - S&P 500 and Nasdaq. "Market intraday momentum" (Gao,
   Han, Li & Zhou, Journal of Financial Economics, 2018): the return from
   the previous close to 10:00 New York time predicts the direction of the
   last half hour. At 15:30 trade in that direction; stop 1 x 15M ATR,
   TP 2R, exit 16:00.

B. nr7_orb - S&P 500, Nasdaq, gold, silver, oil. Toby Crabel's narrow-range
   idea: after the quietest day of the last 7 (NR7), trade the first
   5-minute close outside today's 09:30-09:45 New York opening range,
   until 12:00. Stop: the other side of the range (at least 0.5 x 15M ATR).
   TP 2R, exit 16:00.

Every value is what a trader would have known at the bar's close.
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, resample
from agent.instruments import COMMODITIES, INDICES, Instrument

BAR = pd.Timedelta("5min")
NY = "America/New_York"
LATE_MOMENTUM = {"SP500USDC", "XYZ100USDC"}
NR7 = {"SP500USDC", "XYZ100USDC", "GOLDUSDC", "SILVERUSDC", "WTIOILUSDC"}


def _ny_clock(idx: pd.DatetimeIndex):
    """New York date and minutes-after-midnight at each 5M bar's CLOSE."""
    close = (idx + BAR).tz_convert(NY)
    return pd.Series(close.date, index=idx), pd.Series(close.hour * 60 + close.minute, index=idx)


def _exit_16(day: pd.Series) -> pd.Series:
    """16:00 New York time on each bar's New York date, in UTC."""
    ts = pd.to_datetime(day.astype(str) + " 16:00").dt.tz_localize(NY).dt.tz_convert("UTC")
    return ts.set_axis(day.index)


def _out(f, ok, d, stop, exit_at, rr, name):
    sig = pd.DataFrame({"direction": d, "stop": stop, "score": 10, "exit_at": exit_at},
                       index=f.index)[ok]
    entry = f.loc[sig.index, "close"]
    sig["take_profit"] = entry + rr * (entry - sig["stop"])
    sig["strategy"] = name
    return sig


def late_momentum(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
                  min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.symbol not in LATE_MOMENTUM:
        return pd.DataFrame()
    f = df[["open", "high", "low", "close"]].copy()
    m15_atr = align(atr(resample(df, "15min")), f.index, BAR)
    day, mins = _ny_clock(f.index)
    close = f["close"]
    # Price at 16:00 (previous session close) and at 10:00, per New York date.
    c16 = close[mins == 16 * 60].groupby(day[mins == 16 * 60]).last()
    c10 = close[mins == 10 * 60].groupby(day[mins == 10 * 60]).last()
    prev16 = c16.shift(1)
    first = (c10 / prev16.reindex(c10.index) - 1).dropna()
    at1530 = mins == 15 * 60 + 30
    ret = day.map(first).where(at1530)
    d_all = np.sign(ret).fillna(0).astype(int)
    floor = np.maximum(m15_atr, close * min_stop_frac)
    out = []
    for d in (1, -1):
        ok = (d_all == d) & floor.notna() & (floor > 0)
        out.append(_out(f, ok, d, close - d * floor, _exit_16(day), rr, "late_momentum"))
    return pd.concat(out).sort_index()


def nr7_orb(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
            min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.symbol not in NR7:
        return pd.DataFrame()
    f = df[["open", "high", "low", "close"]].copy()
    m15_atr = align(atr(resample(df, "15min")), f.index, BAR)
    day, mins = _ny_clock(f.index)
    # Daily range per New York date; NR7 = yesterday was the narrowest of 7.
    rng = (f["high"].groupby(day).max() - f["low"].groupby(day).min())
    rng = rng[rng > 0]
    nr7_yesterday = (rng == rng.rolling(7).min()).shift(1).fillna(False).astype(bool)
    today_ok = day.map(nr7_yesterday).fillna(False).astype(bool)
    # Opening range: bars closing 09:35..09:45 New York.
    in_rng = (mins > 9 * 60 + 30) & (mins <= 9 * 60 + 45)
    hi = f["high"].where(in_rng).groupby(day).transform("max")
    lo = f["low"].where(in_rng).groupby(day).transform("min")
    window = (mins > 9 * 60 + 45) & (mins <= 12 * 60)
    up = window & (f["close"] > hi)
    dn = window & (f["close"] < lo)
    any_brk = (up | dn)
    first = any_brk & (any_brk.astype(int).groupby(day).cumsum() == 1)
    out = []
    for d, brk, other in ((1, up, lo), (-1, dn, hi)):
        stop = other.copy()
        floor = np.maximum(0.5 * m15_atr, f["close"] * min_stop_frac)
        dist = (f["close"] - stop) * d
        stop = np.where(dist < floor, f["close"] - d * floor, stop)
        dist = (f["close"] - stop) * d
        ok = first & brk & today_ok & (dist > 0) & (dist <= 4 * m15_atr)
        out.append(_out(f, ok, d, stop, _exit_16(day), rr, "nr7_orb"))
    return pd.concat(out).sort_index()


STRATEGIES = {"late_momentum": late_momentum, "nr7_orb": nr7_orb}
