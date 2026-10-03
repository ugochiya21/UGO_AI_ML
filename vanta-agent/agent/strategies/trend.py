"""Claude's own strategy (Oct 2026): daily trend following, entered on the
hourly chart. Owner's only constraints: 1:2 on every trade, total open risk
never above 1%.

Why: every intraday test lost because Vanta's spread + slippage + fees took
7-17% of the risk on each trade (small stops). Stops of 1.5 x DAILY ATR
make those costs ~2% of the risk. Trend following (time-series momentum)
is the most widely documented pattern across asset classes.

Rules (fixed before any test, no tuning):
  Trend filter   long only if the daily close > 100-day EMA; short only below
  Entry 1        breakout: an hourly close above the highest high of the
                 previous 20 days (below the lowest low for shorts)
  Entry 2        pullback: daily EMA20 > EMA50 > EMA100 (mirror for shorts),
                 price dipped to the daily EMA20 within the last 6 hours and
                 an hourly candle closes back above the previous hour's high
  Stop           1.5 x daily ATR(20) from entry.  TP: 2 x the stop.
Runs on 1-hour bars; daily values are those of finished days only.
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, ema, resample
from agent.instruments import Instrument

BAR = pd.Timedelta("1h")


def generate(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
             min_stop_frac: float = 0.0) -> pd.DataFrame:
    f = df[["open", "high", "low", "close"]].copy()
    d = resample(df, "1D")
    d = d[(d["high"] - d["low"]) > 0]            # drop empty weekend "days"
    e20, e50, e100 = ema(d["close"], 20), ema(d["close"], 50), ema(d["close"], 100)
    a20 = atr(d, 20)
    hh20 = d["high"].rolling(20).max()
    ll20 = d["low"].rolling(20).min()
    for name, s in {"d_close": d["close"], "e20": e20, "e50": e50, "e100": e100,
                    "d_atr": a20, "hh20": hh20, "ll20": ll20}.items():
        f[name] = align(s, f.index, BAR)

    c = f["close"]
    out = []
    for dirn in (1, -1):
        trend = (f.d_close > f.e100) if dirn == 1 else (f.d_close < f.e100)
        if dirn == 1:
            breakout = (c > f.hh20) & (c.shift(1) <= f.hh20)
            stacked = (f.e20 > f.e50) & (f.e50 > f.e100)
            dipped = f["low"].rolling(6).min() <= f.e20
            resume = (c > f["high"].shift(1)) & (c > f.e20)
        else:
            breakout = (c < f.ll20) & (c.shift(1) >= f.ll20)
            stacked = (f.e20 < f.e50) & (f.e50 < f.e100)
            dipped = f["high"].rolling(6).max() >= f.e20
            resume = (c < f["low"].shift(1)) & (c < f.e20)
        pullback = stacked & dipped & resume
        dist = np.maximum(1.5 * f.d_atr, c * min_stop_frac)
        ok = trend & (breakout | pullback) & dist.notna() & (dist > 0)
        sig = pd.DataFrame({"direction": dirn, "stop": c - dirn * dist, "score": 10},
                           index=f.index)[ok]
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = np.where(breakout[ok], "trend_breakout", "trend_pullback_d")
        out.append(sig)
    return pd.concat(out).sort_index()
