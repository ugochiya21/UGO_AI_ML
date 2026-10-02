"""Strategy 1 - Multi-timeframe trend pullback (the core strategy).

Idea: trade WITH the daily trend, but only after the market has pulled back
to value on the 4-hour chart and the 1-hour chart shows buyers (or sellers)
stepping back in. Professional traders call this "buy the dip in an
up-trend". It wins less than half the time but each win pays 2.5x the risk.

Score out of 10 (8+ needed):
  2  daily trend aligned (price > EMA50 > EMA200, or the reverse)
  1  daily EMA50 sloping the same way
  2  4-hour EMA20 above/below EMA50 (swing structure agrees)
  1  4-hour ADX > 20 (trend has strength)
  2  pullback into the 4-hour EMA20 zone + 1-hour trigger candle  (required)
  1  1-hour RSI in a healthy range (not overbought/oversold)
  1  liquid trading hours
"""
import numpy as np
import pandas as pd

NAME = "trend_pullback"


def generate(f: pd.DataFrame, rr: float = 2.5, lookback: int = 6) -> pd.DataFrame:
    out = []
    for d in (1, -1):
        if d == 1:
            touched = (f["low"].rolling(lookback).min() <= f.h4_ema20 + 0.25 * f.h4_atr)
            intact = f["close"] > f.h4_ema50 - 0.5 * f.h4_atr
            trigger = (f["close"] > f["high"].shift(1)) & (f["close"] > f["open"])
            extreme = f["low"].rolling(lookback).min()
            rsi_ok = f.rsi.between(45, 70)
        else:
            touched = (f["high"].rolling(lookback).max() >= f.h4_ema20 - 0.25 * f.h4_atr)
            intact = f["close"] < f.h4_ema50 + 0.5 * f.h4_atr
            trigger = (f["close"] < f["low"].shift(1)) & (f["close"] < f["open"])
            extreme = f["high"].rolling(lookback).max()
            rsi_ok = f.rsi.between(30, 55)

        setup = touched & intact & trigger
        stop = extreme - d * 0.3 * f.atr
        # Stops must be sensible: at least 1 ATR (noise) and not absurdly wide.
        dist = (f["close"] - stop) * d
        stop = np.where(dist < f.atr, f["close"] - d * f.atr, stop)
        dist = (f["close"] - stop) * d
        sane = (dist > 0) & (dist <= 3 * f.h4_atr) & (f.atr_pct < 0.95)

        score = (2 * (f.bias == d) + (np.sign(f.d_ema50_slope) == d)
                 + 2 * (f.h4_trend == d) + (f.h4_adx > 20)
                 + 2 * setup + rsi_ok + f.liquid.astype(bool))
        ok = setup & sane & (f.bias == d)
        sig = pd.DataFrame({"direction": d, "stop": stop, "score": score.astype(int)},
                           index=f.index)[ok]
        # Works for both directions: target is rr times the stop distance away.
        entry = f.loc[sig.index, "close"]
        sig["take_profit"] = entry + rr * (entry - sig["stop"])
        out.append(sig)
    res = pd.concat(out).sort_index()
    res["strategy"] = NAME
    return res
